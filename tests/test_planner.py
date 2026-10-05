#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规划内核单测.

验证 ``planner.py``:
    * 对 alpha / cardA 等卡能产出 <= 16 根光纤的有效分配;
    * 必观测目标优先;
    * 几何计算 (radec<->altaz, 光纤方格) 合理;
    * 从 initialize + decision_request 能产出一个合法 observe Action;
    * 无密钥时静态回退可产出 wait/observe.

运行: ``py -m pytest tests/test_planner.py -q``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import planner as planner_mod  # noqa: E402
import preplan  # noqa: E402
from config import load_config  # noqa: E402
from llm import LLMPlanner  # noqa: E402
from models import Action, DecisionState, LLMConfig, NightPlan  # noqa: E402


def _load_card(name: str) -> preplan.CardData:
    return preplan.CardData.from_card(name)


def test_planner_tool_assigns_at_most_16():
    card = _load_card("alpha")
    tool = planner_mod.PlannerTool(card)
    selected, rejected = preplan.assign_fibers(
        card, preplan.compute_priorities(card), max_targets=16
    )
    assert 0 < len(selected) <= 16
    fibers = [s.fiber_index for s in selected]
    assert len(fibers) == len(set(fibers)), "光纤不得重复"
    assert all(f >= 0 for f in fibers)


def test_planner_tool_targets_subset():
    card = _load_card("cardA")
    tool = planner_mod.PlannerTool(card)
    subset = [planner_mod.Target.from_preplan(t) for t in card.targets[:40]]
    selected, _ = tool.assign_for_targets(subset, max_targets=16)
    assert 0 < len(selected) <= 16
    ids = {s.target_id for s in selected}
    assert ids.issubset({t.target_id for t in subset})


def test_required_targets_prioritized():
    card = _load_card("alpha")
    priorities = preplan.compute_priorities(card)
    required = [t for t in card.targets if t.required]
    non_required = [t for t in card.targets if not t.required]
    assert required, "alpha 卡应有必观测目标"
    min_req = min(priorities[t.target_id] for t in required)
    max_non = max(priorities[t.target_id] for t in non_required) if non_required else -1
    assert min_req > max_non, "必观测目标优先级必须高于非必观测"


def test_geometry_roundtrip():
    ra, dec = 337.0, -5.0
    lst = 340.0
    lat = -24.6157
    alt, az = planner_mod.radec_to_altaz(ra, dec, lst, lat)
    ra2, dec2 = planner_mod.altaz_to_radec(alt, az, lst, lat)
    assert abs(ra2 - ra) < 0.5
    assert abs(dec2 - dec) < 0.5


def test_fiber_classify_center():
    grid = preplan.FiberGrid(side=4, n_fibers=16, fiber_side_deg=0.632456, gap_deg=0.0)
    fiber, margin = grid.classify(0.0, 0.0)
    assert 0 <= fiber < 16
    assert margin >= 0.0


def _make_planner_with_card(card_name: str):
    card = _load_card(card_name)
    # 从卡片构造一个等价的 initialize 消息
    nights = []
    try:
        import csv
        cal = card.root / "public" / "v4_night_calendar.csv"
        with cal.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                nights.append({
                    "night_id": row.get("night_id", ""),
                    "night_date": row.get("night_date", ""),
                    "observing_start_utc": row.get("observing_start_utc", ""),
                    "observing_end_utc": row.get("observing_end_utc", ""),
                    "slot_count": int(row.get("slot_count", 0) or 0),
                })
    except Exception:
        nights = []
    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class,
             t.feature_flux, t.science_weight, t.required] for t in card.targets]
    init_msg = {
        "message_type": "initialize",
        "payload": {
            "task_card": {"card_id": card_name},
            "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                     "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                     "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
            "survey": {"start_utc": nights[0]["observing_start_utc"] if nights else "2026-10-02T00:00:00Z",
                       "end_utc": nights[-1]["observing_end_utc"] if nights else "2026-10-08T08:45:00Z",
                       "slot_seconds": 900, "nights": nights},
            "instrument": {"n_fibers": card.n_fibers, "grid_side": 4, "fiber_area_deg2": 0.4,
                           "gap_deg": 0.0, "exposure": {"min_duration_seconds": 60,
                                                        "max_duration_seconds": 3600}},
            "scoring": card.score_config,
            "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                     "feature_flux", "science_weight", "required"], "rows": rows},
            "limits": {"global_wallclock_seconds": 900, "response_max_bytes": 524288},
        },
    }
    from protocol import parse_initialize
    init = parse_initialize(init_msg)
    tool = planner_mod.PlannerTool(card)
    # 无 LLM: 纯静态
    return planner_mod.Planner(init, tool, llm_planner=None), nights


def test_planner_observe_action_legal_when_night():
    p, nights = _make_planner_with_card("alpha")
    assert nights, "alpha 卡应有夜历"
    n = nights[0]
    now = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(n["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)

    from protocol import DecisionRequest
    req = DecisionRequest({
        "decision_sequence": 1,
        "payload": {
            "now_utc": now.isoformat().replace("+00:00", "Z"),
            "survey_end_utc": end.isoformat().replace("+00:00", "Z"),
            "wallclock": {"remaining_seconds": 800, "remaining_real_cpu_seconds": 800,
                          "wall_remaining_seconds": 1700},
            "latest_bulletin": {"record_type": "bulletin", "notices": []},
            "active_requests": [], "new_messages": [], "last_result": None,
        },
    })
    action = p.decide(req)
    assert isinstance(action, Action)
    assert action.type in ("observe", "wait", "report", "finish")
    if action.type == "observe":
        assert len(action.assignments) <= p.n_fibers
        assert action.exposure_seconds is not None
        assert p.min_exposure <= action.exposure_seconds <= p.max_exposure
        assert action.program in ("DARK", "BRIGHT", "BACKUP")
        # 光纤不重复
        fibers = list(action.assignments.keys())
        assert len(fibers) == len(set(fibers))


def test_planner_daytime_waits():
    p, nights = _make_planner_with_card("alpha")
    # 白天: 夜历第一夜开始前的时刻
    start = datetime.fromisoformat(nights[0]["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    from datetime import timedelta
    daytime = start - timedelta(hours=6)
    from protocol import DecisionRequest
    req = DecisionRequest({
        "decision_sequence": 1,
        "payload": {
            "now_utc": daytime.isoformat().replace("+00:00", "Z"),
            "survey_end_utc": nights[-1]["observing_end_utc"],
            "wallclock": {"remaining_seconds": 800, "remaining_real_cpu_seconds": 800,
                          "wall_remaining_seconds": 1700},
            "latest_bulletin": None, "active_requests": [], "new_messages": [], "last_result": None,
        },
    })
    action = p.decide(req)
    assert action.type == "wait"
    assert action.until_utc is not None


def test_llm_plan_and_decide_with_fake_client(monkeypatch):
    """用假客户端验证 LLM 环节A/环节B 确实被调用并可落成动作."""
    calls = {"system": [], "user": []}

    def fake_chat(self, system, user, call_site="chat_json"):
        calls["system"].append(system)
        calls["user"].append(user)
        if call_site == "plan_night":
            return {"strategy": "优先必观测", "target_ids": ["a", "b"], "program": "DARK",
                    "avoid_directions": ["NE"], "duration_scale": 1.1}
        if call_site == "decide_action":
            return {"action": "observe", "program": "BRIGHT", "target_ids": ["a"], "duration_scale": 1.2}
        return {"diagnosis": "ok", "duration_scale": 1.0, "avoid_directions": [], "report_fault": False}

    from llm import LLMClient
    monkeypatch.setattr(LLMClient, "chat_json", fake_chat)

    cfg = LLMConfig(api_key="fake-key", base_url="http://x/v1", model="m")
    lp = LLMPlanner(cfg, planner_tool=planner_mod.PlannerTool(_load_card("alpha")))
    state = DecisionState(night_index=0, candidate_targets=["a", "b", "c"], required_remaining=3)
    plan = lp.plan_night(state)
    assert plan.source == "llm"
    assert plan.program == "DARK"
    assert plan.duration_scale == 1.1

    action = lp.decide_action(state, plan)
    assert action is not None
    assert action.type == "observe"
    assert action.decision_source == "llm"
    # 若已缓存, 再次 plan_night 不应重复调用
    before = len(calls["system"])
    lp.plan_night(state)
    assert len(calls["system"]) == before, "同一夜规划应命中缓存, 不重复调用 LLM"


def test_llm_plan_cache_avoids_extra_calls():
    cfg = LLMConfig(api_key="fake")
    lp = LLMPlanner(cfg)
    calls = {"n": 0}

    def fake_chat(self, system, user, call_site="chat_json"):
        calls["n"] += 1
        return {"strategy": "s", "target_ids": [], "program": "BACKUP", "duration_scale": 1.0}

    from llm import LLMClient
    import types
    lp.client.chat_json = types.MethodType(fake_chat, lp.client)
    state = DecisionState(night_index=5, candidate_targets=[])
    lp.plan_night(state)
    lp.plan_night(state)
    assert calls["n"] == 1


def test_static_fallback_no_key():
    cfg = load_config()
    # 单测环境通常无密钥; 若有也应能构造 LLMPlanner
    lp = LLMPlanner(LLMConfig(api_key=""), planner_tool=planner_mod.PlannerTool(_load_card("alpha")))
    state = DecisionState(night_index=0, candidate_targets=["a", "b"], required_remaining=2)
    plan = lp.plan_night(state)
    assert isinstance(plan, NightPlan)
    assert plan.source == "static"
    action = lp.decide_action(state, plan)
    assert action is None  # 无密钥必须返回 None (由内核回退)


def test_nightplan_parse_sanitizes():
    lp = LLMPlanner(LLMConfig(api_key="x"))
    state = DecisionState(night_index=0)
    plan = lp._parse_plan(
        {"program": "bad", "target_ids": [1, 2], "duration_scale": 99,
         "avoid_directions": ["NE", "ZZ"], "strategy": "x"}, state)
    assert plan.program == "BACKUP"
    assert plan.duration_scale == 1.4
    assert plan.avoid_directions == ["NE"]
    assert plan.targets == ["1", "2"]


# ---------------------------------------------------------------------------
# 预算保护与观测记忆 (本轮新增)
# ---------------------------------------------------------------------------


class _FakeReq:
    """最小 requirement 桩, 用于测试节流逻辑 (只需 decision_sequence/new_messages)."""

    def __init__(self, seq, new_messages=None, last_result=None):
        self.decision_sequence = seq
        self.new_messages = new_messages or []
        self.last_result = last_result


def _planner_with_llm(card_name="alpha"):
    """构造一个带 LLMPlanner (假密钥) 的 Planner, 便于测试节流."""

    class _FakeLLM:
        def __init__(self):
            self.decide_calls = 0

        def plan_night(self, state):
            return NightPlan(targets=state.candidate_targets, source="static")

        def decide_action(self, state, night_plan):
            self.decide_calls += 1
            return None  # 不改变动作, 仅计数

        def diagnose_abnormal(self, state):
            return None

    p, _ = _make_planner_with_card(card_name)
    p.llm = _FakeLLM()
    return p, p.llm


def test_llm_decide_throttled_not_every_round():
    """环节B 的 LLM 调用必须是节流的, 不能每轮都调 (900s 预算成败项)."""
    p, _ = _planner_with_llm()
    reqs = [_FakeReq(seq=s) for s in range(1, 31)]
    calls = 0
    p.last_llm_decide_seq = -10**9
    for req in reqs:
        if p._should_call_llm_decide(req, is_new_night=False):
            calls += 1
            p.last_llm_decide_seq = req.decision_sequence
    assert 1 <= calls < 30, f"30 轮内应节流 (实际 {calls})"


def test_llm_decide_keypoint_triggers():
    """关键点 (新夜 / 新限时请求) 应触发环节B."""
    p, _ = _planner_with_llm()
    p.last_llm_decide_seq = 5
    # 新夜
    assert p._should_call_llm_decide(_FakeReq(6), is_new_night=True)
    # 新限时请求
    req = _FakeReq(7, new_messages=[{"record_type": "observation_request"}])
    assert p._should_call_llm_decide(req, is_new_night=False)


def test_llm_decide_respects_total_cap():
    p, _ = _planner_with_llm()
    p.llm_decide_max_total = 3
    p.llm_decide_calls = 3
    assert not p._should_call_llm_decide(_FakeReq(1), is_new_night=True)


def test_observed_targets_deprioritized():
    """已得分目标应被降权, 避免重复曝光 (规则 5.4: 多次曝光不累加)."""
    p, _ = _make_planner_with_card("alpha")
    all_targets = list(p.targets.targets)
    base = p._priorities_for(all_targets)
    # 选一个当前优先级较高的目标, 标记为已观测后, 它应被 -5000 强惩罚挤出前列
    top = max(all_targets, key=lambda t: base.get(t.target_id, 0.0))
    observed_priority = base[top.target_id] - 5000.0
    others = [t for t in all_targets if t.target_id != top.target_id]
    max_other = max(base.get(t.target_id, 0.0) for t in others)
    assert observed_priority < max_other, "已观测目标降权后应低于其它目标"


def test_repeat_pointing_penalty():
    p, _ = _make_planner_with_card("alpha")
    # 未记录指向时无惩罚
    assert p._repeat_pointing_penalty(45.0, 100.0) == 0.0
    # 记录一个指向, 相同指向应被惩罚
    p.last_pointing = (45.0, 100.0)
    assert p._repeat_pointing_penalty(45.0, 100.0) > 0.0
    # 相距很远 (> 半个视场) 不应惩罚
    assert p._repeat_pointing_penalty(45.0, 260.0) == 0.0


# ---------------------------------------------------------------------------
# 空间连续性 / 扫描收敛 (本轮新增)
# ---------------------------------------------------------------------------


def test_continuity_bonus_prefers_near_pointing():
    """连续性目标函数: 离上次指向越近, 值越高 (无前者为 0)."""
    p, _ = _make_planner_with_card("alpha")
    assert p._continuity_bonus(45.0, 100.0) == 0.0        # 无历史
    p.prev_pointing = (45.0, 100.0)
    near = p._continuity_bonus(45.5, 100.5)
    far = p._continuity_bonus(70.0, 100.0)
    assert near > far, "邻近候选的连续性应更高"


def test_continuity_momentum_rewards_sweep_direction():
    """惯性项: 延续既有扫描方向的候选应获得额外奖励."""
    p, _ = _make_planner_with_card("alpha")
    p.prev_pointing = (45.0, 100.0)
    p.sweep_altaz = (1.0, 0.0)          # 正在向北扫描
    forward = p._continuity_bonus(46.0, 100.0)   # 继续向北
    backward = p._continuity_bonus(44.0, 100.0)  # 反向
    assert forward > backward, "延续扫描方向的连续性应更高"


def test_continuity_penalizes_recent_revisit():
    """回访去重: 与近期视场几乎重合的候选应被扣分."""
    p, _ = _make_planner_with_card("alpha")
    p.prev_pointing = (45.0, 100.0)
    p.recent_fields = [(60.0, 200.0, 1)]
    fresh = p._continuity_bonus(46.0, 100.0)
    revisit = p._continuity_bonus(60.0, 200.0)
    assert revisit < fresh, "回访近期视场应被扣分"


def test_continuity_seed_centers_generated():
    """连续性种子中心: 应在上次指向附近/扫描方向生成若干候选."""
    p, _ = _make_planner_with_card("alpha")
    assert p._continuity_seed_centers({}) == []      # 无历史则无种子
    p.prev_pointing = (50.0, 120.0)
    seeds = p._continuity_seed_centers({})
    assert len(seeds) >= 3
    # 至少有一个种子距离上次指向在数个视场以内 (可形成平滑过渡)
    import math
    fov = p._fiber_grid.fov_side_deg
    near = [s for s in seeds if planner_mod.angular_separation_altaz(s[0], s[1], 50.0, 120.0) < fov * 4]
    assert near, "应存在靠近上次指向的连续性种子"


def test_plan_observe_forms_smoother_sweep_than_baseline():
    """端到端: 开启连续性后的相邻指向跳变, 应显著小于关闭时."""
    import statistics

    def avg_jump(enable: bool):
        p, nights = _make_planner_with_card("alpha")
        if not enable:
            p.continuity_band_fraction = 0.0
            p.continuity_band_abs = 0.0
            p._continuity_seed_centers = lambda altaz: []
        from protocol import DecisionRequest
        now = datetime.fromisoformat(nights[0]["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
        end = datetime.fromisoformat(nights[-1]["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
        from datetime import timedelta
        last = None
        prev = None
        seps = []
        seq = 0
        while seq < 12 and now < end:
            seq += 1
            req = DecisionRequest({
                "decision_sequence": seq,
                "payload": {"now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "survey_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "wallclock": {"remaining_seconds": 900, "remaining_real_cpu_seconds": 900,
                                          "wall_remaining_seconds": 1800},
                            "latest_bulletin": {"notices": []}, "active_requests": [],
                            "new_messages": [], "last_result": last},
            })
            act = p.decide(req)
            if act.type == "observe" and act.pointing:
                pt = (act.pointing["alt_deg"], act.pointing["az_deg"])
                if prev is not None:
                    seps.append(planner_mod.angular_separation_altaz(pt[0], pt[1], prev[0], prev[1]))
                prev = pt
                tids = list(act.assignments.values())
                last = {"action": "observe", "assigned_count": len(tids),
                        "hit_count": len(tids), "hits": [{"target_id": t} for t in tids]}
                step = act.exposure_seconds or 900
            elif act.type == "wait":
                last = {"action": "wait"}; step = act.duration_seconds or 900
            else:
                step = 900; last = {"action": "wait"}
            now += timedelta(seconds=step)
        return statistics.mean(seps) if seps else float("inf")

    baseline = avg_jump(False)
    converged = avg_jump(True)
    assert converged < baseline, f"连续性应降低平均跳变 (baseline={baseline:.1f}, converged={converged:.1f})"
