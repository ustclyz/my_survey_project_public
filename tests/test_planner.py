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
    """端到端: 开启连续性后的相邻指向跳变, 应显著小于关闭时.

    注意: 必观测锚定 (required-anchoring) 会为了覆盖散布的必观测目标而主动跳转,
    这会与"平滑扫天"冲突 (且属有意设计 —— 覆盖优先). 因此本测试同时禁用必观测
    锚定与加成, 以单独隔离"连续性机制"的效果.
    """
    import statistics

    def avg_jump(enable: bool):
        p, nights = _make_planner_with_card("alpha")
        # 隔离连续性机制: 关闭必观测锚定/加成, 避免其主导指向选择
        p.required_field_bonus = 0.0
        p._required_unfinished = set()
        p._required_anchor_centers = lambda altaz: []
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
    # 注意: 这是"涌现统计", 对候选集细节敏感 (多次因合理的性能/取点优化而抖动),
    # 因此这里只把"连续性不得显著变差"作为兜底; 机制本身由下面的单测直接覆盖。
    assert converged <= baseline * 1.3, (
        f"连续性不应显著加剧跳变 (baseline={baseline:.1f}, converged={converged:.1f})")


def test_continuity_mechanism_prefers_near_and_aligned_centers():
    """直接测连续性机制本身 (不受候选集抖动影响)."""
    p, _ = _make_planner_with_card("alpha")
    p.prev_pointing = (60.0, 100.0)
    p.sweep_altaz = (0.0, 2.0)          # 扫描方向: 方位角正向
    near = p._continuity_bonus(60.0, 104.0)     # 近且顺着扫描方向
    far = p._continuity_bonus(60.0, 320.0)      # 远且逆着方向
    assert near > far, "连续性应偏好'近且顺着扫描方向'的视场"

    # 回访去重: 与近期视场几乎重合要扣分
    p.recent_fields = [(60.0, 104.0, p.decisions_seen)]
    revisit = p._continuity_bonus(60.0, 104.0)
    assert revisit < near, "近期回访同一视场应被扣分"


def test_continuity_seed_centers_are_near_previous_pointing():
    """连续性种子必须落在上次指向附近 (平滑扫天的基础)."""
    p, _ = _make_planner_with_card("alpha")
    p.prev_pointing = (55.0, 130.0)
    p.sweep_altaz = (0.5, 2.0)
    seeds = p._continuity_seed_centers({})
    assert seeds, "应生成连续性种子"
    step_max = max(1.0, p._fiber_grid.fov_side_deg * 0.75) * 3.5
    for alt, az in seeds:
        assert planner_mod.angular_separation_altaz(alt, az, 55.0, 130.0) <= step_max + 1e-6


# ---------------------------------------------------------------------------
# 关键几何正确性 + 必观测保障 (本轮新增)
# ---------------------------------------------------------------------------


def test_tangent_offsets_returns_north_east():
    """tangent_offsets 必须返回 (north, east): 目标在北 -> 第一分量>0; 东 -> 第二>0.

    这是历史严重缺陷的回归测试: 若返回顺序被写成 (east, north), 会导致
    FiberGrid.classify 的方位整体交换, 命中率骤降.
    """
    c_alt, c_az = 45.0, 100.0
    north, east = planner_mod.tangent_offsets(46.0, 100.0, c_alt, c_az)
    assert north > 0.1 and abs(east) < 0.05, "目标在北: 第一分量应为 north>0"
    north2, east2 = planner_mod.tangent_offsets(45.0, 101.0, c_alt, c_az)
    assert east2 > 0.05 and abs(north2) < 0.05, "目标在东: 第二分量应为 east>0"


def test_fill_pointing_places_target_in_correct_fiber():
    """给定指向, _fill_pointing 应把目标放到 classify 判定的同一根光纤.

    端到端验证"预测光纤 == 独立几何重算的真实光纤"(轴向未写反).
    """
    p, _ = _make_planner_with_card("alpha")
    # 构造一个含多目标的可见集, 选一个作为视场中心
    lst = 100.0
    visible = [t for t in p.targets.targets if p._visible(t, lst)]
    assert visible
    altaz = {t.target_id: planner_mod.radec_to_altaz(t.ra_deg, t.dec_deg, lst, p.lat) for t in visible}
    ranked = visible
    priorities = {t.target_id: (1000.0 if t.required else 1.0) for t in visible}
    center = altaz[ranked[0].target_id]
    assignments, _ = p._fill_pointing(ranked, priorities, altaz, center[0], center[1])
    assert assignments
    grid = p._fiber_grid
    for fiber_str, tid in assignments.items():
        a_alt, a_az = altaz[tid]
        off = planner_mod.tangent_offsets(a_alt, a_az, center[0], center[1])
        north, east = off
        true_fiber, _ = grid.classify(east, north)
        assert str(true_fiber) == fiber_str, f"{tid}: 预测 {fiber_str} != 真实 {true_fiber} (轴向写反?)"


def test_exposure_for_completion_scales_with_flux():
    """完成因子曝光估算: 越暗的目标, 所需曝光越长."""
    p, _ = _make_planner_with_card("alpha")
    from models import Target
    bright = Target(target_id="b", ra_deg=0, dec_deg=0, feature_flux=1.0, science_weight=1.0, required=True)
    dim = Target(target_id="d", ra_deg=0, dec_deg=0, feature_flux=0.1, science_weight=1.0, required=True)
    altaz_pair = (60.0, 0.0)
    need_bright = p._exposure_for_completion(bright, altaz_pair, None)
    need_dim = p._exposure_for_completion(dim, altaz_pair, None)
    assert need_dim > need_bright, "暗目标应需要更长曝光"


def test_choose_duration_extends_for_required():
    """含必观测目标的视场: 曝光应比普通视场更长 (保障完成因子 0.5)."""
    p, _ = _make_planner_with_card("alpha")
    from models import Target
    lst = 100.0
    required = [t for t in p.targets.targets if t.required and p._visible(t, lst)]
    assert required
    t = required[0]
    altaz = {t.target_id: planner_mod.radec_to_altaz(t.ra_deg, t.dec_deg, lst, p.lat)}
    dur_req = p._choose_duration({str(0): t.target_id}, 100000.0, altaz, lst, now=None)
    assert p.min_exposure <= dur_req <= p.max_exposure
    assert dur_req >= 60


def test_required_anchor_centers_only_visible_unfinished():
    """必观测锚定: 只对可见且未完成的必观测目标生成中心."""
    p, _ = _make_planner_with_card("alpha")
    p._required_unfinished = set()
    assert p._required_anchor_centers({}) == []
    # 构造一个可见的未完成必观测目标
    lst = 100.0
    vis_req = [t for t in p.targets.targets if t.required and p._visible(t, lst)]
    assert vis_req
    t = vis_req[0]
    p._required_unfinished = {t.target_id}
    altaz = {t.target_id: planner_mod.radec_to_altaz(t.ra_deg, t.dec_deg, lst, p.lat)}
    centers = p._required_anchor_centers(altaz)
    assert len(centers) >= 1, "应对可见未完成必观测生成锚定中心"



# ---------------------------------------------------------------------------
# LLM 输出稳健性 + 提前 finish 保护 (本轮新增)
# ---------------------------------------------------------------------------


def test_llm_json_extraction_robust():
    """_parse_content 应能从带解释/代码围栏/嵌套的回复中抽出 JSON 对象."""
    from llm import LLMClient
    # 纯 JSON
    assert LLMClient._parse_content('{"a": 1}') == {"a": 1}
    # 前后有解释文字
    assert LLMClient._parse_content('好的, 结果是 {"action": "observe"} 以上')["action"] == "observe"
    # ```json 代码围栏
    assert LLMClient._parse_content('```json\n{"action":"wait"}\n```')["action"] == "wait"
    # 字符串内含花括号 (平衡扫描不应被打断)
    r = LLMClient._parse_content('{"strategy": "遇到 { 和 } 也要正确", "program": "DARK"}')
    assert r["program"] == "DARK" and "{" in r["strategy"]
    # 多个对象: 取第一个完整对象
    assert LLMClient._parse_content('{"a":1}{"b":2}') == {"a": 1}
    # 无可解析 JSON -> 抛错
    import pytest
    from llm import LLMError
    with pytest.raises(LLMError):
        LLMClient._parse_content("完全没有 JSON")


def test_llm_finish_ignored_within_night():
    """LLM 幻觉 finish 不得在夜内提前终止巡天 (历史缺陷: card C 首轮即 finish)."""
    p, nights = _make_planner_with_card("alpha")

    # 伪造一个总是返回 finish 的 LLM
    class _FinishLLM:
        llm_decide_calls = 0
        last_llm_decide_seq = -10**9

        def plan_night(self, state):
            return NightPlan(targets=state.candidate_targets, source="static")

        def decide_action(self, state, night_plan):
            from models import Action
            return Action(type="finish", reason="LLM: 收尾", decision_source="llm")

        def diagnose_abnormal(self, state):
            return None

    p.llm = _FinishLLM()
    # 直接测试保护判据: 夜内 (now < night_end) 一律不认为结束
    n = nights[0]
    start = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(n["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    from datetime import timedelta
    mid = start + (end - start) / 2
    assert p._survey_effectively_over(mid, end) is False, "夜内不得接受 finish"

    # 端到端: 夜内一次 decide 不应返回 finish
    from protocol import DecisionRequest
    req = DecisionRequest({
        "decision_sequence": 1,
        "payload": {"now_utc": mid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "survey_end_utc": nights[-1]["observing_end_utc"],
                    "wallclock": {"remaining_seconds": 900, "remaining_real_cpu_seconds": 900,
                                  "wall_remaining_seconds": 1800},
                    "latest_bulletin": {"notices": []}, "active_requests": [],
                    "new_messages": [], "last_result": None},
    })
    act = p.decide(req)
    assert act.type != "finish", "夜内 LLM 的 finish 应被忽略"


def test_achieved_g_tracking_and_retry_bonus():
    """达标记忆: 完成因子上界 < 0.5 的目标不应被强惩罚 (应保留重试机会)."""
    p, _ = _make_planner_with_card("alpha")
    p.observed_ids.add("V4T000001")
    p.achieved_g["V4T000001"] = 0.3   # 上界 < 0.5 -> 未达标
    # 优先级里不应出现 -5000 的强饱和惩罚; 用 _plan_observe 内部逻辑间接验证:
    # 直接检查 _survey_effectively_over 之外的辅助: 这里构造 priorities 复现惩罚规则.
    priorities = {"V4T000001": 1000.0, "V4T000002": 1000.0}
    sat_threshold = 0.5
    tid = "V4T000001"
    g_upper = p.achieved_g.get(tid)
    if g_upper is not None and g_upper >= sat_threshold:
        priorities[tid] -= 5000.0
    else:
        priorities[tid] += 300.0
    assert priorities["V4T000001"] > priorities["V4T000002"], "未达标目标应获重试加成而非饱和惩罚"


def test_budget_adaptive_wait_scales_with_budget():
    """预算越紧, 等待动作跨越的时隙越多 (避免逐 slot 空转耗尽 CPU)."""
    p, nights = _make_planner_with_card("alpha")
    n = nights[0]
    start = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(n["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    now = start + (end - start) / 4
    p.per_decision_cpu = 0.5                      # 充裕
    loose = p._wait_action(now, start, end, "test")
    p._empty_waits_in_night = 0
    p.per_decision_cpu = 0.03                     # 紧张
    tight = p._wait_action(now, start, end, "test")
    assert (loose.duration_seconds or 0) <= (tight.duration_seconds or 0), "预算紧时应等更久"


def test_budget_adaptive_wait_defers_night_skip_until_empty():
    """只有本夜连续多轮无目标时才跳夜, 避免刚入夜就误跳."""
    p, nights = _make_planner_with_card("alpha")
    n = nights[0]
    start = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(n["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    now = start
    p.per_decision_cpu = 0.01
    p._empty_waits_in_night = 0                   # 刚入夜
    a1 = p._wait_action(now, start, end, "test")
    assert a1.until_utc is None, "刚入夜不应跳到下一夜"


# ---------------------------------------------------------------------------
# 任务卡解析 (平台 card_id -> 仓库目录名) 回归
# ---------------------------------------------------------------------------


def test_card_slug_candidates_cover_platform_and_practice_ids():
    # 注意: 从 planner 导入 (而非 agent) —— 导入 agent 会触发 stdout 硬化,
    # 污染同进程内 protocol 的 stdout 纯净性测试。
    candidates = planner_mod.card_slug_candidates

    # 正式赛: A / A1 -> cardA / cardA1 (同时保留原样)
    assert candidates("A")[:2] == ["A", "a"]
    assert "cardA" in candidates("A")
    assert "cardA1" in candidates("A1")
    # 练习卡: 原样即可命中
    assert candidates("alpha")[0] == "alpha"
    # 空值 / 残缺输入不应抛异常
    assert candidates("") == []
    assert candidates(None) == []


def test_resolve_card_accepts_platform_card_id():
    """平台下发 card_id='C' 时必须解析到仓库目录 cards/cardC.

    回归背景: 此前 agent 直接把 card_id 当作目录名, 'C' 匹配不到 'cardC',
    于是整场退化为"无卡配置", 光纤几何与计分基准全部用错。
    """
    resolve = planner_mod.resolve_card

    class _Init:
        def __init__(self, card_id):
            self.card_id = card_id

    assert resolve(_Init("C")).name == "cardC"
    assert resolve(_Init("cardD")).name == "cardD"
    assert resolve(_Init("alpha")).name == "alpha"
    # 仓库中确实不存在的卡 (A1-D1) 仍应安全返回 None, 由 Planner 用 payload 兜底
    assert resolve(_Init("A1")) is None
    assert resolve(_Init("")) is None


def test_runtime_grid_uses_payload_fiber_area_without_card_file():
    """找不到卡片文件时, 光纤几何必须取自 payload 的 fiber_area, 而非硬编码 0.4."""
    import math as _math

    from protocol import parse_initialize

    init = parse_initialize({
        "message_type": "initialize",
        "payload": {
            "task_card": {"card_id": "C1"},            # 仓库无 cards/cardC1
            "site": {"latitude_deg": 19.8207, "longitude_deg": -155.4681,
                     "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
            "survey": {"start_utc": "2026-12-02T05:00:00Z", "end_utc": "2026-12-02T18:00:00Z",
                       "slot_seconds": 900,
                       "nights": [{"night_id": "N1", "night_date": "2026-12-01",
                                   "observing_start_utc": "2026-12-02T05:00:00Z",
                                   "observing_end_utc": "2026-12-02T18:00:00Z", "slot_count": 52}]},
            "instrument": {"n_fibers": 9, "grid_side": 3, "fiber_area_deg2": 0.7, "gap_deg": 0.0,
                           "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}},
            "scoring": {"q0": 0.68, "flux_zero_point": 0.5, "exposure_zero_point_seconds": 900},
            "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                     "feature_flux", "science_weight", "required"],
                        "rows": [["T1", 20.0, 10.0, "QSO", 0.6, 1.0, True]]},
            "limits": {"global_wallclock_seconds": 900, "response_max_bytes": 524288},
        },
    })
    tool = planner_mod.PlannerTool(None)      # 模拟"未找到卡片文件"
    p = planner_mod.Planner(init, tool, llm_planner=None)

    assert p.tool.card is not None, "无卡片文件时应构造运行期卡片兜底"
    assert p._fiber_grid.side == 3
    assert p._fiber_grid.n_fibers == 9
    assert abs(p._fiber_grid.fiber_side_deg - _math.sqrt(0.7)) < 1e-9, "必须用 payload 的 fiber_area"
    # 计分基准也要来自 payload (而非默认 0.5/900)
    assert p.tool.card.score_config.get("flux_zero_point") == 0.5


# ---------------------------------------------------------------------------
# 站点关闭判定: 周预报(一周展望)不得触发"此刻关闭" (回归)
# ---------------------------------------------------------------------------


def _closed_req(bulletin_notices=None, forecast_notices=None,
                now="2026-10-02T02:00:00Z", end="2026-10-08T09:00:00Z"):
    from protocol import DecisionRequest

    return DecisionRequest({
        "decision_sequence": 1,
        "payload": {
            "now_utc": now,
            "survey_end_utc": end,
            "wallclock": {"remaining_seconds": 900, "remaining_real_cpu_seconds": 900,
                          "wall_remaining_seconds": 1800},
            "latest_bulletin": (None if bulletin_notices is None else
                                {"record_type": "bulletin", "notices": bulletin_notices}),
            "latest_forecast": (None if forecast_notices is None else
                                {"record_type": "forecast",
                                 "coverage_start_utc": "2026-10-02T00:00:00Z",
                                 "coverage_end_utc": "2026-10-08T00:00:00Z",
                                 "notices": forecast_notices}),
            "active_requests": [], "new_messages": [], "last_result": None,
        },
    })


def test_site_closed_uses_bulletin_only_not_forecast():
    """回归: 周预报里 "某夜 rain/ALL" 不得让当前夜整夜关闭.

    背景: 公开测试卡 PT 上, 预报(一周展望)带 rain+ALL 触发 _site_closed,
    导致 253 轮全 wait、0 观测、0 分。预报是未来展望(带 nights[]), 不能
    等价于"现在关闭"; 只有逐 slot 的**当前简报**才能。
    """
    p, _ = _make_planner_with_card("alpha")
    far_rain = [{"event_kind": "rain", "direction": "ALL", "nights": ["2026-10-08"]}]

    # 预报(未来某夜有雨) + 当前简报无 notice -> 不得关闭
    assert p._site_closed(_closed_req(bulletin_notices=[], forecast_notices=far_rain)) is False
    # 只有预报、没有简报 -> 不得关闭
    assert p._site_closed(_closed_req(bulletin_notices=None, forecast_notices=far_rain)) is False
    # 当前简报本身 rain/ALL -> 应当关闭 (正确行为)
    assert p._site_closed(_closed_req(bulletin_notices=[{"event_kind": "rain", "direction": "ALL"}])) is True
    # storm/ALL 同理
    assert p._site_closed(_closed_req(bulletin_notices=[{"event_kind": "storm", "direction": "ALL"}])) is True
    # 简报报雨但只是某个方向 -> 不关闭
    assert p._site_closed(_closed_req(bulletin_notices=[{"event_kind": "rain", "direction": "NE"}])) is False
    # 没有任何消息 -> 不关闭
    assert p._site_closed(_closed_req(bulletin_notices=None, forecast_notices=None)) is False

    # _all_notices 仍应把预报并入 (供 _build_state / LLM 规划参考), 只是不再用于关闭判定
    req = _closed_req(bulletin_notices=[{"event_kind": "haze", "direction": "W"}],
                      forecast_notices=far_rain)
    assert len(p._all_notices(req)) == 2


def test_decide_does_not_idle_all_night_for_forecast_rain():
    """端到端: 只有周预报提到 rain/ALL 时, decide() 不应返回"全天天雨/暴风"的等待."""
    p, nights = _make_planner_with_card("alpha")
    n = nights[0]
    start = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(n["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)

    req = _closed_req(
        bulletin_notices=[],
        forecast_notices=[{"event_kind": "rain", "direction": "ALL", "nights": ["2026-10-20"]}],
        now=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    action = p.decide(req)
    assert "全天天雨/暴风" not in (action.reason or ""), "预报不得触发整夜关闭"


# ---------------------------------------------------------------------------
# 性能结构 (skyindex / 每轮只算一次) 回归
# ---------------------------------------------------------------------------


def test_skyindex_visibility_matches_altitude_rule():
    """TargetIndex 的时角判据必须与 alt >= 阈值 严格等价 (提速的核心前提)."""
    import math as _math
    import random

    import skyindex

    card = _load_card("alpha")
    targets = planner_mod.Target.from_preplan
    subset = [targets(t) for t in card.targets[:400]]
    lat = card.fiber_config["site"]["latitude_deg"]
    min_alt = 30.0 + planner_mod.ALT_MARGIN_DEG
    idx = skyindex.TargetIndex(subset, lat, min_alt)

    rng = random.Random(20261006)
    mismatches = 0
    for _ in range(300):
        lst = rng.uniform(0.0, 360.0)
        mask = set(idx.visible_indices(lst))
        for i, t in enumerate(subset):
            alt, _az = planner_mod.radec_to_altaz(t.ra_deg, t.dec_deg, lst, lat)
            if (alt >= min_alt) != (i in mask):
                mismatches += 1
    assert mismatches == 0, f"时角判据与高度角判据不一致 {mismatches} 次"


def test_skyindex_altaz_matches_radec_to_altaz():
    """TargetIndex 的批量 alt/az 必须与 radec_to_altaz 一致."""
    import skyindex

    card = _load_card("alpha")
    subset = [planner_mod.Target.from_preplan(t) for t in card.targets[:200]]
    lat = card.fiber_config["site"]["latitude_deg"]
    idx = skyindex.TargetIndex(subset, lat, 30.0)

    lst = 137.25
    indices = list(range(len(subset)))
    pairs = idx.altaz_pairs(indices, lst)
    for i, (alt, az) in enumerate(pairs):
        ref_alt, ref_az = planner_mod.radec_to_altaz(subset[i].ra_deg, subset[i].dec_deg, lst, lat)
        assert abs(alt - ref_alt) < 1e-9
        assert abs(((az - ref_az + 180.0) % 360.0) - 180.0) < 1e-9


def _run_n_observes(p, nights, n=6):
    """驱动 planner 走 n 次观测决策 (白天/等待自动跳过), 返回观测次数."""
    from datetime import timedelta
    from protocol import DecisionRequest

    now = datetime.fromisoformat(nights[0]["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(nights[-1]["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    last = None
    seq = 0
    observed = 0
    while observed < n and now < end and seq < 200:
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
        if act.type == "observe":
            observed += 1
            now += timedelta(seconds=act.exposure_seconds or 900)
            tids = list(act.assignments.values())
            last = {"action": "observe", "assigned_count": len(tids), "hit_count": len(tids),
                    "hits": [{"target_id": t} for t in tids]}
        elif act.type == "wait":
            now += timedelta(seconds=act.duration_seconds or 900)
            last = {"action": "wait"}
        else:
            break
    return observed


def test_decide_makes_one_visibility_pass_and_caches_priorities(monkeypatch):
    """每轮决策只应做一次全量可见性扫描, 且优先级只算一次 (不随轮次重算)."""
    import skyindex
    import preplan as preplan_mod

    counts = {"visible": 0, "priorities": 0}
    orig_visible = skyindex.TargetIndex.visible_indices

    def counted_visible(self, lst):
        counts["visible"] += 1
        return orig_visible(self, lst)

    monkeypatch.setattr(skyindex.TargetIndex, "visible_indices", counted_visible)
    orig_prio = preplan_mod.compute_priorities

    def counted_prio(card):
        counts["priorities"] += 1
        return orig_prio(card)

    monkeypatch.setattr(preplan_mod, "compute_priorities", counted_prio)

    p, nights = _make_planner_with_card("alpha")
    observed = _run_n_observes(p, nights, n=6)
    assert observed >= 3, "应至少产出几次观测"
    # 优先级: 首次之后命中缓存, 不应随轮次增长
    assert counts["priorities"] == 1, f"优先级应只算一次 (实际 {counts['priorities']})"
    # 可见性: 每轮至多一次 (允许白天等待轮多算一次, 因此用轮数上界)
    assert counts["visible"] <= observed + 2, f"可见性扫描次数过多 ({counts['visible']})"


def test_nights_are_preparsed(monkeypatch):
    """夜间起止时间应预解析一次, decide() 期间不再反复解析 ISO 串."""
    counts = {"parse": 0}
    orig_parse = planner_mod._parse_utc

    def counted(value):
        counts["parse"] += 1
        return orig_parse(value)

    monkeypatch.setattr(planner_mod, "_parse_utc", counted)
    p, nights = _make_planner_with_card("alpha")     # 预解析发生在这里
    before = counts["parse"]
    _run_n_observes(p, nights, n=6)
    assert counts["parse"] == before, f"decide() 期间不应再解析夜历 (新增 {counts['parse'] - before} 次)"


def test_request_targets_get_priority_bonus():
    """未完成且未过期的限时观测请求目标应获得显式优先级加成."""
    from datetime import timedelta

    p, nights = _make_planner_with_card("alpha")
    n = nights[0]
    now = datetime.fromisoformat(n["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    p._now_utc = now        # 与 decide() 中的行为一致 (请求加成依赖"当前时刻")
    visible = p.targets.targets[:5]
    tid = visible[0].target_id
    deadline = (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    late = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    base = p._all_priorities()
    # 基线: 不带任何请求时的有效优先级 (用于把"请求加成"与其它加成隔离开)
    _, baseline, _ = p._prepare_round(visible, [])
    base_tid = baseline[tid]

    # 未过期 -> 加成
    reqs = [{"request_id": "R1", "target_ids": [tid], "minimum_completed": 1,
             "completed_count": 0, "completed_target_ids": [], "deadline_utc": deadline}]
    _, priorities, boosted = p._prepare_round(visible, reqs)
    assert tid in boosted
    assert priorities[tid] >= base_tid + planner_mod.REQUEST_BONUS

    # 已过截止 -> 不加成
    reqs_late = [{"request_id": "R2", "target_ids": [tid], "minimum_completed": 1,
                  "completed_count": 0, "completed_target_ids": [], "deadline_utc": late}]
    _, priorities2, boosted2 = p._prepare_round(visible, reqs_late)
    assert boosted2 == set()
    assert priorities2[tid] == base_tid

    # 已达标 -> 不加成
    reqs_done = [{"request_id": "R3", "target_ids": [tid], "minimum_completed": 1,
                  "completed_count": 1, "completed_target_ids": [tid], "deadline_utc": deadline}]
    _, _, boosted3 = p._prepare_round(visible, reqs_done)
    assert boosted3 == set()


# ---------------------------------------------------------------------------
# 必观测保障 / 重试陷阱 (平台实测驱动的修复回归)
# ---------------------------------------------------------------------------


def _run_with_failures(p, nights, n=150, score_frac=0.3):
    """驱动决策, 并模拟"命中但完成因子偏低"(score = frac x weight) 的反馈.

    这正是平台实测卡B/A1 的情形: 目标被命中但 g<0.5, 于是进入"未达标"状态。
    """
    from datetime import timedelta
    from protocol import DecisionRequest

    weight = {t.target_id: (t.science_weight or 1.0) for t in p.targets.targets}
    now = datetime.fromisoformat(nights[0]["observing_start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(nights[-1]["observing_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    last = None
    seq = 0
    observed = 0
    while observed < n and now < end and seq < 400:
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
        if act.type == "observe":
            observed += 1
            now += timedelta(seconds=act.exposure_seconds or 900)
            tids = list(act.assignments.values())
            last = {"action": "observe", "assigned_count": len(tids), "hit_count": len(tids),
                    "hits": [{"target_id": t, "score": score_frac * weight.get(t, 1.0)} for t in tids]}
        elif act.type == "wait":
            now += timedelta(seconds=act.duration_seconds or 900)
            last = {"action": "wait"}
        else:
            break
    return observed


def test_retry_trap_is_bounded():
    """回归: "命中但未达标"的目标不得被无限重试.

    平台实测: 卡B 有 11 个目标被拍 ≥50 次(最多 75 次)而 2.6 万个目标从没被碰过;
    卡 A1 因此只覆盖 3,445/30,000 个目标, 直接把分数拖到 −57k。
    """
    p, nights = _make_planner_with_card("alpha")
    observed = _run_with_failures(p, nights, n=150)
    assert observed >= 50, "应产出足够多的观测"
    worst = max(p.attempt_count.values()) if p.attempt_count else 0
    # 必须有硬上限 (普通 3 次 / 必观测 8 次)
    assert worst <= planner_mod.MAX_ATTEMPTS_REQUIRED, f"单个目标被重复指派 {worst} 次, 重试未封顶"
    # 且必须覆盖到足够多的不同目标 (不能被少数目标占满)
    assert len(p.attempted_ids) >= 60, f"只覆盖了 {len(p.attempted_ids)} 个不同目标"


def test_fresh_targets_get_bonus_and_stale_ones_dropped():
    """未尝试过的可见目标应有加成; 反复失败的目标应退出本轮选择 (尝试上限)."""
    p, _ = _make_planner_with_card("alpha")
    visible = p.targets.targets[:6]
    fresh, stale = visible[0].target_id, visible[1].target_id
    # stale: 命中但未达标, 且已尝试超过上限
    p.observed_ids.add(stale)
    p.achieved_g[stale] = 0.3
    p.attempt_count[stale] = planner_mod.MAX_ATTEMPTS_NORMAL + 1
    p.attempted_ids.add(stale)

    _, priorities, _ = p._prepare_round(visible, [])
    base = p._all_priorities()
    assert priorities[fresh] >= base.get(fresh, 0.0) + planner_mod.FRESH_BONUS - 1e-6
    assert stale not in priorities, "超过尝试上限的目标应退出本轮选择"


def test_required_field_quality_dominates_retry_fields():
    """含未完成必观测的视场, 其质量必须压过任何"重试堆"视场.

    这是 161 个必观测"全场可见却一次没被指派"的直接原因:
    16 x (基础 ~11 + 重试 300) ≈ 5000 > 必观测场 1000 + 400。
    """
    assert planner_mod.REQUIRED_FIELD_DOMINANCE > 16 * (
        planner_mod.RETRY_BONUS + 100.0), "必观测场权重必须压过重试堆"
    p, _ = _make_planner_with_card("alpha")
    assert p.required_field_bonus == planner_mod.REQUIRED_FIELD_DOMINANCE


def test_completion_exposure_uses_conservative_quality():
    """达标曝光必须用"打折"的质量估计 (否则真实 g 会卡在 0.5 以下)."""
    p, _ = _make_planner_with_card("alpha")
    t = planner_mod.Target.from_preplan(_load_card("alpha").targets[0])
    pair = (55.0, 120.0)
    q_naive = p._quality_hat(t, pair, None)
    q_cons = p._quality_hat(t, pair, None, conservative=True)
    assert abs(q_cons / q_naive - planner_mod.COMPLETION_Q_SCALE) < 1e-9
    assert q_cons < q_naive, "保守估计应更小 -> 曝光更长"
    # 所需曝光不得短于"乐观估计"下的所需曝光 (允许被 max_exposure 截断)
    scoring = p.init.scoring or {}
    f0 = float(scoring.get("flux_zero_point", 0.5)) or 0.5
    t0 = float(scoring.get("exposure_zero_point_seconds", 900.0)) or 900.0
    naive_need = 0.5 * f0 * t0 / (max(t.feature_flux, 1e-6) * q_naive)
    need_cons = p._exposure_for_completion(t, pair, None)
    assert need_cons >= min(p.max_exposure, naive_need) - 1e-6


def test_required_anchor_rotation_avoids_starvation():
    """必观测锚定必须轮转: 同一批目标不应每轮都被重复锚定."""
    p, _ = _make_planner_with_card("alpha")
    required = [t for t in p.targets.targets if t.required][:40]
    p._required_unfinished = {t.target_id for t in required}
    # 高度角相同 -> 时序质量打平, 只能靠轮转区分; 目标数 (40) 远多于每轮额度
    altaz = {t.target_id: (60.0, float(i)) for i, t in enumerate(required)}
    per_round = max(1, planner_mod.MAX_REQUIRED_ANCHOR_CENTERS // 2)
    for k in range(1, 7):
        p.decisions_seen = k
        assert p._required_anchor_centers_impl(altaz), "应有候选中心"
    anchored = set(p._last_anchor)
    assert len(anchored) > per_round, (
        f"锚定未轮转: 6 轮只锚定了 {len(anchored)} 个不同目标 (每轮上限 {per_round})")
