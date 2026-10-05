#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_demo.py - 本地闭环回放 (重点自测).

用练习卡 ``alpha`` (天气公开, 含 targets/night_calendar/配置) 模拟若干轮
``decision_request`` -> 智能体 -> ``decision_response`` 的闭环, 并对回复做
协议级校验, 确认:

    * 输出只走 stdout JSON, 日志走 stderr;
    * 程序不崩溃, 能在预算内完成;
    * 每个 observe 动作合法 (光纤不重复/曝光在范围内/目标公开);
    * 能观察到 LLM 环节被调用, 或明确标记静态回退.

用法::

    py tests/replay_demo.py --card alpha --rounds 40 [--with-llm]

不依赖模拟器; 天气/得分用一个确定性的简易模型代替 (仅用于跑通闭环).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import preplan  # noqa: E402
import protocol  # noqa: E402
from config import load_config  # noqa: E402
from llm import LLMPlanner  # noqa: E402
from models import Action  # noqa: E402
from planner import Planner, PlannerTool  # noqa: E402


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_nights(card) -> list:
    import csv

    nights = []
    with (card.root / "public" / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nights.append(
                {
                    "night_id": row["night_id"],
                    "night_date": row["night_date"],
                    "observing_start_utc": row["observing_start_utc"],
                    "observing_end_utc": row["observing_end_utc"],
                    "slot_count": int(row.get("slot_count") or 0),
                }
            )
    return nights


def _build_initialize(card, nights) -> dict:
    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class,
             t.feature_flux, t.science_weight, t.required] for t in card.targets]
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "message_type": "initialize",
        "payload": {
            "schema_version": "v4-initialize-v1",
            "task_card": {"card_id": card.name},
            "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                     "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                     "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
            "survey": {"start_utc": nights[0]["observing_start_utc"],
                       "end_utc": nights[-1]["observing_end_utc"],
                       "slot_seconds": 900, "nights": nights},
            "instrument": {"n_fibers": card.n_fibers, "grid_side": 4, "fiber_area_deg2": 0.4,
                           "gap_deg": 0.0,
                           "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}},
            "scoring": card.score_config,
            "footprint": [],
            "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                     "feature_flux", "science_weight", "required"], "rows": rows},
            "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32,
                       "response_max_bytes": 524288},
        },
    }


def _validate_response(obj: dict, seq: int, known_targets: set, n_fibers: int) -> None:
    """对回复做协议级校验; 不合法直接抛 AssertionError."""
    assert obj["protocol_version"] == protocol.PROTOCOL_VERSION
    assert obj["message_type"] == "decision_response"
    assert obj["decision_sequence"] == seq, "decision_sequence 必须回填"
    action = obj["action"]
    assert action in ("observe", "wait", "report", "finish")
    if action == "observe":
        p = obj["pointing"]
        assert 0.0 <= p["alt_deg"] <= 90.0
        assert 0.0 <= p["az_deg"] < 360.0
        assignments = obj.get("assignments") or {}
        assert assignments, "observe 必须有 assignments"
        seen_f, seen_t = set(), set()
        for fiber, tid in assignments.items():
            fi = int(fiber)
            assert 0 <= fi < n_fibers
            assert fi not in seen_f, "光纤重复"
            assert tid not in seen_t, "目标重复"
            assert tid in known_targets, f"未知目标 {tid}"
            seen_f.add(fi)
            seen_t.add(tid)
        assert isinstance(obj["duration_seconds"], int)
        assert 60 <= obj["duration_seconds"] <= 3600
        assert obj["program"] in ("DARK", "BRIGHT", "BACKUP")
    elif action == "wait":
        has_d = "duration_seconds" in obj
        has_u = "until_utc" in obj
        assert has_d ^ has_u, "wait 只能有 duration_seconds 或 until_utc 之一"
        if has_u:
            assert obj["until_utc"].endswith("Z")


def _fake_observe_result(action_obj: dict, known_alt: dict, assigned_index: int) -> dict:
    """用一个确定性的简易模型模拟一次观测结果 (仅用于跑通闭环).

    以约 70% 概率把所分配目标判为命中, 并给出一个与权重/曝光相关的得分.
    """
    assignments = action_obj.get("assignments") or {}
    hits = []
    for tid in assignments.values():
        # 确定性伪随机: 以 target_id 哈希决定是否命中
        if (abs(hash(tid)) % 10) < 7:
            score = 0.3 + (abs(hash(tid + "s")) % 100) / 100.0
            hits.append({"target_id": tid, "score": round(score, 4)})
    return {
        "action": "observe",
        "observe_index": assigned_index,
        "assigned_count": len(assignments),
        "hit_count": len(hits),
        "hits": hits,
    }


def run(card_name: str, rounds: int, with_llm: bool) -> int:
    card = preplan.CardData.from_card(card_name)
    nights = _load_nights(card)
    init_msg = _build_initialize(card, nights)
    init = protocol.parse_initialize(init_msg)

    cfg = load_config()
    if not with_llm:
        cfg.llm.enabled = False  # 强制静态, 确保可复现

    tool = PlannerTool(card)
    llm_planner = LLMPlanner(cfg.llm, planner_tool=tool, log=protocol.log) if cfg.llm_enabled else None
    planner = Planner(init, tool, llm_planner=llm_planner, log_fn=protocol.log)

    known = {t.target_id for t in card.targets}
    n_fibers = card.n_fibers

    # 从第一夜第一个 slot 开始
    now = _parse_utc(nights[0]["observing_start_utc"])
    survey_end = _parse_utc(nights[-1]["observing_end_utc"])
    remaining = 900.0
    wall_remaining = 1800.0
    last_result = None
    observe_index = 0
    llm_seen = False
    static_seen = False
    actions = {"observe": 0, "wait": 0, "report": 0, "finish": 0}

    for seq in range(1, rounds + 1):
        if now >= survey_end:
            protocol.log("replay: 巡天时间到, 结束")
            break
        req_msg = {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "message_type": "decision_request",
            "decision_sequence": seq,
            "payload": {
                "schema_version": "v4-decision-snapshot-v1",
                "now_utc": _fmt(now),
                "survey_end_utc": _fmt(survey_end),
                "observe_action_index": observe_index,
                "running_total": 0.0,
                "wallclock": {"remaining_seconds": remaining,
                              "remaining_real_cpu_seconds": remaining,
                              "wall_remaining_seconds": wall_remaining, "speed_factor": 1.0},
                "latest_bulletin": {"record_type": "bulletin", "slot_id": f"S{seq:03d}",
                                    "night_id": nights[0]["night_id"], "initial": seq == 1,
                                    "notices": []},
                "latest_forecast": None,
                "active_requests": [],
                "new_messages": [],
                "last_result": last_result,
            },
        }
        req = protocol.parse_decision_request(req_msg)
        action = planner.decide(req)

        # 记录 LLM 落点是否被触发
        if action.decision_source == "llm":
            llm_seen = True
        elif action.decision_source in ("static", "fallback"):
            static_seen = True

        obj = protocol.build_response_object(seq, action)
        _validate_response(obj, seq, known, n_fibers)
        # 真实写出 stdout 以验证"stdout 只走 JSON, 日志走 stderr"
        protocol.send_response(seq, action)

        actions[action.type] = actions.get(action.type, 0) + 1

        # 推进模拟时间 / 结果
        if action.type == "observe":
            last_result = _fake_observe_result(obj, {}, observe_index)
            observe_index += 1
            step = action.exposure_seconds or 900
        elif action.type == "wait":
            if action.until_utc:
                target = _parse_utc(action.until_utc)
                step = max(1, int((target - now).total_seconds()))
                now = target
                last_result = {"action": "wait"}
            else:
                step = action.duration_seconds or 900
                last_result = {"action": "wait"}
        elif action.type == "report":
            last_result = {"action": "report", "correct": False, "repaired": False, "score_delta": 0}
            step = 0
        else:  # finish
            break

        if action.type != "wait" or not action.until_utc:
            now = now + timedelta(seconds=step)
        remaining = max(0.0, remaining - 0.02)  # 模拟 CPU 消耗
        wall_remaining = max(0.0, wall_remaining - 0.05)

    # 摘要写 stderr, 保持 stdout 只有 decision_response JSON (协议纪律)
    protocol.log(f"replay: 完成 {seq} 轮; 动作统计={actions}; "
                 f"LLM触发={llm_seen} 静态={static_seen}")
    print(f"[replay_demo] card={card_name} rounds={seq} actions={actions} "
          f"llm_seen={llm_seen} static_seen={static_seen}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", default="alpha", help="练习卡名称 (默认 alpha)")
    parser.add_argument("--rounds", type=int, default=40, help="模拟轮数 (默认 40)")
    parser.add_argument("--with-llm", action="store_true", help="允许调用 LLM (需配置密钥)")
    args = parser.parse_args(argv)
    return run(args.card, args.rounds, args.with_llm)


if __name__ == "__main__":
    raise SystemExit(main())
