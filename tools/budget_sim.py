#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""budget_sim.py - 本地 "CPU 预算" 全程仿真 (仅本地, 不进提交包).

目的
----
平台按"智能体自己的 CPU 时间(按机器速度归一化)"计费, 上限 900s/卡。本工具用
真实的 planner.Planner 闭环跑完一张卡的全部观测窗口, 用 time.process_time()
统计 CPU, 回答一个具体问题: 这张卡能不能在 900s 内跑完整场巡天。

与 tools/local_scorer.py 的区别: 这里不评分、不模拟天气, 只推进时间, 目的是压测
CPU (决策数 x 单决策耗时)。LLM 默认关闭(静态路径), 结果确定可复现, 也符合
"等模型不计 CPU"的口径。

口径与假设 (影响解读)
--------------------
1) 每次观测都假定"命中且达标": hits 的 score = 0.75 x science_weight, 故完成
   因子上界 0.75 >= 0.5, 已达标目标不会被重复观测 —— 与真实运行 ~98% 命中率接近。
2) remaining_real_cpu_seconds 随已消耗 CPU 递减, 因此 _update_pace 的档位会像
   线上一样逐步收紧。
3) 平台机器速度系数实测约 0.82 (慢于基准机 -> 计费更贵), 故同时输出"按 0.82
   折算"的等效耗时, 并以折算值判定是否达标。

用法:
    py tools/budget_sim.py --card cardA
    py tools/budget_sim.py --card cardA --card cardB --card cardC --card cardD
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import preplan  # noqa: E402
from planner import Planner, PlannerTool, _format_utc  # noqa: E402
from protocol import DecisionRequest, parse_initialize  # noqa: E402

DEFAULT_SPEED_FACTOR = 0.82
DEFAULT_BUDGET_SECONDS = 900.0
WALL_HARD_CAP_SECONDS = 3600.0


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _load_nights(card) -> list:
    nights = []
    with (card.root / "public" / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nights.append({
                "night_id": row["night_id"], "night_date": row["night_date"],
                "observing_start_utc": row["observing_start_utc"],
                "observing_end_utc": row["observing_end_utc"],
                "slot_count": int(row.get("slot_count") or 0),
            })
    return nights


def _build_initialize(card, nights) -> dict:
    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class,
             t.feature_flux, t.science_weight, t.required] for t in card.targets]
    return {
        "message_type": "initialize",
        "payload": {
            "task_card": {"card_id": card.name},
            "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                     "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                     "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
            "survey": {"start_utc": nights[0]["observing_start_utc"],
                       "end_utc": nights[-1]["observing_end_utc"],
                       "slot_seconds": 900, "nights": nights},
            "instrument": {"n_fibers": card.n_fibers,
                           "grid_side": int(round(card.n_fibers ** 0.5)),
                           "fiber_area_deg2": card.fiber_config["field"]["fiber_area_deg2"],
                           "gap_deg": card.fiber_config["field"]["gap_deg"],
                           "exposure": card.fiber_config["exposure"]},
            "scoring": card.score_config,
            "footprint": [],
            "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                     "feature_flux", "science_weight", "required"], "rows": rows},
            "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32,
                       "response_max_bytes": 524288},
        },
    }


def run_card(card_name: str, budget: float = DEFAULT_BUDGET_SECONDS,
             speed_factor: float = DEFAULT_SPEED_FACTOR,
             max_decisions: int = 200000, nights_limit: int = 0,
             progress: int = 0) -> dict:
    """跑完一张卡的全部观测窗口, 返回 CPU / 决策数 / 覆盖率等统计.

    计费口径与平台一致: ``charged = CPU 秒 / speed_factor``, 一旦 ``charged``
    达到 ``budget`` 就停 (慢机器上允许的原始 CPU 更少)。

    Args:
        nights_limit: ``>0`` 时只跑前 N 夜 (探针模式, 用于快速看 ms/决策)。
        max_decisions: 只跑 N 轮决策就停 (探针模式)。
    """
    card = preplan.CardData.from_card(card_name)
    nights = _load_nights(card)
    if nights_limit and nights_limit > 0:
        nights = nights[:nights_limit]
    init = parse_initialize(_build_initialize(card, nights))
    planner = Planner(init, PlannerTool(card), llm_planner=None, log_fn=(lambda _s: None))

    weight = {t.target_id: t.science_weight for t in card.targets}
    required_ids = {t.target_id for t in card.targets if t.required}
    assigned_count = {}          # target_id -> 被指派次数 (用于验证"重试陷阱")
    survey_start = _parse_utc(nights[0]["observing_start_utc"])
    survey_end = _parse_utc(nights[-1]["observing_end_utc"])
    now = survey_start

    cpu0 = time.process_time()
    mark = cpu0
    decisions = 0
    observes = 0
    waits = 0
    last_result = None
    exhausted = False
    seq = 0

    while now < survey_end and seq < max_decisions:
        used_cpu = time.process_time() - cpu0
        charged = used_cpu / max(1e-6, speed_factor)
        remaining_norm = budget - charged          # 平台归一化预算
        if remaining_norm <= 0.0:
            exhausted = True
            break
        remaining_real = remaining_norm * speed_factor   # 本机真实 CPU 秒
        seq += 1
        req = DecisionRequest({"decision_sequence": seq, "payload": {
            "now_utc": _format_utc(now), "survey_end_utc": _format_utc(survey_end),
            "observe_action_index": observes, "running_total": 0.0,
            "wallclock": {"remaining_seconds": max(0.0, remaining_norm),
                          "remaining_real_cpu_seconds": max(0.0, remaining_real),
                          "wall_remaining_seconds": max(0.0, WALL_HARD_CAP_SECONDS - used_cpu),
                          "speed_factor": speed_factor},
            "latest_bulletin": {"record_type": "bulletin", "initial": seq == 1, "notices": []},
            "latest_forecast": None, "active_requests": [], "new_messages": [],
            "last_result": last_result}})
        action = planner.decide(req)
        decisions += 1
        if progress and decisions % progress == 0:
            now_cpu = time.process_time()
            print(f"    [{card.name}] seq={decisions} 最近 {progress} 轮 "
                  f"{(now_cpu - mark) / progress * 1000:.1f} ms/决策, 可见={planner.perf_visible_n}, "
                  f"fill={planner.perf_fill_calls}, pace={planner.pace_level}, "
                  f"observed={len(planner.observed_ids)}, attempted={len(planner.attempted_ids)}",
                  file=sys.stderr)
            mark = now_cpu

        if action.type == "observe":
            observes += 1
            step = action.exposure_seconds or 900
            tids = list(action.assignments.values())
            for t in tids:
                assigned_count[t] = assigned_count.get(t, 0) + 1
            last_result = {"action": "observe", "observe_index": observes,
                           "assigned_count": len(tids), "hit_count": len(tids),
                           "hits": [{"target_id": t, "score": 0.75 * (weight.get(t, 1.0) or 1.0)}
                                    for t in tids]}
        elif action.type == "wait":
            waits += 1
            if action.until_utc:
                target = _parse_utc(action.until_utc)
                now = target
                last_result = {"action": "wait"}
                continue
            step = action.duration_seconds or 900
            last_result = {"action": "wait"}
        else:
            step = 0
            last_result = {"action": action.type, "correct": False, "repaired": False,
                           "score_delta": 0}
        now = now + timedelta(seconds=step)

    cpu_seconds = time.process_time() - cpu0
    charged_seconds = cpu_seconds / max(1e-6, speed_factor)
    req_assigned = sum(1 for t in required_ids if assigned_count.get(t, 0) > 0)
    repeats = sorted(assigned_count.values(), reverse=True)
    total_span = max(1.0, (survey_end - survey_start).total_seconds())
    covered = (now - survey_start).total_seconds()
    return {
        "card": card.name,
        "targets": len(card.targets),
        "fibers": card.n_fibers,
        "nights": len(nights),
        "decisions": decisions,
        "observes": observes,
        "waits": waits,
        "cpu_seconds": round(cpu_seconds, 1),
        "cpu_ms_per_decision": round(cpu_seconds / max(1, decisions) * 1000.0, 1),
        "charged_seconds": round(charged_seconds, 1),
        "budget_seconds": budget,
        "finished": bool(now >= survey_end),
        "budget_exhausted": exhausted,
        "probe": bool(seq >= max_decisions and now < survey_end),
        "coverage_pct": round(min(100.0, covered / total_span * 100.0), 1),
        "required_total": len(required_ids),
        "required_assigned": req_assigned,
        "required_never_assigned": len(required_ids) - req_assigned,
        "targets_assigned_distinct": len(assigned_count),
        "max_repeat": repeats[0] if repeats else 0,
        "repeat_ge5": sum(1 for c in repeats if c >= 5),
        "perf": planner.cpu_summary(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", action="append", default=None,
                        help="卡片目录名 (可重复; 默认 alpha)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="每卡 CPU 预算 (秒)")
    parser.add_argument("--speed-factor", type=float, default=DEFAULT_SPEED_FACTOR,
                        help="平台机器速度系数 (计费 = CPU / 该系数)")
    parser.add_argument("--nights", type=int, default=0,
                        help="探针模式: 只跑前 N 夜 (>0 时生效)")
    parser.add_argument("--max-decisions", type=int, default=200000,
                        help="探针模式: 最多跑 N 轮决策")
    parser.add_argument("--progress", type=int, default=0,
                        help="每 N 轮打印一次进度 (0 = 关闭)")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args(argv)
    cards = args.card or ["alpha"]

    results = []
    for name in cards:
        try:
            results.append(run_card(name, budget=args.budget, speed_factor=args.speed_factor,
                                    max_decisions=args.max_decisions, nights_limit=args.nights,
                                    progress=args.progress))
        except Exception as exc:  # noqa: BLE001 - 单卡失败不影响其它卡
            results.append({"card": name, "error": f"{type(exc).__name__}: {exc}"})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    header = ("card     targets fibers nights decisions observes    CPU(s)  ms/dec  charged  done  cov%")
    print(header)
    for r in results:
        if "error" in r:
            print(f"{r['card']:<8} ERROR {r['error']}")
            continue
        print(f"{r['card']:<8} {r['targets']:>7} {r['fibers']:>5} {r['nights']:>6} {r['decisions']:>9} "
              f"{r['observes']:>8} {r['cpu_seconds']:>9.1f} {r['cpu_ms_per_decision']:>7.1f} "
              f"{r['charged_seconds']:>8.1f} {('Y' if r['finished'] else 'N'):>5} "
              f"{r['coverage_pct']:>6.1f}")
        print(f"         必观测: 共 {r['required_total']}, 已指派 {r['required_assigned']}, "
              f"从未指派 {r['required_never_assigned']} | 覆盖目标数 {r['targets_assigned_distinct']} | "
              f"单目标最多被拍 {r['max_repeat']} 次, >=5 次的有 {r['repeat_ge5']} 个")
    print(f"\nacceptance: charged_seconds < budget ({args.budget:.0f}s); speed_factor={args.speed_factor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
