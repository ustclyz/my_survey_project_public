#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pro_sim.py - 本地驱动官方 pro 内核跑完一张卡, 测 CPU/时间覆盖 (仅本地, 不进提交包).

目的: 平台按"智能体自己的标准化 CPU 秒"计费, 上限 900s/卡。本工具把 pro 内核
(``pro.agent.ObserverAgent``, 规则模式, 不联网) 通过一次真实的 decision 循环跑满
一张卡的观测日历, 回答: **这张卡能否在 900s 预算内跑完, 决策成本多少, 覆盖多少目标**。

与 ``tools/budget_sim.py`` 的区别: 那个驱动的是我们自己的 ``planner``, 这个驱动的是
``pro`` 移植内核, 用于对比与调参。

口径假设:
* 命中模型是简化的 (所有被指派目标都命中, score = w·g·m, g 由公开公式估算), 因此
  **不能**当作线上分数, 只用于看 CPU 与覆盖率;
* 计费口径与平台一致: charged = CPU秒 / speed_factor, 达 900 即停。

用法:
    py tools/pro_sim.py --card cardA
    py tools/pro_sim.py --card cardB --max-decisions 3000 --progress 500
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import preplan  # noqa: E402
from pro.agent import ObserverAgent  # noqa: E402
from pro.skymath import format_utc, parse_utc  # noqa: E402

DEFAULT_SPEED_FACTOR = 0.82
DEFAULT_BUDGET_SECONDS = 900.0
WALL_HARD_CAP_SECONDS = 3600.0


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


def _instrument(card) -> dict:
    """重建平台下发的 instrument 组 (含 pro FiberGrid 需要的几何派生量)."""
    field = card.fiber_config["field"]
    side = int(round(float(field["n_fibers"]) ** 0.5))
    area = float(field["fiber_area_deg2"])
    gap = float(field.get("gap_deg", 0.0) or 0.0)
    glass = math.sqrt(area)
    pitch = glass + gap
    return {
        "n_fibers": int(field["n_fibers"]), "grid_side": side,
        "fiber_area_deg2": area, "gap_deg": gap,
        "glass_side_deg": glass, "pitch_deg": pitch, "fov_side_deg": side * pitch,
        "exposure": card.fiber_config["exposure"],
    }


def _build_init(card, nights) -> dict:
    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class,
             t.feature_flux, t.science_weight, t.required] for t in card.targets]
    return {
        "task_card": {"card_id": card.name},
        "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                 "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                 "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
        "survey": {"start_utc": nights[0]["observing_start_utc"],
                   "end_utc": nights[-1]["observing_end_utc"],
                   "slot_seconds": 900, "nights": nights},
        "instrument": _instrument(card),
        "scoring": card.score_config,
        "footprint": [],
        "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                 "feature_flux", "science_weight", "required"], "rows": rows},
        "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32,
                   "response_max_bytes": 524288},
    }


def run_card(card_name: str, budget: float = DEFAULT_BUDGET_SECONDS,
             speed_factor: float = DEFAULT_SPEED_FACTOR,
             max_decisions: int = 200000, nights_limit: int = 0,
             progress: int = 0) -> dict:
    card = preplan.CardData.from_card(card_name)
    nights = _load_nights(card)
    if nights_limit and nights_limit > 0:
        nights = nights[:nights_limit]
    init = _build_init(card, nights)
    agent = ObserverAgent(init, rules_only=True)

    f0t0 = float(card.score_config["flux_zero_point"]) * float(card.score_config["exposure_zero_point_seconds"])
    weight = {t.target_id: t.science_weight for t in card.targets}
    flux = {t.target_id: t.feature_flux for t in card.targets}
    required_ids = {t.target_id for t in card.targets if t.required}
    survey_start = parse_utc(nights[0]["observing_start_utc"])
    survey_end = parse_utc(nights[-1]["observing_end_utc"])
    now = survey_start

    cpu0 = time.process_time()
    mark = cpu0
    seq = 0
    observes = 0
    waits = 0
    assigned: dict = {}
    last_result = None
    exhausted = False

    while now < survey_end and seq < max_decisions:
        used_cpu = time.process_time() - cpu0
        charged = used_cpu / max(1e-6, speed_factor)
        remaining_norm = budget - charged
        if remaining_norm <= 0.0:
            exhausted = True
            break
        remaining_real = remaining_norm * speed_factor
        seq += 1
        payload = {
            "now_utc": format_utc(now), "survey_end_utc": format_utc(survey_end),
            "observe_action_index": observes, "running_total": 0.0,
            "wallclock": {"remaining_seconds": max(0.0, remaining_norm),
                          "remaining_real_cpu_seconds": max(0.0, remaining_real),
                          "wall_remaining_seconds": max(0.0, WALL_HARD_CAP_SECONDS - used_cpu),
                          "speed_factor": speed_factor},
            "latest_bulletin": {"record_type": "bulletin", "initial": seq == 1, "notices": []},
            "latest_forecast": None, "active_requests": [], "new_messages": [],
            "last_result": last_result,
        }
        action = agent.respond(payload)
        kind = action.get("action")
        if progress and seq % progress == 0:
            now_cpu = time.process_time()
            print(f"    [{card.name}] seq={seq} {(now_cpu - mark) / progress * 1000:.1f} ms/决策 "
                  f"charged={charged:.0f}s level={agent.planner.fast_level} "
                  f"observes={observes} distinct={len(assigned)}",
                  file=sys.stderr)
            mark = now_cpu

        if kind == "observe":
            observes += 1
            step = int(action.get("duration_seconds") or 900)
            tids = [str(v) for v in (action.get("assignments") or {}).values()]
            mult = float(card.score_config["program"]["multipliers"].get(action.get("program"), 1.0))
            hits = []
            for t in tids:
                assigned[t] = assigned.get(t, 0) + 1
                w = weight.get(t, 1.0) or 1.0
                g = min(1.0, max(0.05, flux.get(t, 0.0) * step / max(1e-9, f0t0)))
                hits.append({"target_id": t, "score": w * g * mult})
            last_result = {"action": "observe", "observe_index": observes,
                           "assigned_count": len(tids), "hit_count": len(tids), "hits": hits}
        elif kind == "wait":
            waits += 1
            until = action.get("until_utc")
            if until:
                target = parse_utc(until)
                if target <= now:
                    target = now + timedelta(seconds=900)
                now = target
                last_result = {"action": "wait"}
                continue
            step = int(action.get("duration_seconds") or 900)
            last_result = {"action": "wait"}
        elif kind == "report":
            step = 0
            last_result = {"action": "report", "correct": False, "repaired": False, "score_delta": 0}
        elif kind == "finish":
            break
        else:
            step = 0
            last_result = {"action": kind}
        now = now + timedelta(seconds=step)

    cpu_seconds = time.process_time() - cpu0
    charged_seconds = cpu_seconds / max(1e-6, speed_factor)
    req_assigned = sum(1 for t in required_ids if assigned.get(t, 0) > 0)
    repeats = sorted(assigned.values(), reverse=True)
    total_span = max(1.0, (survey_end - survey_start).total_seconds())
    covered = (now - survey_start).total_seconds()
    return {
        "card": card.name, "targets": len(card.targets), "fibers": card.n_fibers,
        "nights": len(nights), "decisions": seq, "observes": observes, "waits": waits,
        "cpu_seconds": round(cpu_seconds, 1),
        "cpu_ms_per_decision": round(cpu_seconds / max(1, seq) * 1000.0, 1),
        "charged_seconds": round(charged_seconds, 1), "budget_seconds": budget,
        "finished": bool(now >= survey_end), "budget_exhausted": exhausted,
        "coverage_pct": round(min(100.0, covered / total_span * 100.0), 1),
        "required_total": len(required_ids), "required_assigned": req_assigned,
        "required_never_assigned": len(required_ids) - req_assigned,
        "targets_assigned_distinct": len(assigned),
        "max_repeat": repeats[0] if repeats else 0,
        "final_level": agent.planner.fast_level,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", action="append", default=None, help="卡片目录名 (可重复; 默认 cardA)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--speed-factor", type=float, default=DEFAULT_SPEED_FACTOR)
    parser.add_argument("--nights", type=int, default=0, help="探针模式: 只跑前 N 夜")
    parser.add_argument("--max-decisions", type=int, default=200000)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cards = args.card or ["cardA"]

    results = []
    for name in cards:
        try:
            results.append(run_card(name, budget=args.budget, speed_factor=args.speed_factor,
                                    max_decisions=args.max_decisions, nights_limit=args.nights,
                                    progress=args.progress))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"card": name, "error": f"{type(exc).__name__}: {exc}"})

    if args.json:
        import json
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("card     targets fibers nights decisions observes  CPU(s) ms/dec charged done  cov%  req_never")
    for r in results:
        if "error" in r:
            print(f"{r['card']:<8} ERROR {r['error']}")
            continue
        print(f"{r['card']:<8} {r['targets']:>7} {r['fibers']:>5} {r['nights']:>6} {r['decisions']:>9} "
              f"{r['observes']:>8} {r['cpu_seconds']:>7.1f} {r['cpu_ms_per_decision']:>6.1f} "
              f"{r['charged_seconds']:>7.1f} {('Y' if r['finished'] else 'N'):>4} {r['coverage_pct']:>5.1f} "
              f"{r['required_never_assigned']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
