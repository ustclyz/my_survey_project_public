#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pro_score_sim.py - 用本地合成天气真值给 pro 内核打分 (仅本地, 不进提交包).

把 ``pro.agent.ObserverAgent`` (规则模式, 不联网) 接上 ``tools.local_scorer`` 的
官方公式复刻引擎, 在**合成天气真值**下闭环跑完一张卡, 直接算出本地总分。
这样可以在不消耗平台评测额度的前提下做参数扫描 (PRO_* 环境变量)。

注意: 合成天气与平台真值不同, 分数**只用于相对比较** (A/B 两套参数谁更好),
不要当成线上分数。

用法:
    py tools/pro_score_sim.py --card cardA --rounds 2000
    $env:PRO_LAMBDA_FRAC=0.5; py tools/pro_score_sim.py --card alpha
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import preplan  # noqa: E402
from models import Action  # noqa: E402
from protocol import parse_initialize  # noqa: E402
from tools.local_scorer import LocalScorer, _jain_uniformity, _parse_utc, _format_utc  # noqa: E402

DEFAULT_SPEED_FACTOR = 0.82
DEFAULT_BUDGET_SECONDS = 900.0
WALL_HARD_CAP_SECONDS = 3600.0


def _load_nights(card) -> list:
    nights = []
    with (card.root / "public" / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nights.append(row)
    return nights


def _instrument(card) -> dict:
    field = card.fiber_config["field"]
    side = int(round(float(field["n_fibers"]) ** 0.5))
    area = float(field["fiber_area_deg2"])
    gap = float(field.get("gap_deg", 0.0) or 0.0)
    glass = math.sqrt(area)
    pitch = glass + gap
    return {"n_fibers": int(field["n_fibers"]), "grid_side": side,
            "fiber_area_deg2": area, "gap_deg": gap,
            "glass_side_deg": glass, "pitch_deg": pitch, "fov_side_deg": side * pitch,
            "exposure": card.fiber_config["exposure"]}


def _payload(card, nights, instrument) -> dict:
    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class,
             t.feature_flux, t.science_weight, t.required] for t in card.targets]
    return {
        "task_card": {"card_id": card.root.name},
        "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                 "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                 "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
        "survey": {"start_utc": nights[0]["observing_start_utc"],
                   "end_utc": nights[-1]["observing_end_utc"],
                   "slot_seconds": 900, "nights": nights},
        "instrument": instrument,
        "scoring": card.score_config,
        "footprint": [],
        "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                 "feature_flux", "science_weight", "required"], "rows": rows},
        "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32,
                   "response_max_bytes": 524288},
    }


def _fast_slot_lookup(scorer) -> None:
    """把 LocalScorer 的 O(n) slot 查找换成二分 (大卡上必要)."""
    slots = sorted(scorer.slots, key=lambda s: s["start_dt"])
    starts = [s["start_dt"] for s in slots]
    import bisect

    def slot_for(moment):
        i = bisect.bisect_right(starts, moment) - 1
        if 0 <= i < len(slots) and slots[i]["start_dt"] <= moment < slots[i]["end_dt"]:
            return slots[i]
        return None

    scorer.slots = slots
    scorer._slot_for = slot_for


def run(card_name: str, seed: int = 20261005, rounds: int = 4000,
        d_alt: float = 0.0, d_az: float = 0.0, speed_factor: float = DEFAULT_SPEED_FACTOR,
        budget: float = DEFAULT_BUDGET_SECONDS) -> dict:
    from pro.agent import ObserverAgent  # 延迟导入, 以便 PRO_* 环境变量先生效

    card = preplan.CardData.from_card(card_name)
    nights = _load_nights(card)
    instrument = _instrument(card)
    payload = _payload(card, nights, instrument)

    # 合成天气真值 (缺失时生成; 与 local_scorer/weather_gen 同源)
    data_dir = PROJECT_ROOT / "sim_data" / card.root.name
    if not (data_dir / "weather_slots.csv").is_file():
        from tools.weather_gen import WeatherGenerator
        WeatherGenerator(card.root, seed=seed).write(data_dir)
    slots = []
    with (data_dir / "weather_slots.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["index"] = int(row["index"])
            for k in ("seeing", "transparency", "sky_quality", "instrument_efficiency"):
                row[k] = float(row[k])
            row["closed"] = int(row["closed"])
            row["start_dt"] = _parse_utc(row["start_utc"])
            row["end_dt"] = _parse_utc(row["end_utc"])
            slots.append(row)
    events = []
    ev = data_dir / "events.csv"
    if ev.is_file():
        with ev.open("r", encoding="utf-8", newline="") as fh:
            events = list(csv.DictReader(fh))

    init = parse_initialize({"message_type": "initialize", "payload": payload})
    scorer = LocalScorer(init, slots, events, pointing_offset=(d_alt, d_az))
    _fast_slot_lookup(scorer)
    agent = ObserverAgent(payload, rules_only=True)

    best, best_g = {}, {}
    now = _parse_utc(nights[0]["observing_start_utc"])
    survey_end = _parse_utc(nights[-1]["observing_end_utc"])
    last_result = None
    observe_index = 0
    actions = 0
    cpu0 = time.process_time()
    exhausted = False

    while now < survey_end and actions < rounds:
        used_cpu = time.process_time() - cpu0
        charged = used_cpu / max(1e-6, speed_factor)
        remaining_norm = budget - charged
        if remaining_norm <= 0.0:
            exhausted = True
            break
        remaining_real = remaining_norm * speed_factor
        actions += 1
        req = {
            "now_utc": _format_utc(now), "survey_end_utc": _format_utc(survey_end),
            "observe_action_index": observe_index, "running_total": sum(best.values()),
            "wallclock": {"remaining_seconds": max(0.0, remaining_norm),
                          "remaining_real_cpu_seconds": max(0.0, remaining_real),
                          "wall_remaining_seconds": max(0.0, WALL_HARD_CAP_SECONDS - used_cpu),
                          "speed_factor": speed_factor},
            "latest_bulletin": {"record_type": "bulletin", "slot_id": "S", "initial": actions == 1, "notices": []},
            "latest_forecast": None, "active_requests": [], "new_messages": [], "last_result": last_result,
        }
        act = agent.respond(req)
        kind = act.get("action")
        if kind == "observe":
            action = Action(type="observe", pointing=act.get("pointing"),
                            assignments=act.get("assignments") or {},
                            exposure_seconds=int(act.get("duration_seconds") or 900),
                            program=act.get("program") or "BACKUP")
            res = scorer.score_exposure(action, now)
            for tid, d in res["per_target"].items():
                if d["c"] > best.get(tid, 0.0):
                    best[tid] = d["c"]
                if d["g"] > best_g.get(tid, 0.0):
                    best_g[tid] = d["g"]
            hits = [{"target_id": tid, "score": round(d["c"], 4)}
                    for tid, d in res["per_target"].items() if d["hit"]]
            last_result = {"action": "observe", "observe_index": observe_index,
                           "assigned_count": res["assigned"], "hit_count": res["hits"], "hits": hits}
            observe_index += 1
            step = action.exposure_seconds or 900
        elif kind == "wait":
            until = act.get("until_utc")
            if until:
                target = _parse_utc(until)
                if target <= now:
                    target = now + timedelta(seconds=900)
                now = target
                last_result = {"action": "wait"}
                continue
            last_result = {"action": "wait"}
            step = int(act.get("duration_seconds") or 900)
        elif kind == "report":
            last_result = {"action": "report", "correct": False, "repaired": False, "score_delta": 0}
            step = 0
        else:
            break
        now += timedelta(seconds=step)

    target_sum = sum(best.values())
    required = [t for t in card.targets if t.required]
    required_missing = sum(1 for t in required if best_g.get(t.target_id, 0.0) < scorer.req_threshold)
    required_pen = scorer.req_penalty * required_missing
    J = _jain_uniformity(card, best_g, scorer.uni_band, scorer.uni_threshold)
    uniform_pen = scorer.uni_weight * (1.0 - J)
    cpu_seconds = time.process_time() - cpu0
    return {
        "card": card.root.name, "actions": actions, "observes": observe_index,
        "cpu_seconds": round(cpu_seconds, 1), "charged_seconds": round(cpu_seconds / max(1e-6, speed_factor), 1),
        "budget_exhausted": exhausted,
        "target_sum": round(target_sum, 2), "required_missing": required_missing,
        "required_penalty": round(required_pen, 2), "jain": round(J, 4),
        "uniform_penalty": round(uniform_pen, 2),
        "total": round(target_sum - required_pen - uniform_pen, 2), "best_count": len(best),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--d-alt", type=float, default=0.0)
    parser.add_argument("--d-az", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cards = args.card or ["alpha"]
    results = []
    for name in cards:
        try:
            results.append(run(name, seed=args.seed, rounds=args.rounds, d_alt=args.d_alt, d_az=args.d_az))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"card": name, "error": f"{type(exc).__name__}: {exc}"})
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    for r in results:
        if "error" in r:
            print(f"{r['card']}: ERROR {r['error']}")
            continue
        print(f"{r['card']}: total={r['total']}  target_sum={r['target_sum']}  "
              f"req_missing={r['required_missing']} (-{r['required_penalty']})  jain={r['jain']} "
              f"(-{r['uniform_penalty']})  observes={r['observes']}  charged={r['charged_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
