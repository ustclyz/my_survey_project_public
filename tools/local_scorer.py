#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""local_scorer.py - 本地评分器 + 闭环模拟器 (仅本地调试用).

用 ``tools/weather_gen.py`` 生成的**合成天气真值**, 按官方计分公式复刻评分,
从而在本地就测出智能体"在真实天气下会得多少分".

官方公式 (复刻自 docs/participant-guide 与官方示例 agent_core/scoring.py):
    单次曝光 e, 目标 i, 实际时长 T_e:
        quality(i,p) = (eta*transp*K*L_i(t)) / (seeing * airmass(alt_i)^beta) / q0
        Q(i,e)       = sum_p dt_p * quality(i,p) / T_e            # 完成因子(含所有项)
        g(i,e)       = min(1, f_i * T_e * Q(i,e) / (f0 * T0))      # 基础贡献
        s(i,e)       = w_i * g(i,e)
        band(i,e)    = (transp*K*L_i) / (seeing*airmass^beta) / q0  # 程序档位(不含 eta)
        actual       = DARK if band>=0.65 else BRIGHT if band>=0.40 else BACKUP
        m(i,e)       = multipliers[declared] if declared==actual else 1.0
        c(i,e)       = s(i,e) * m(i,e)
    总分:
        S = sum_i max_e c(i,e)                                     # 每目标最好一次
            - 50 * (# 必观测目标 max_e g < 0.5 的个数)
            - 200 * (1 - Jain(uniformity))                          # 均匀度
            + sum_request_rewards
            + report_score
    命中条件 (决定 c 是否为 0):
        * 目标落在其**被分配光纤**的可指派方格内 (曝光开始时判定, 望远镜跟踪);
        * 全程高度角 >= 30°;
        * 该 slot 未因事件/背景关闭 (closed → C=0).

命中判定用与官方一致的球面几何 (gnomonic 投影 + 方格分类), 并可选加入
"指向偏差 (pointing_offset)" 以模拟 Hard 模式. 指向偏差逻辑独立于 agent,
以检验 agent 的鲁棒性.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import preplan  # noqa: E402
from protocol import parse_initialize, DecisionRequest  # noqa: E402
from planner import (Planner, PlannerTool, _parse_utc, _format_utc,  # noqa: E402
                     local_sidereal_deg, radec_to_altaz, tangent_offsets)

SLOT_SECONDS = 900
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ---------------------------------------------------------------------------
# 几何: 与官方 skymath 一致
# ---------------------------------------------------------------------------


def normalized_airmass(alt_deg: float) -> float:
    if alt_deg <= 0.0:
        return float("inf")
    zenith = 90.0 - alt_deg
    raw = 1.0 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    return raw / (1.0 / (1.0 + 0.50572 * 96.07995 ** -1.6364))


def _julian_date(moment: datetime) -> float:
    return moment.timestamp() / 86400.0 + 2440587.5


def _sun_radec(moment: datetime):
    days = _julian_date(moment) - 2451545.0
    mean_longitude = (280.460 + 0.9856474 * days) % 360.0
    anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    longitude = math.radians((mean_longitude + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly)) % 360.0)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return (math.degrees(math.atan2(math.cos(obliquity) * math.sin(longitude), math.cos(longitude))) % 360.0,
            math.degrees(math.asin(math.sin(obliquity) * math.sin(longitude))))


def _moon_radec(moment: datetime):
    days = _julian_date(moment) - 2451545.0
    ml = math.radians((218.316 + 13.176396 * days) % 360.0)
    an = math.radians((134.963 + 13.064993 * days) % 360.0)
    al = math.radians((93.272 + 13.229350 * days) % 360.0)
    longitude = ml + math.radians(6.289) * math.sin(an)
    latitude = math.radians(5.128) * math.sin(al)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    x = math.cos(longitude) * math.cos(latitude)
    y = math.sin(longitude) * math.cos(latitude) * math.cos(obliquity) - math.sin(latitude) * math.sin(obliquity)
    z = math.sin(longitude) * math.cos(latitude) * math.sin(obliquity) + math.sin(latitude) * math.cos(obliquity)
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))


def _separation(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def lunar_factor(moment, lst, lat, target_ra, target_dec, model):
    m_ra, m_dec = _moon_radec(moment)
    m_alt, _ = radec_to_altaz(m_ra, m_dec, lst, lat)
    if m_alt <= 0.0:
        return 1.0
    s_ra, s_dec = _sun_radec(moment)
    illum = (1.0 - math.cos(math.radians(_separation(s_ra, s_dec, m_ra, m_dec)))) / 2.0
    sep = _separation(target_ra, target_dec, m_ra, m_dec)
    penalty = (float(model.get("maximum_penalty", 0.5)) * illum
               * math.sin(math.radians(max(0.0, m_alt))) ** float(model.get("altitude_exponent", 1.0))
               * math.exp(-sep / float(model.get("angular_decay_scale_deg", 40.0))))
    return max(0.0, min(1.0, 1.0 - penalty))


# ---------------------------------------------------------------------------
# 方格
# ---------------------------------------------------------------------------


class Grid:
    def __init__(self, side, glass, gap):
        self.side = side
        self.glass = glass
        self.gap = gap
        self.pitch = glass + gap
        self.fov = side * glass + (side - 1) * gap

    def classify(self, d_north, d_east):
        half = self.fov / 2.0
        if abs(d_north) > half or abs(d_east) > half:
            return None, -1.0
        middle = self.side / 2.0
        row = min(max(int(math.floor(d_north / self.pitch + middle)), 0), self.side - 1)
        col = min(max(int(math.floor(d_east / self.pitch + middle)), 0), self.side - 1)
        fiber = row * self.side + col
        c_north = (row - (self.side - 1) / 2.0) * self.pitch
        c_east = (col - (self.side - 1) / 2.0) * self.pitch
        margin = self.glass / 2.0 - max(abs(d_north - c_north), abs(d_east - c_east))
        return (fiber, margin) if margin >= 0.0 else (None, margin)


# ---------------------------------------------------------------------------
# 评分器
# ---------------------------------------------------------------------------


class LocalScorer:
    """按官方公式评分; 输入为一次 observe 的几何 + 天气真值."""

    def __init__(self, init, slots, events, pointing_offset=(0.0, 0.0)):
        self.init = init
        self.slots = slots                    # list[dict], 逐 slot
        self.slot_by_id = {s["slot_id"]: s for s in slots}
        self.events = events
        sc = init.scoring or {}
        self.q0 = float(sc.get("q0", 1.0))
        self.f0 = float(sc.get("flux_zero_point", 0.5))
        self.t0 = float(sc.get("exposure_zero_point_seconds", 900.0))
        self.f0t0 = max(1e-9, self.f0 * self.t0)
        self.beta = float(sc.get("airmass_exponent", 0.6))
        self.lunar_model = sc.get("lunar_model") or {}
        prog = sc.get("program") or {}
        self.bands = {**{"DARK": 0.65, "BRIGHT": 0.40}, **(prog.get("bands") or {})}
        self.mults = {**{"DARK": 1.2, "BRIGHT": 1.12, "BACKUP": 1.06}, **(prog.get("multipliers") or {})}
        self.mismatch = float(prog.get("mismatch_multiplier", 1.0))
        req = sc.get("required") or {}
        self.req_penalty = float(req.get("penalty_per_missing", 50.0))
        self.req_threshold = float(req.get("observed_factor_threshold", 0.5))
        uni = sc.get("uniformity") or {}
        self.uni_weight = float(uni.get("weight", 200.0))
        self.uni_band = float(uni.get("ra_band_width_deg", 10.0))
        self.uni_threshold = float(uni.get("observed_factor_threshold", 0.5))
        self.lat = init.lat_deg
        self.lon = init.lon_deg
        self.min_alt = init.min_altitude_deg
        self.pointing_offset = pointing_offset
        self.target_by_id = {t.target_id: t for t in self._all_targets()}

    def _all_targets(self):
        from models import Target
        out = []
        for row in self.init.target_rows:
            t = Target.from_protocol_row(self.init.target_columns, row)
            if t.target_id:
                out.append(t)
        return out

    def _slot_for(self, moment):
        """找 moment 所在的 slot (按全局 UTC 对齐 900s)."""
        for s in self.slots:
            if s["start_dt"] <= moment < s["end_dt"]:
                return s
        return None

    def _weather_at(self, moment):
        s = self._slot_for(moment)
        if s is None:
            return 1.0, 1.0, 1.0, 1.0, True
        return (s["seeing"], s["transparency"], s["sky_quality"], s["instrument_efficiency"], bool(s["closed"]))

    def score_exposure(self, action, start_utc):
        """对一次 observe 评分. 返回 dict:
        {
          'per_target': {tid: {'c':float,'g':float,'hit':bool,'band':str,'declared':str}},
          'assigned': int, 'hits': int, 'cl': int,
        }
        """
        duration = int(action.exposure_seconds or 0)
        if duration <= 0 or not action.assignments or not action.pointing:
            return {"per_target": {}, "assigned": 0, "hits": 0}
        cmd_alt = action.pointing["alt_deg"]
        cmd_az = action.pointing["az_deg"]
        d_alt, d_az = self.pointing_offset
        c_alt = cmd_alt + d_alt
        c_az = (cmd_az + d_az) % 360.0
        declared = action.program or "BACKUP"

        # 曝光按 slot 边界切分, 每段再细分 <=120s (与官方一致)
        segments = self._split_segments(start_utc, duration)
        grid = self._grid()

        per_target = {}
        n_hit = 0
        for fiber_str, tid in action.assignments.items():
            t = self.target_by_id.get(tid)
            if t is None:
                continue
            pred_fiber = int(fiber_str)
            # 命中判定: 曝光开始时刻, 目标落在被分配光纤方格内
            lst0 = local_sidereal_deg(start_utc, self.lon)
            alt0, az0 = radec_to_altaz(t.ra_deg, t.dec_deg, lst0, self.lat)
            off = tangent_offsets(alt0, az0, c_alt, c_az)
            hit = False
            margin = -1.0
            if off is not None:
                true_fiber, margin = grid.classify(off[0], off[1])
                hit = (true_fiber == pred_fiber)
            # 全程高度角 >= min_alt
            alt_ok = True
            for (t0, dt) in segments:
                lst = local_sidereal_deg(t0, self.lon)
                a, _ = radec_to_altaz(t.ra_deg, t.dec_deg, lst, self.lat)
                if a < self.min_alt:
                    alt_ok = False
                    break
            if not (hit and alt_ok):
                per_target[tid] = {"c": 0.0, "g": 0.0, "hit": False, "band": "-", "declared": declared}
                continue
            n_hit += 1

            # 完成因子 Q 与程序档位 B: 逐段累加
            q_num = 0.0     # sum dt * (eta*transp*K*L)/(seeing*airmass^beta)/q0
            b_num = 0.0     # sum dt * (transp*K*L)/(seeing*airmass^beta)/q0  (不含 eta)
            total_dt = 0.0
            for (t0, dt) in segments:
                total_dt += dt
                tmid = t0 + timedelta(seconds=dt / 2.0)
                lst = local_sidereal_deg(tmid, self.lon)
                alt, _ = radec_to_altaz(t.ra_deg, t.dec_deg, lst, self.lat)
                seeing, transp, K, eta, closed = self._weather_at(tmid)
                if closed:
                    continue  # C=0: 该段不计 (但保留在分母里)
                L = lunar_factor(tmid, lst, self.lat, t.ra_deg, t.dec_deg, self.lunar_model)
                X = normalized_airmass(max(alt, 1.0))
                base = L / (self.q0 * (X ** self.beta) * max(1e-6, seeing))
                q_num += dt * (eta * transp * K) * base
                b_num += dt * transp * K * base
            T = max(1e-6, total_dt)
            Q = q_num / T
            B = b_num / T
            g = max(0.0, min(1.0, t.feature_flux * duration * Q / self.f0t0))
            s = t.science_weight * g
            actual = "DARK" if B >= self.bands["DARK"] else ("BRIGHT" if B >= self.bands["BRIGHT"] else "BACKUP")
            m = self.mults.get(declared, 1.0) if declared == actual else self.mismatch
            c = s * m
            per_target[tid] = {"c": c, "g": g, "hit": True, "band": actual,
                               "declared": declared, "margin": margin}
        return {"per_target": per_target, "assigned": len(action.assignments), "hits": n_hit}

    def _grid(self):
        side = self.init.grid_side
        # glass: 由 fiber_area 反推 (字段可能缺失)
        area = 0.4
        for key in ("fiber_area_deg2",):
            if key in (self.init.instrument or {}):
                area = float(self.init.instrument[key])
        gap = float((self.init.instrument or {}).get("gap_deg", 0.0))
        return Grid(side, math.sqrt(area), gap)

    def _split_segments(self, start, duration):
        segs = []
        cur = start
        end = start + timedelta(seconds=duration)
        while cur < end:
            # 下一个 slot 边界
            slot_end = None
            s = self._slot_for(cur)
            if s is not None:
                slot_end = s["end_dt"]
            nxt = min(end, slot_end or end, cur + timedelta(seconds=120))
            dt = (nxt - cur).total_seconds()
            if dt <= 0:
                break
            segs.append((cur, dt))
            cur = nxt
        return segs


# ---------------------------------------------------------------------------
# 闭环运行 + 汇总
# ---------------------------------------------------------------------------


def run_simulation(card_name, seed=20261005, rounds=400, pointing_offset=(0.0, 0.0),
                   with_llm=False, verbose=False):
    card = preplan.CardData.from_card(card_name)
    nights = []
    with (card.root / "public" / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nights.append(row)

    # 加载/生成天气真值
    data_dir = PROJECT_ROOT / "sim_data" / card.root.name
    slots_path = data_dir / "weather_slots.csv"
    if not slots_path.is_file():
        from tools.weather_gen import WeatherGenerator
        WeatherGenerator(card.root, seed=seed).write(data_dir)
    slots = []
    with slots_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["index"] = int(row["index"])
            row["seeing"] = float(row["seeing"])
            row["transparency"] = float(row["transparency"])
            row["sky_quality"] = float(row["sky_quality"])
            row["instrument_efficiency"] = float(row["instrument_efficiency"])
            row["closed"] = int(row["closed"])
            row["start_dt"] = _parse_utc(row["start_utc"])
            row["end_dt"] = _parse_utc(row["end_utc"])
            slots.append(row)
    events = []
    ev_path = data_dir / "events.csv"
    if ev_path.is_file():
        with ev_path.open("r", encoding="utf-8", newline="") as fh:
            events = list(csv.DictReader(fh))

    rows = [[t.target_id, t.ra_deg, t.dec_deg, t.target_class, t.feature_flux, t.science_weight, t.required]
            for t in card.targets]
    init = parse_initialize({"message_type": "initialize", "payload": {
        "task_card": {"card_id": card.root.name},
        "site": {"latitude_deg": card.fiber_config["site"]["latitude_deg"],
                 "longitude_deg": card.fiber_config["site"]["longitude_deg"],
                 "minimum_altitude_deg": 30.0, "sun_altitude_limit_deg": -18.0},
        "survey": {"start_utc": nights[0]["observing_start_utc"], "end_utc": nights[-1]["observing_end_utc"],
                   "slot_seconds": 900, "nights": nights},
        "instrument": {"n_fibers": card.n_fibers, "grid_side": round(card.n_fibers ** 0.5),
                       "fiber_area_deg2": card.fiber_config["field"]["fiber_area_deg2"],
                       "gap_deg": card.fiber_config["field"]["gap_deg"],
                       "exposure": card.fiber_config["exposure"]},
        "scoring": card.score_config, "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                                              "feature_flux", "science_weight", "required"], "rows": rows},
        "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32}}})

    cfg = None
    llm_planner = None
    if with_llm:
        from config import load_config
        from llm import LLMPlanner
        cfg = load_config()
        if cfg.llm_enabled:
            llm_planner = LLMPlanner(cfg.llm, planner_tool=PlannerTool(card), log=lambda s: None)
    planner = Planner(init, PlannerTool(card), llm_planner=llm_planner, log_fn=(lambda s: None))

    scorer = LocalScorer(init, slots, events, pointing_offset=pointing_offset)

    best = {}          # tid -> best c
    best_g = {}        # tid -> best g
    now = _parse_utc(nights[0]["observing_start_utc"])
    survey_end = _parse_utc(nights[-1]["observing_end_utc"])
    last_result = None
    remaining = 900.0
    wall = 1800.0
    observe_index = 0
    action_count = 0
    log = []

    while now < survey_end and action_count < rounds:
        action_count += 1
        req = DecisionRequest({"decision_sequence": action_count, "payload": {
            "now_utc": _format_utc(now), "survey_end_utc": _format_utc(survey_end),
            "observe_action_index": observe_index, "running_total": sum(best.values()),
            "wallclock": {"remaining_seconds": remaining, "remaining_real_cpu_seconds": remaining,
                          "wall_remaining_seconds": wall, "speed_factor": 1.0},
            "latest_bulletin": {"record_type": "bulletin", "slot_id": "S", "initial": action_count == 1, "notices": []},
            "latest_forecast": None, "active_requests": [], "new_messages": [], "last_result": last_result}})
        action = planner.decide(req)

        if action.type == "observe":
            # 用仿真器的"真值"判定命中并评分
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
            log.append({"seq": action_count, "type": "observe", "n": res["assigned"], "hits": res["hits"],
                        "exposure": action.exposure_seconds, "program": action.program})
            observe_index += 1
            step = action.exposure_seconds or 900
        elif action.type == "wait":
            last_result = {"action": "wait"}
            step = action.duration_seconds or 900
        elif action.type == "report":
            last_result = {"action": "report", "correct": False, "repaired": False, "score_delta": 0}
            step = 0
        else:
            break

        now += timedelta(seconds=step)
        remaining = max(0.0, remaining - 0.03)
        wall = max(0.0, wall - 0.05)

    # -- 汇总总分 ---------------------------------------------------------
    target_sum = sum(best.values())
    # 必观测漏掉
    required = [t for t in card.targets if t.required]
    required_missing = sum(1 for t in required if best_g.get(t.target_id, 0.0) < scorer.req_threshold)
    required_pen = scorer.req_penalty * required_missing
    # 均匀度 Jain
    J = _jain_uniformity(card, best_g, scorer.uni_band, scorer.uni_threshold)
    uniform_pen = scorer.uni_weight * (1.0 - J)
    total = target_sum - required_pen - uniform_pen

    return {
        "card": card.root.name,
        "actions": action_count,
        "observes": sum(1 for r in log if r["type"] == "observe"),
        "target_sum": round(target_sum, 2),
        "required_missing": required_missing,
        "required_penalty": round(required_pen, 2),
        "jain": round(J, 4),
        "uniform_penalty": round(uniform_pen, 2),
        "total": round(total, 2),
        "best_count": len(best),
        "log": log,
    }


def _jain_uniformity(card, best_g, band_width, threshold):
    import collections
    bands = collections.defaultdict(lambda: [0, 0])  # band -> [observed, total]
    for t in card.targets:
        b = int(t.ra_deg // band_width)
        bands[b][1] += 1
        if best_g.get(t.target_id, 0.0) >= threshold:
            bands[b][0] += 1
    ratios = [obs / tot for obs, tot in bands.values() if tot > 0]
    if not ratios or sum(ratios) == 0:
        return 0.0
    s = sum(ratios)
    return (s * s) / (len(ratios) * sum(r * r for r in ratios))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", default="alpha")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--d-alt", type=float, default=0.0, help="指向偏差 alt (度), Hard 模式模拟")
    parser.add_argument("--d-az", type=float, default=0.0, help="指向偏差 az (度)")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 汇总")
    args = parser.parse_args(argv)

    result = run_simulation(args.card, seed=args.seed, rounds=args.rounds,
                            pointing_offset=(args.d_alt, args.d_az), with_llm=args.with_llm)
    if args.json:
        r = dict(result); r.pop("log")
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f"卡片={result['card']}  动作={result['actions']} 观察={result['observes']}")
        print(f"  目标得分和        = {result['target_sum']}")
        print(f"  必观测漏掉        = {result['required_missing']}  (扣 {result['required_penalty']})")
        print(f"  均匀度 Jain       = {result['jain']}  (扣 {result['uniform_penalty']})")
        print(f"  ------------------------------")
        print(f"  本地合成天气总分  = {result['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
