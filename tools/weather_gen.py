#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_gen.py - 本地"合成天气真值"生成器 (仅本地调试用).

背景 (关键):
    任务卡只下发 ``public/`` (目标/天区/夜历/计分配置), **不下发** ``truth/``
    (逐 slot 的 seeing / transparency / sky_quality / instrument_efficiency、
    天气事件、观测请求真值等). 正式卡 A-D 的天气更由云端运行时下发. 因此本地
    无法复现真实得分.

    本模块随机生成一份**语义与官方一致**的合成天气真值, 供本项目的本地评分器
    (``tools/local_scorer.py``) 使用, 从而在本地就能测出"在真实天气下会得多少分",
    并据此定位策略失分点. 它**不是**官方真值, 只是同构的随机模拟.

官方语义 (来自 docs/participant-guide):
    * 时间以 900s 的 slot 为网格 (UTC 刻钟对齐), 天气逐 slot 更新;
    * 四个隐藏标量, 均为"越大越好", 除 seeing 是"越小越好":
        - seeing              (角秒, 越小越好; 进入计分公式的**分母**)
        - transparency        (0-1, 越大越好)
        - sky_quality         (0-1, 越大越好; program 档位主要由它决定)
        - instrument_efficiency (0-1, 越大越好; 地震/故障会降低)
    * 事件: rain / storm / overcast / haze / cold_snap (定位到方向 sector),
      earthquake (降低后续若干夜 instrument_efficiency), rocket_launch 等;
    * 简报 notices 只给 event_kind + direction, 且**正常也可能无 notices
      (背景天气在无声关闭)**.

输出 (写入 ``sim_data/<card>/``):
    * weather_slots.csv   逐 slot 真值: slot_id,night_id,index,start_utc,end_utc,
                          seeing,transparency,sky_quality,instrument_efficiency,
                          closed(0/1),event_kind
    * events.csv          事件表: event_id,event_kind,start_slot,end_slot,
                          direction,az_min_deg,az_max_deg,alt_max_deg,intensity
    * meta.json           生成参数 (种子/统计), 便于复现.

纯标准库; 命令行:
    py tools/weather_gen.py --card alpha --seed 12345
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "sim_data"

SLOT_SECONDS = 900
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# 方向 -> 方位角中心 (度), 用于把事件落到扇区
DIR_AZ_CENTER = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                 "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
DIR_AZ_HALF = 45.0


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_nights(card_root: Path):
    nights = []
    with (card_root / "public" / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            nights.append(row)
    return nights


class WeatherGenerator:
    """生成一整段巡天的逐 slot 合成天气与事件.

    设计取向 (让天气"有结构、可复现、且与官方量级相近"):
        * sky_quality 用平滑的夜间趋势 (夜晚中段最好) + 随机扰动;
        * transparency 与 sky_quality 相关但有独立噪声;
        * seeing 随 sky_quality 变差而变差 (越差越大), 并叠加噪声;
        * instrument_efficiency 平时接近 1, 地震后按夜衰减恢复;
        * 低概率产生持续数 slot 的天气事件 (雨/暴/阴/霾/寒潮), 事件期间
          sky/transparency 下降、seeing 上升, 雨/暴整场关闭 (closed=1).
    """

    def __init__(self, card_root: Path, seed: int = 20261005):
        self.card_root = card_root
        self.rng = random.Random(seed)
        self.seed = seed
        self.nights = load_nights(card_root)
        self.scenario = self._load_scenario()
        self.site_lat = float(self.scenario.get("site", {}).get("latitude_deg", -24.6))

    def _load_scenario(self):
        path = self.card_root / "config" / "v4_scenario.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    # -- 事件调度 ----------------------------------------------------------
    def _plan_events(self, slots):
        """在整段巡天上随机布置少量事件.

        返回 ``(events, slot_event_map)``:
            events: list of dict
            slot_event_map: {slot_index: event_dict} (被事件覆盖的 slot)
        """
        events = []
        slot_event_map = {}
        n_slots = len(slots)
        # 事件数量: 与夜数相关, 但保持稀疏 (不影响大部分观测)
        n_weather_events = max(1, n_slots // 220)   # 约每 ~2.3 夜一个
        n_eq = max(1, len(self.nights) // 60)       # 少见的"地震"影响 instrument
        n_rocket = max(0, n_slots // 500)

        # 天气事件 (有方向扇区, 持续 2-8 slot)
        for k in range(n_weather_events):
            kind = self.rng.choice(["overcast", "haze", "rain", "storm", "cold_snap", "overcast"])
            start = self.rng.randrange(0, max(1, n_slots - 1))
            length = self.rng.randint(2, 8)
            end = min(n_slots - 1, start + length)
            direction = self.rng.choice(DIRECTIONS + ["ALL", "ALL"])
            if kind in ("rain", "storm"):
                direction = "ALL"  # 雨/暴常全天关闭
            ev = {
                "event_id": f"EV{k + 1:04d}",
                "event_kind": kind,
                "start_slot": start,
                "end_slot": end,
                "direction": direction,
                "az_min_deg": (DIR_AZ_CENTER.get(direction, 0.0) - DIR_AZ_HALF) % 360.0,
                "az_max_deg": (DIR_AZ_CENTER.get(direction, 0.0) + DIR_AZ_HALF) % 360.0,
                "alt_max_deg": round(self.rng.uniform(25.0, 60.0), 2),
                "intensity": round(self.rng.uniform(0.3, 0.9), 3),
            }
            events.append(ev)
            for s in range(start, end + 1):
                slot_event_map[s] = ev

        # 地震: 从某夜开始, 降低后续若干夜的 instrument_efficiency
        for k in range(n_eq):
            night_idx = self.rng.randrange(0, max(1, len(self.nights) - 3))
            events.append({
                "event_id": f"EQ{k + 1:04d}",
                "event_kind": "earthquake",
                "start_slot": night_idx,          # 复用字段: 表示"第几个夜开始"
                "end_slot": min(len(self.nights) - 1, night_idx + self.rng.randint(2, 5)),
                "direction": "ALL", "az_min_deg": 0.0, "az_max_deg": 360.0,
                "alt_max_deg": 90.0, "intensity": round(self.rng.uniform(0.1, 0.4), 3),
            })

        # 火箭发射: 短暂方向性影响
        for k in range(n_rocket):
            start = self.rng.randrange(0, max(1, n_slots - 1))
            events.append({
                "event_id": f"RK{k + 1:04d}", "event_kind": "rocket_launch",
                "start_slot": start, "end_slot": min(n_slots - 1, start + 1),
                "direction": self.rng.choice(DIRECTIONS),
                "az_min_deg": 0.0, "az_max_deg": 360.0, "alt_max_deg": 30.0,
                "intensity": 1.0,
            })
        return events, slot_event_map

    # -- 生成 --------------------------------------------------------------
    def generate(self):
        slots = []
        for night in self.nights:
            start = _parse_utc(night["observing_start_utc"])
            end = _parse_utc(night["observing_end_utc"])
            count = int(night.get("slot_count") or 0)
            if count <= 0:
                count = int(max(1, (end - start).total_seconds() // SLOT_SECONDS))
            for i in range(count):
                s0 = start + timedelta(seconds=i * SLOT_SECONDS)
                s1 = min(end, s0 + timedelta(seconds=SLOT_SECONDS))
                slots.append({
                    "night_id": night["night_id"],
                    "night_date": night["night_date"],
                    "index_in_night": i,
                    "n_in_night": count,
                    "start_utc": s0,
                    "end_utc": s1,
                })

        events, slot_event_map = self._plan_events(slots)

        # 地震 -> 每夜的 instrument_efficiency 衰减
        eq_efficiency = {}
        for ev in events:
            if ev["event_kind"] != "earthquake":
                continue
            base = 1.0 - ev["intensity"]
            for k in range(ev["start_slot"], ev["end_slot"] + 1):
                if k < len(self.nights):
                    eq_efficiency[k] = min(eq_efficiency.get(k, 1.0), base)

        rows = []
        # 逐夜平滑趋势: 用 AR(1) 让 sky_quality 缓慢漂移
        sq = self.rng.uniform(0.45, 0.8)
        tr = self.rng.uniform(0.6, 0.95)
        night_counter = -1
        prev_night = None
        for idx, slot in enumerate(slots):
            if slot["night_id"] != prev_night:
                prev_night = slot["night_id"]
                night_counter += 1
                # 每夜重置一个"基准"
                night_base = self.rng.uniform(0.5, 0.85)
                sq = night_base
                tr = self.rng.uniform(0.65, 0.95)
            # 夜晚中段最好 (U 形: 用 sin 拟合)
            frac = slot["index_in_night"] / max(1, slot["n_in_night"] - 1)
            mid_bonus = 0.12 * math.sin(math.pi * frac)
            # AR(1) 漂移
            sq = 0.85 * sq + 0.15 * (night_base + mid_bonus) + self.rng.gauss(0, 0.02)
            tr = 0.85 * tr + 0.15 * (0.8 + mid_bonus) + self.rng.gauss(0, 0.02)
            sky = max(0.05, min(1.0, sq))
            transp = max(0.05, min(1.0, tr))
            # seeing: 天空越差 seeing 越差 (数值越大)
            seeing = max(0.4, 1.6 - 1.0 * sky + self.rng.gauss(0, 0.15))
            instr = eq_efficiency.get(night_counter, 1.0) * (1.0 + self.rng.gauss(0, 0.01))
            instr = max(0.2, min(1.0, instr))

            closed = 0
            ev_kind = ""
            ev = slot_event_map.get(idx)
            if ev is not None:
                ev_kind = ev["event_kind"]
                k = ev["intensity"]
                if ev["event_kind"] in ("rain", "storm"):
                    closed = 1
                    sky *= max(0.0, 1.0 - k)
                    transp *= max(0.0, 1.0 - k)
                elif ev["event_kind"] == "overcast":
                    sky *= max(0.05, 1.0 - 0.9 * k)
                    transp *= max(0.05, 1.0 - 0.85 * k)
                    seeing += 1.0 * k
                elif ev["event_kind"] == "haze":
                    transp *= max(0.1, 1.0 - 0.6 * k)
                    seeing += 0.6 * k
                elif ev["event_kind"] == "cold_snap":
                    instr *= max(0.3, 1.0 - 0.4 * k)
                elif ev["event_kind"] == "rocket_launch":
                    transp *= 0.7

            rows.append({
                "slot_id": f"{slot['night_id']}-S{slot['index_in_night'] + 1:03d}",
                "night_id": slot["night_id"],
                "index": idx,
                "start_utc": _fmt(slot["start_utc"]),
                "end_utc": _fmt(slot["end_utc"]),
                "seeing": round(seeing, 4),
                "transparency": round(transp, 4),
                "sky_quality": round(sky, 4),
                "instrument_efficiency": round(instr, 4),
                "closed": closed,
                "event_kind": ev_kind,
            })
        return rows, events

    # -- 落盘 --------------------------------------------------------------
    def write(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, events = self.generate()
        with (out_dir / "weather_slots.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (out_dir / "events.csv").open("w", encoding="utf-8", newline="") as fh:
            if events:
                w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
                w.writeheader()
                w.writerows(events)
            else:
                fh.write("event_id,event_kind,start_slot,end_slot,direction,az_min_deg,az_max_deg,alt_max_deg,intensity\n")
        meta = {
            "card": self.card_root.name,
            "seed": self.seed,
            "n_slots": len(rows),
            "n_events": len(events),
            "closed_slots": sum(1 for r in rows if r["closed"]),
            "sky_quality_mean": round(sum(r["sky_quality"] for r in rows) / len(rows), 4),
            "seeing_mean": round(sum(r["seeing"] for r in rows) / len(rows), 4),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", default="alpha", help="卡片名或路径")
    parser.add_argument("--seed", type=int, default=20261005, help="随机种子 (可复现)")
    parser.add_argument("--out", type=Path, default=None, help="输出目录 (默认 sim_data/<card>)")
    args = parser.parse_args(argv)

    card_root = Path(args.card)
    if not card_root.is_dir():
        card_root = PROJECT_ROOT / "cards" / args.card
    if not card_root.is_dir():
        print(f"[错误] 找不到卡片: {args.card}", file=__import__("sys").stderr)
        return 1

    out_dir = args.out or (DEFAULT_OUT / card_root.name)
    meta = WeatherGenerator(card_root, seed=args.seed).write(out_dir)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
