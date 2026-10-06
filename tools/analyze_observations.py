#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_observations.py - 分析平台结果里的 observations.csv / decisions.csv.

回答: 每目标最好成绩、档位匹配率、曝光时长分布、未饱和比例、时间都花在哪。

用法:
    python tools/analyze_observations.py <dir_with_observations_csv>
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _find(root: Path, name: str):
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    obs = _find(root, "observations.csv")
    dec = _find(root, "decisions.csv")
    if not obs:
        print("no observations.csv")
        return 1

    per_target_best = defaultdict(float)
    per_target_best_factor = {}
    per_target_best_mult = {}
    prog_mult = Counter()
    n_obs = 0
    valid = 0
    for row in csv.DictReader(obs.open("r", encoding="utf-8", newline="")):
        n_obs += 1
        if str(row.get("valid", "")).lower() != "true":
            continue
        valid += 1
        m = float(row.get("prog_mult", 0) or 0)
        prog_mult[round(m, 2)] += 1
        s = float(row.get("score", 0) or 0)
        tid = row.get("target_id")
        if s > per_target_best[tid]:
            per_target_best[tid] = s
            try:
                per_target_best_factor[tid] = float(row.get("factor", 0) or 0)
            except ValueError:
                per_target_best_factor[tid] = 0.0
            per_target_best_mult[tid] = m

    print(f"observations rows: {n_obs}, valid: {valid}, distinct targets: {len(per_target_best)}")
    print(f"total best score (Σ per-target best) = {sum(per_target_best.values()):.1f}")
    tot = sum(prog_mult.values()) or 1
    print("program multiplier distribution (valid exposures):")
    for k in sorted(prog_mult):
        print(f"   m={k:<5} {prog_mult[k]:>7}  ({prog_mult[k] / tot * 100:.1f}%)")
    mismatch = sum(v for k, v in prog_mult.items() if abs(k - 1.0) < 1e-9)
    print(f"   mismatch share = {mismatch / tot * 100:.1f}%")

    # 饱和分布: 每目标"最好那次"的完成因子 g
    facs = sorted(per_target_best_factor.values())
    if facs:
        n = len(facs)
        sat = sum(1 for f in facs if f >= 0.99)
        near = sum(1 for f in facs if 0.8 <= f < 0.99)
        mid = sum(1 for f in facs if 0.5 <= f < 0.8)
        low = sum(1 for f in facs if f < 0.5)
        print(f"\nper-target best factor g: median={facs[n//2]:.3f} mean={sum(facs)/n:.3f}")
        print(f"   g>=0.99 (saturated): {sat} ({sat/n*100:.1f}%)")
        print(f"   0.8<=g<0.99        : {near} ({near/n*100:.1f}%)")
        print(f"   0.5<=g<0.8         : {mid} ({mid/n*100:.1f}%)")
        print(f"   g<0.5              : {low} ({low/n*100:.1f}%)")
        # 潜在增益: 若把每个目标的 g 抬到 1 (不可能, 但给出上界)
        gain = 0.0
        for t in per_target_best:
            f = per_target_best_factor.get(t, 0.0)
            m = per_target_best_mult.get(t, 0.0)
            if f > 0 and m > 0:
                score_at_g1 = per_target_best[t] / (f * m) * m  # w*m
                gain += score_at_g1 - per_target_best[t]
        print(f"   upper bound if all g->1 (keeping w*m): +{gain:.0f}")

    if dec:
        durs = []
        progs = Counter()
        rows = list(csv.DictReader(dec.open("r", encoding="utf-8", newline="")))
        for r in rows:
            if r.get("action") != "observe":
                continue
            progs[r.get("program", "?")] += 1
            try:
                durs.append(int(r.get("duration_seconds") or 0))
            except ValueError:
                pass
        if durs:
            durs.sort()
            print(f"\nobserve actions: {len(durs)}  durations: min={durs[0]} "
                  f"p25={durs[len(durs)//4]} median={statistics.median(durs)} "
                  f"p75={durs[3*len(durs)//4]} max={durs[-1]} mean={statistics.mean(durs):.0f}")
            c = Counter(durs)
            print("   top durations: " + ", ".join(f"{k}s×{v}" for k, v in c.most_common(8)))
        print("declared programs: " + ", ".join(f"{k}={v}" for k, v in progs.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
