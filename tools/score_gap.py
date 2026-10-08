#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_gap.py - 从一次平台运行的 observations.csv 反推得分损失构成.

对每张卡给出:
  * 每个目标"最好一次曝光"相对理论上限 1.2*w 的分布;
  * 若把欠曝目标补到上限可回收的分数;
  * 每次曝光实际耗时/得分的时间线(定位整段废片).

用法:
    py tools/score_gap.py --run <下载目录> --targets A --obs 1-a
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

CARD_TARGETS = {
    "A": BASE / "cards/cardA/public/targets.csv",
    "B": BASE / "cards/cardB/public/targets.csv",
    "C": BASE / "cards/cardC/public/targets.csv",
    "D": BASE / "cards/cardD/public/targets.csv",
}


def load_targets(path: Path):
    out = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["target_id"]] = (
                float(row["feature_flux"]),
                float(row["science_weight"]),
                str(row["required"]).lower() in ("true", "1"),
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="observations.csv")
    ap.add_argument("--targets", required=True, help="targets.csv")
    args = ap.parse_args(argv)

    targets = load_targets(Path(args.targets))
    best: dict[str, float] = {}
    nobs: Counter = Counter()
    by_index: dict[int, list[float]] = {}
    with open(args.obs, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["valid"] != "true":
                continue
            tid = row["target_id"]
            score = float(row["score"])
            nobs[tid] += 1
            if score > best.get(tid, -1.0):
                best[tid] = score
            if row["observe_index"]:
                by_index.setdefault(int(row["observe_index"]), []).append(score)

    sum_best = sum(best.values())
    n_obs = sum(nobs.values())
    print(f"targets={len(targets)} observed={len(best)} obs={n_obs} sum_best={sum_best:.1f}")
    buckets = Counter()
    for tid, (_f, w, _req) in targets.items():
        cap = 1.2 * w
        if tid not in best:
            buckets["never"] += 1
            continue
        ratio = best[tid] / cap if cap else 1.0
        for label, th in (("cap>=.999", 0.999), (".9+", 0.9), (".75+", 0.75), (".5+", 0.5)):
            if ratio >= th:
                buckets[label] += 1
                break
        else:
            buckets["<.5"] += 1
    print("buckets:", dict(buckets))
    pot = sum(1.2 * w - best.get(tid, 0.0) for tid, (_f, w, _r) in targets.items())
    print(f"potential gain to cap: {pot:.1f}")
    print("repeat histogram:", dict(sorted(Counter(nobs.values()).items())))

    # 时间段概览: 每 200 次曝光一行
    keys = sorted(by_index)
    if keys:
        print("\nexposeIdx-range  n   meanScore  zeroFrac")
        chunk = 200
        for start in range(keys[0], keys[-1] + 1, chunk):
            vals = []
            for k in range(start, start + chunk):
                vals.extend(by_index.get(k, ()))
            if not vals:
                continue
            zeros = sum(1 for v in vals if v <= 1e-9)
            print(f"{start:>6}-{start+chunk:<6} {len(vals):>5}  "
                  f"{sum(vals)/len(vals):>9.4f}  {zeros/len(vals):>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
