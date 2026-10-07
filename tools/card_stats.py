#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""card_stats.py - 每张卡的目标规模/权重/必观测数/理论上限 (用于分榜差距分析)."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

BASE = Path(r"D:\WorkTable\codes\survey26")
SRC = {
    "A": BASE / "my_survey_project_public/cards/cardA/public/targets.csv",
    "B": BASE / "my_survey_project_public/cards/cardB/public/targets.csv",
    "C": BASE / "my_survey_project_public/cards/cardC/public/targets.csv",
    "D": BASE / "my_survey_project_public/cards/cardD/public/targets.csv",
    "A1": BASE / "hardcards/v4-a1-v5.targets.csv",
    "B1": BASE / "hardcards/v4-b1-v5.targets.csv",
    "C1": BASE / "hardcards/v4-c1-v5.targets.csv",
    "D1": BASE / "hardcards/v4-d1-v5.targets.csv",
}


def main() -> int:
    print(f"{'card':<4}{'N':>7}{'req':>6}{'sum_w':>10}{'cap(1.2w)':>11}{'mean_flux':>11}{'mean_w':>8}")
    for key in ("A", "B", "C", "D", "A1", "B1", "C1", "D1"):
        with SRC[key].open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        w = [float(r["science_weight"]) for r in rows]
        fl = [float(r["feature_flux"]) for r in rows]
        req = sum(1 for r in rows if str(r["required"]).lower() in ("true", "1"))
        print(f"{key:<4}{len(rows):>7}{req:>6}{sum(w):>10.0f}{1.2 * sum(w):>11.0f}"
              f"{statistics.mean(fl):>11.2f}{statistics.mean(w):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
