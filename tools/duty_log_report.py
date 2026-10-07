#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""duty_log_report.py - 从 agent.log 里抽出"值班日志 -> 决策"的证据链.

用法:
    py tools/duty_log_report.py <agent.log> [--full]

输出: 每轮值班日志解析结果(报修时刻数/禁报窗口/规避/地形/坏夜/优先/时长/时间价格),
      以及实际发出的报修(含是否到点)与报修结果统计。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERNS = {
    "duty_parse": re.compile(r"pro: duty log #"),
    "duty_report": re.compile(r"pro: duty-log report at"),
    "report_ok": re.compile(r"pro: report correct"),
    "report_bad": re.compile(r"pro: report false"),
    "suspected": re.compile(r"fault suspected"),
    "trim": re.compile(r"duty history 裁剪|duty history trimmed"),
    "night_plan": re.compile(r"llm night plan"),
    "fault_review": re.compile(r"llm fault review"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--full", action="store_true", help="打印每一行 duty log 解析结果")
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    counts = {key: 0 for key in PATTERNS}
    shown = 0
    for line in lines:
        for key, pattern in PATTERNS.items():
            if pattern.search(line):
                counts[key] += 1
                if key == "duty_parse" and (args.full or shown < 8):
                    print(line[:240])
                    shown += 1
                if key in ("duty_report", "trim"):
                    print("   ", line[:200])

    print("\n=== 统计 ===")
    print(f"值班日志解析轮次      : {counts['duty_parse']}")
    print(f"按日志排程发出的报修  : {counts['duty_report']}")
    print(f"报修正确 / 误报       : {counts['report_ok']} / {counts['report_bad']}")
    print(f"质量下限触发的疑似故障: {counts['suspected']}")
    print(f"历史裁剪次数          : {counts['trim']}")
    print(f"夜间计划 / 故障复核   : {counts['night_plan']} / {counts['fault_review']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
