#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""duty_lines.py - 从 messages.jsonl 的值班日志里抽出关键句, 便于人工/脚本核对解析结果.

用法:
    py tools/duty_lines.py --messages <run>/messages.jsonl --kind guider
    (--kind guider|flat|terrain|weather|all)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    "guider": re.compile(r"导星|ガイダー|guider", re.I),
    "flat": re.compile(r"平场|镜盖|flat|cover", re.I),
    "terrain": re.compile(r"山|稜線|棱线|遮挡|天頂距|天顶距"),
    "weather": re.compile(r"雨|雲|云|霾|寒|冷|storm|rain|haze|overcast"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", required=True)
    parser.add_argument("--kind", default="guider", choices=[*PATTERNS, "all"])
    args = parser.parse_args()

    pattern = None if args.kind == "all" else PATTERNS[args.kind]
    index = 0
    with open(args.messages, encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (obj.get("record_type") or obj.get("message_type")) != "observation_request":
                continue
            text = str(obj.get("reason") or "")
            if len(text) < 80:
                continue
            index += 1
            for raw in text.splitlines():
                sentence = raw.strip()
                if not sentence or (pattern is not None and not pattern.search(sentence)):
                    continue
                print(f"[{index:>2} {obj.get('issued_at_utc')}] {sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
