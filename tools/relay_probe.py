#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""relay_probe.py - 用平台 relay 验证 LLM 客户端 (本地开发用, 不联网到评测平台)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    token = json.loads(Path(os.path.expanduser("~/.config/survey26/config.json")).read_text(encoding="utf-8"))["token"]
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_BASE_URL"] = "https://vdiemcofukuxglqsmlyz.supabase.co/functions/v1/kimi-relay/v1"
    os.environ["OPENAI_MODEL"] = "kimi-for-coding"
    from pro.llm_client import LLMClient

    client = LLMClient(log=lambda text: print("[client]", text[:200]))
    messages = [{"role": "system", "content": "只输出一个 JSON 对象, 不要解释。"},
                {"role": "user", "content": "请返回 {\"ok\": true, \"n\": 3}"}]
    try:
        print("OK", client._request(messages, 60.0))
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode("utf-8", "replace")[:800])
    except Exception as exc:  # noqa: BLE001
        print("ERR", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
