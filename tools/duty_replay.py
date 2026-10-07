#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""duty_replay.py - 用一次真实平台运行里的值班日志回放多轮解析流程 (本地, 不联网).

目的: 验证"历史值班日志进入 LLM 请求"的改造在**真实日志序列**下的行为:
   * 逐条到达时, 每一轮的 messages 是否带着完整的 user/assistant 历史;
   * 超预算时是否按预期裁剪并保留继承状态;
   * 富字段(报修时刻/禁报窗口/方位/地形/坏夜)是否正确写进规划器与报事逻辑。

用法:
    py tools/duty_replay.py --messages <run>/messages.jsonl [--paylaod-from cardA]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import preplan  # noqa: E402
from pro.agent import ObserverAgent  # noqa: E402
from pro_sim import _build_init, _load_nights  # noqa: E402


class _StubCall:
    def __init__(self, answer):
        self.answer = answer

    def done(self):
        return True


class _StubClient:
    """不做网络调用: 把请求记下来, 返回一个"全量状态" (模拟模型输出)."""

    model = "stub"

    def __init__(self):
        self.requests: list = []
        self.n_users = 0

    def submit_messages(self, tag, messages, wallclock_left):
        self.requests.append(messages)
        self.n_users += sum(1 for m in messages if m["role"] == "user")
        answer = {"report_utc": ["2026-10-02T01:30:00Z"],
                  "no_report_utc": ["2026-10-02T03:00:00Z"],
                  "avoid_directions": ["SW"],
                  "terrain": [{"direction": "SW", "min_alt_deg": 37}],
                  "bad_nights": ["2026-10-20"],
                  "notes": f"turn {len(self.requests)}",
                  "reason": "stub"}
        return _StubCall(answer)

    def collect(self, call):
        return call.answer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", required=True, help="一次平台运行的 messages.jsonl")
    parser.add_argument("--card", default="cardA", help="提供仪器/计分配置的卡 (A1 与 A 相同)")
    parser.add_argument("--relay", action="store_true",
                        help="用平台 relay(kimi) 真机解析 (否则用 stub); 需要 ~/.config/survey26 的 token")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条 (>0 时生效)")
    args = parser.parse_args(argv)

    card = preplan.CardData.from_card(args.card)
    init = _build_init(card, _load_nights(card))
    init["site"]["utc_offset_hours"] = -4.0
    agent = ObserverAgent(init, rules_only=True)
    if args.relay:
        token = json.loads(Path(os.path.expanduser("~/.config/survey26/config.json"))
                           .read_text(encoding="utf-8")).get("token", "")
        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_BASE_URL"] = ("https://vdiemcofukuxglqsmlyz.supabase.co"
                                         "/functions/v1/kimi-relay/v1")
        os.environ["OPENAI_MODEL"] = "kimi-for-coding"
        from pro.llm_client import LLMClient
        agent.client = LLMClient(log=lambda text: print("[llm]", text[:160]), call_timeout=180.0)
        print("使用 relay 真机模型解析 (逐条, 串行)")
    else:
        agent.client = _StubClient()

    payloads = []
    with open(args.messages, encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (obj.get("record_type") or obj.get("message_type")) == "observation_request":
                payloads.append(obj)

    print(f"值班日志条数: {len(payloads)}")
    used = 0
    for idx, obj in enumerate(payloads, 1):
        text = str(obj.get("reason") or "")
        if len(text) < 80:
            continue
        if args.limit and used >= args.limit:
            break
        used += 1
        agent.duty_pending.append({"issued_at_utc": obj.get("issued_at_utc"), "duty_log": text})
        agent._duty_tick({"now_utc": obj.get("issued_at_utc")})
        if args.relay and agent.duty_call is not None:
            agent.duty_call.wait(200.0)
            agent._duty_tick({"now_utc": obj.get("issued_at_utc")})   # 收取上一轮结果
        if args.relay:
            print(f"  #{idx:>2} chars={agent._duty_chars():>7} report={len(agent.duty_times)} "
                  f"no_report={len(agent.duty_no_report)} avoid={sorted(agent.planner.extra_avoid)} "
                  f"terrain={agent.planner.terrain_min_alt} bad={len(agent.duty_bad_nights)} "
                  f"prefer={agent.planner.llm_prefer} dur={agent.planner.llm_duration_scale} "
                  f"lam={agent.planner.llm_lambda_scale}")
        else:
            msgs = agent.client.requests[-1]
            users = sum(1 for m in msgs if m["role"] == "user")
            assistants = sum(1 for m in msgs if m["role"] == "assistant")
            print(f"  #{idx:>2} chars={agent._duty_chars():>7}  msgs={len(msgs):>3} "
                  f"(user={users}, assistant={assistants})  report={len(agent.duty_times)} "
                  f"terrain={agent.planner.terrain_min_alt} avoid={sorted(agent.planner.extra_avoid)}")
    if args.relay:
        print("\n=== 最终解析出的报修时刻 (UTC) ===")
        for moment in agent.duty_times:
            print("  ", moment.strftime("%Y-%m-%dT%H:%M:%SZ"))
        print("notes:", str(agent.duty_state.get("notes") or "")[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
