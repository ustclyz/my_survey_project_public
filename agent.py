#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent.py - 智能体主程序: 命令行入口 + 评测主循环.

协议: ``participant-agent-protocol-v4`` (jsonl-v4). stdin 每行一个 JSON 消息,
stdout 每行一个 ``decision_response``; 所有日志写 stderr.

消息处理:
    initialize        -> 构建 Planner (含 preplan 规划工具 + LLM 环节)
    decision_request  -> Planner.decide() -> 校验 -> decision_response
    finish            -> 打印运行摘要并退出

LLM 落点 (评奖硬门槛, 至少 2 个环节由 LLM 驱动):
    A. 任务规划 plan_night     (llm.py::LLMPlanner.plan_night)
    B. 行动决策 decide_action  (llm.py::LLMPlanner.decide_action)
    C. 计划自适应 diagnose_abnormal (加分)
无密钥时全部静态回退, 仍可跑通本地评测.

预算保护: 绝不每轮调 LLM, 仅在新夜/异常等关键决策点调用; 由 wallclock 控制.

无密钥时**不退出**, 退化为纯静态模式 (与官方示例不同), 便于本地调试.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.stderr.write("agent: 需要 Python 3.9 或更高版本\n")
    raise SystemExit(3)

from config import load_config
from llm import LLMPlanner
from planner import Planner, PlannerTool, load_card_for
from protocol import InitializeData, DecisionRequest, log, parse_initialize, parse_decision_request, read_messages, send_response


def _card_slug_candidates(card_id: str):
    """把平台下发的 card_id 映射为可能的卡片目录名 (按优先级去重).

    正式赛下发的 ``card_id`` 是 ``A`` / ``B`` / ``C`` / ``D`` / ``A1`` ...,
    而仓库中的卡片目录名是 ``cardA`` / ``cardB`` / ...; 练习卡则直接叫
    ``alpha`` / ``beta`` / ...。这里同时尝试"原样"与"加 card 前缀"两种写法,
    避免因命名不一致而误判为"找不到任务卡"。
    """
    raw = str(card_id or "").strip()
    if not raw:
        return []
    candidates = []
    for variant in (raw, raw.lower(), raw.upper(),
                    "card" + raw, "card" + raw.upper(), "card" + raw.lower()):
        if variant and variant not in candidates:
            candidates.append(variant)
    return candidates


def _resolve_card(init_data: InitializeData):
    """尝试定位对应的任务卡 (供 preplan 复用); 找不到返回 None."""
    card_id = init_data.card_id
    for slug in _card_slug_candidates(card_id):
        card = load_card_for(slug)
        if card is not None:
            log(f"agent: 任务卡已加载 card_id={card_id!r} -> {card.name!r}")
            return card
    log(f"agent: 未找到任务卡 {card_id!r} (尝试过 {_card_slug_candidates(card_id)}); "
        f"将使用协议 payload 中的目标与仪器参数")
    return None


def main() -> int:
    cfg = load_config()
    log(f"agent: {cfg.describe()}")
    if not cfg.openai_package_available and cfg.llm_enabled:
        log("agent: 未安装 openai 包, 将使用标准库 urllib 直连 /chat/completions")

    planner = None
    init_data = None

    for message in read_messages(sys.stdin):
        kind = message.get("message_type")

        if kind == "initialize":
            try:
                init_data = parse_initialize(message)
                card = _resolve_card(init_data)
                tool = PlannerTool(card)
                llm_planner = LLMPlanner(cfg.llm, planner_tool=tool, log=log) if cfg.llm_enabled else None
                if cfg.llm_enabled:
                    log(f"agent: LLM 后端 = {llm_planner.client.backend}")
                planner = Planner(init_data, tool, llm_planner=llm_planner, log_fn=log)
                log(f"agent: 初始化完成, 卡片={init_data.card_id or '(未知)'}, "
                    f"夜数={len(init_data.nights)}, 光纤={init_data.n_fibers}")
            except Exception as exc:  # noqa: BLE001 - 初始化失败不崩溃, 后续走安全回退
                log(f"agent: 初始化失败 ({type(exc).__name__}: {exc}); 后续决策走安全回退")
                planner = None

        elif kind == "decision_request":
            sequence = int(message.get("decision_sequence", 0))
            if planner is not None:
                try:
                    req = parse_decision_request(message)
                    action = planner.decide(req)
                except Exception as exc:  # noqa: BLE001 - 策略 bug 绝不能终止运行
                    log(f"agent: 决策异常 ({type(exc).__name__}: {exc}); 使用安全回退")
                    from models import Action

                    action = Action(type="wait", duration_seconds=900, reason="decision-exception",
                                    decision_source="fallback")
            else:
                from models import Action

                action = Action(type="wait", duration_seconds=900, reason="not-initialized",
                                decision_source="fallback")
            send_response(sequence, action)

        elif kind == "finish":
            payload = message.get("payload") or {}
            if planner is not None:
                try:
                    log(f"agent: finish termination_reason={payload.get('termination_reason')} "
                        f"observe_actions={payload.get('observe_actions')} "
                        f"reports={planner.reports_made} ")
                except Exception as exc:  # noqa: BLE001
                    log(f"agent: finish 摘要记录失败 ({type(exc).__name__}: {exc})")

        else:
            log(f"agent: 忽略未知消息类型 {kind!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
