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


def _harden_stdout() -> None:
    """确保 stdout **只**承载协议消息, 其余输出一律走 stderr.

    做法: 保存真实 stdout 作为协议流, 再把 ``sys.stdout`` 重定向到 ``sys.stderr``.
    这样即便我们的代码或第三方库 (openai/httpx 等) 在任意时刻意外 print, 也只会
    落到 stderr, 不会污染平台解析的协议通道. 所有日志/异常本就走 stderr.

    必须在**导入其它模块之前**尽早调用 (第三方库导入时可能打印), 故直接在本模块
    顶层执行.
    """
    real_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
    except Exception:  # noqa: BLE001 - 极端环境下重定向失败也不影响协议流
        pass
    return real_stdout


# 尽早硬化 stdout, 并把真实 stdout 绑定为协议输出流.
_REAL_STDOUT = _harden_stdout()

from config import load_config
from llm import LLMPlanner
from planner import Planner, PlannerTool, load_card_for
from protocol import (InitializeData, DecisionRequest, bind_protocol_stdout, log,
                      parse_initialize, parse_decision_request, read_messages, send_response)

bind_protocol_stdout(_REAL_STDOUT)




def _resolve_card(init_data: InitializeData):
    """尝试定位对应的任务卡 (供 preplan 复用); 找不到返回 None."""
    card_id = init_data.card_id
    if not card_id:
        slug = ""
    else:
        slug = card_id
    card = load_card_for(slug) if slug else None
    if card is None:
        # 回退: 用 public 目录下的目标直接规划 (无卡配置)
        log(f"agent: 未找到任务卡 '{slug}', 将仅使用协议 payload 的目标数据")
    return card


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
