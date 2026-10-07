# -*- coding: utf-8 -*-
"""回归测试: 仪器故障的"绝对质量下限"判据不能被 E 表的过期早退锁死.

线上实测 (卡 A1, eval 1fdc6d60): 导星相机故障把命中分数压到 ~0.05x, 于是
``planner.band_obs`` 不再更新, ``clean_e()`` 返回 None, ``planner.e_hours`` 冻结;
``_fault_verdict`` 开头的 "rows 过期就返回 False" 便永久阻止报修 —— 从 2026-11-14
起近 90 个夜晚数据全废 (sum_best 7853, 总分 42)。

这里固定住那个场景: E 表为空/过期, 但近几小时 scale 远低于晴夜模型时, 仍必须判故障。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import preplan  # noqa: E402
from pro_sim import _build_init, _load_nights  # noqa: E402

from pro.agent import ObserverAgent  # noqa: E402


def _agent() -> ObserverAgent:
    card = preplan.CardData.from_card("cardA")
    nights = _load_nights(card)[:3]
    return ObserverAgent(_build_init(card, nights), rules_only=True)


def _frozen_fault(agent: ObserverAgent, hours: float = 5.0) -> None:
    """复刻故障持续中的状态: 质量塌到 0.05, E 表已停止更新."""
    agent.hard_mode = True          # 绝对判据只在 Hard 模式 (A1-D1) 启用
    agent.ref_from_hours = hours - 3.0
    agent.planner.e_hours = []
    agent.planner.scale = 0.05
    agent.scale_hours = {int(hours) - 1: [0.05, 0.05], int(hours): [0.05, 0.05]}


def test_normal_cards_never_use_the_aggressive_criterion():
    """普通卡 (A-D) 没有值班日志, 低质量按天气处理, 避免连续误报 (-150/次)."""
    agent = _agent()
    _frozen_fault(agent)
    agent.hard_mode = False
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is False


def test_scale_fault_reports_even_when_e_table_is_frozen():
    agent = _agent()
    _frozen_fault(agent)
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is True


def test_all_sky_blackout_is_not_treated_as_a_fault():
    agent = _agent()
    _frozen_fault(agent)
    agent.planner.notices = {("rain", "ALL")}
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is False


def test_healthy_scale_never_reports():
    agent = _agent()
    _frozen_fault(agent)
    agent.planner.scale = 0.8
    agent.scale_hours = {4: [0.8, 0.8], 5: [0.8, 0.8]}
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is False


def test_false_report_blocks_immediate_repeats():
    """一次误报后必须封锁: A-D 上一小时的低质量不能变成连续误报 (每次 -150)."""
    agent = _agent()
    _frozen_fault(agent)
    agent.episode_blocked = True
    agent.blocked_at_hour = 3
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is False
    # 但质量确实回来了 -> 解锁
    agent.planner.scale = 0.8
    agent.scale_hours = {4: [0.8], 5: [0.8]}
    assert agent._fault_verdict(5.0, {"now_utc": "2026-10-02T05:00:00Z"}) is False
    assert agent.episode_blocked is False


def test_stuck_quality_rearms_a_probe_after_a_long_block():
    agent = _agent()
    _frozen_fault(agent, hours=30.0)
    agent.episode_blocked = True
    agent.blocked_at_hour = 3
    agent.scale_hours = {29: [0.05], 30: [0.05]}
    assert agent._fault_verdict(30.0, {"now_utc": "2026-10-03T06:00:00Z"}) is True
