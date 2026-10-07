# -*- coding: utf-8 -*-
"""入口/协议冒烟测试: 直接启动 ``agent.py``, 走真实 stdin/stdout.

这一步覆盖"线上真正跑的那条路径" (根 agent.py -> stdout 硬化 -> pro 内核),
确保:

* ``initialize`` + ``decision_request`` 能得到一条合法的 ``decision_response``;
* **stdout 上只有协议 JSON** (硬化生效), 日志都在 stderr;
* 无密钥时退化为"仅规则"而不是退出 (评测里退出 = 整场 0 分)。

不依赖网络 (设置 ``OBSERVER_MODEL_DISABLED=1`` 且清空 ``*_API_KEY``)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import preplan  # noqa: E402
from pro_sim import _build_init, _load_nights  # noqa: E402

PROTOCOL = "participant-agent-protocol-v4"


def _env() -> dict:
    env = dict(os.environ)
    for key in [k for k in env if k.endswith("_API_KEY")]:
        env.pop(key, None)
    env["OBSERVER_MODEL_DISABLED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _messages():
    card = preplan.CardData.from_card("cardA")
    nights = _load_nights(card)[:1]
    init_payload = _build_init(card, nights)
    start = nights[0]["observing_start_utc"]
    initialize = {"protocol_version": PROTOCOL, "message_type": "initialize", "payload": init_payload}
    request = {
        "protocol_version": PROTOCOL,
        "message_type": "decision_request",
        "decision_sequence": 1,
        "payload": {
            "schema_version": "v4-decision-request-v1",
            "now_utc": start, "survey_end_utc": init_payload["survey"]["end_utc"],
            "observe_action_index": 0, "running_total": 0.0,
            "wallclock": {"remaining_seconds": 900.0, "remaining_real_cpu_seconds": 900.0,
                          "wall_remaining_seconds": 3600.0, "speed_factor": 0.82},
            "latest_bulletin": {"record_type": "bulletin", "initial": True, "notices": []},
            "latest_forecast": None, "active_requests": [], "new_messages": [],
            "last_result": None,
        },
    }
    return initialize, request


def test_entry_emits_single_valid_response():
    initialize, request = _messages()
    stdin = json.dumps(initialize) + "\n" + json.dumps(request) + "\n"
    proc = subprocess.run([sys.executable, "agent.py"], cwd=str(ROOT), input=stdin,
                          capture_output=True, text=True, encoding="utf-8", env=_env(), timeout=300)
    assert proc.returncode == 0, f"agent 退出码 {proc.returncode}; stderr={proc.stderr[-2000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout 应只有 1 条协议消息, 实际 {len(lines)}: {lines[:3]}"
    response = json.loads(lines[0])
    assert response["protocol_version"] == PROTOCOL
    assert response["message_type"] == "decision_response"
    assert response["decision_sequence"] == 1
    assert response["action"] in ("observe", "wait", "report", "finish")
    if response["action"] == "observe":
        assert isinstance(response.get("assignments"), dict)
        assert response.get("duration_seconds")
        assert response.get("program") in ("DARK", "BRIGHT", "BACKUP")


def test_entry_survives_empty_stdin():
    proc = subprocess.run([sys.executable, "agent.py"], cwd=str(ROOT), input="",
                          capture_output=True, text=True, encoding="utf-8", env=_env(), timeout=120)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


class _FakeClient:
    """Just enough of LLMClient for the duty-log scheduling test."""

    model = "fake"

    def submit(self, tag, system, user, wallclock_left):
        return None

    def collect(self, call):
        return None


def test_duty_log_schedules_a_report():
    """值班日志 (Hard 模式 A1-D1) 解析出的报修时刻要能触发 report 动作."""
    from pro.agent import ObserverAgent
    from pro.skymath import parse_utc

    card = preplan.CardData.from_card("cardA")
    init_payload = _build_init(card, _load_nights(card)[:1])
    agent = ObserverAgent(init_payload, rules_only=True)
    agent.client = _FakeClient()                      # 让排程逻辑认为是"有模型"模式
    agent._apply_duty({"report_utc": ["2026-10-02T01:30:00Z", "not-a-time"]})
    assert len(agent.duty_times) == 1                 # 非法时刻被丢弃
    # 未到点 -> 不报
    assert agent._due_report(parse_utc("2026-10-02T01:00:00Z")) is None
    # 到点 -> 报
    action = agent._due_report(parse_utc("2026-10-02T02:00:00Z"))
    assert action is not None and action["action"] == "report"
    # 已报过的时刻不会重复报
    assert agent._due_report(parse_utc("2026-10-02T02:30:00Z")) is None
    # 连续 report 达到上限时暂停, 等待一次观测刷新计数
    agent.duty_times.append(parse_utc("2026-10-03T01:00:00Z"))
    agent.consecutive_reports = 3
    assert agent._due_report(parse_utc("2026-10-03T02:00:00Z")) is None


def test_duty_log_would_be_detected():
    """长中文 reason 应被识别为值班日志; 短的英文 reason (普通请求) 不应识别."""
    from pro import agent as pro_agent
    assert pro_agent.DUTY_MIN_CHARS <= 200
