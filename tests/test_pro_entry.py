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


# ---------------------------------------------------------------------------
# 值班日志: 历史入请求 / 全量状态累积 / 富字段接规划器 / 禁报窗口
# ---------------------------------------------------------------------------


class _FakeCall:
    def __init__(self, answer):
        self.answer = answer

    def done(self):
        return True


class _RecordingClient:
    """记录每次 submit_messages 的完整 messages 数组, 并立即返回预设答案."""

    model = "fake"

    def __init__(self, answers=None):
        self.requests: list = []
        self.answers = list(answers or [])
        self.ok = 0
        self.failed = 0

    def submit_messages(self, tag, messages, wallclock_left, max_tokens=None):
        self.requests.append(messages)
        return _FakeCall(self.answers.pop(0) if self.answers else None)

    def collect(self, call):
        return call.answer


def _duty_agent(answers=None):
    from pro.agent import ObserverAgent
    card = preplan.CardData.from_card("cardA")
    agent = ObserverAgent(_build_init(card, _load_nights(card)[:1]), rules_only=True)
    agent.client = _RecordingClient(answers)
    return agent


def test_duty_log_keeps_history_in_llm_request():
    """目标要求: 历史值班日志必须出现在 LLM 请求的历史记录里, 且状态是全量累积的."""
    agent = _duty_agent([
        {"report_utc": ["2026-10-02T01:30:00Z"], "notes": "first"},
        {"report_utc": ["2026-10-02T01:30:00Z", "2026-10-03T02:00:00Z"], "notes": "second"},
    ])
    payload = {"now_utc": "2026-10-02T00:00:00Z"}

    agent.duty_pending = [{"issued_at_utc": "2026-10-02T00:00:00Z", "duty_log": "LOG-ONE"}]
    agent._duty_tick(payload)
    assert len(agent.client.requests) == 1
    first = agent.client.requests[0]
    assert first[0]["role"] == "system" and "LOG-ONE" in json.dumps(first, ensure_ascii=False)

    agent.duty_pending = [{"issued_at_utc": "2026-10-03T00:00:00Z", "duty_log": "LOG-TWO"}]
    agent._duty_tick(payload)
    assert len(agent.client.requests) == 2
    second = agent.client.requests[1]
    roles = [m["role"] for m in second]
    assert roles[0] == "system"
    assert "assistant" in roles                      # 上一轮回复作为历史一起发
    blob = json.dumps(second, ensure_ascii=False)
    assert "LOG-ONE" in blob and "LOG-TWO" in blob   # 新旧日志都在请求里
    assert len(agent.duty_times) == 1                # 第一轮的全量状态已生效

    agent.duty_pending = [{"issued_at_utc": "2026-10-04T00:00:00Z", "duty_log": "LOG-THREE"}]
    agent._duty_tick(payload)
    assert len(agent.duty_times) == 2                # 第二轮的全量状态覆盖并累积


def test_duty_log_rich_fields_reach_planner():
    """日志解析出的规避方向/地形阈值/坏夜/禁报窗口要真正驱动决策."""
    from pro.skymath import parse_utc

    agent = _duty_agent()
    agent._apply_duty({
        "report_utc": [],
        "no_report_utc": ["2026-10-02T03:00:00Z"],
        "avoid_directions": ["SW"],
        "terrain": [{"direction": "SE", "min_alt_deg": 32}],
        "bad_nights": ["2026-10-20"],
        "prefer_directions": [{"direction": "E", "weight": 0.8}],
        "duration_scale": 1.3,
        "lambda_scale": 0.7,
        "notes": "keep going",
    })
    assert "SW" in agent.planner.extra_avoid
    assert agent.planner.terrain_min_alt.get("SE") == 32.0
    assert agent.duty_bad_nights == {"2026-10-20"}
    assert agent.planner.llm_prefer.get("E") == 0.8
    assert abs(agent.planner.llm_duration_scale - 1.3) < 1e-9
    assert abs(agent.planner.llm_lambda_scale - 0.7) < 1e-9
    # 地形阈值真的进了方向因子
    assert agent.planner._direction_factor(25.0, 135.0) == 0.0    # SE, 低于 32 度
    assert agent.planner._direction_factor(60.0, 135.0) > 0.0     # 高于阈值仍可用
    # 禁报窗口
    assert agent._in_no_report_window(parse_utc("2026-10-02T03:20:00Z")) is True
    assert agent._in_no_report_window(parse_utc("2026-10-02T06:00:00Z")) is False
    # 非法时刻/非法字段不应抛异常
    agent._apply_duty({"report_utc": ["nope"], "terrain": [{"direction": "XX", "min_alt_deg": "bad"}]})
    assert agent.duty_times == []


def test_llm_directives_actually_change_decisions():
    """LLM 的旋钮必须真的改变规划器行为 (不是"知道了但改不动")."""
    agent = _duty_agent()
    planner = agent.planner
    agent._apply_duty({"prefer_targets": [planner.ids[0]], "duration_scale": 1.5, "lambda_scale": 0.5,
                       "prefer_directions": [{"direction": "E", "weight": 0.9}]})
    assert planner.llm_target_boost.get(planner.ids[0]) == 2.0     # 目标价值被抬高
    assert planner.value(0) > 0.0
    assert planner._direction_preference(45.0, 90.0) == 0.9        # E = 方位角 90
    assert planner._direction_preference(45.0, 270.0) == 0.0
    info = {0: (45.0, 90.0, 0.8, 0.8, 100000.0, 1.0)}   # info 以目标下标为键
    assert planner._scaled_duration(600, {0: 0}, info, 1e9) == 900  # 1.5x, 30s 对齐
    assert planner._scaled_duration(3600, {0: 0}, info, 500) == 500  # 不超剩余窗口
    # 越界值被夹紧
    agent._apply_duty({"duration_scale": 99, "lambda_scale": 0.0})
    assert planner.llm_duration_scale == 2.0 and planner.llm_lambda_scale == 0.3


def test_no_report_window_suppresses_false_report():
    """平场灯/镜盖测试时段即使判据成立也不许报修 (报了算误报)."""
    agent = _duty_agent()
    agent._apply_duty({"no_report_utc": ["2026-10-02T03:00:00Z"]})
    agent._fault_verdict = lambda *a, **k: True      # 强制"判据成立"
    assert agent._maybe_report(1000.0, {"now_utc": "2026-10-02T03:10:00Z"}) is None
    out = agent._maybe_report(1000.0, {"now_utc": "2026-10-05T03:10:00Z"})
    assert out is not None and out["action"] == "report"


def test_llm_client_builds_multi_turn_request():
    """LLM 客户端必须把完整 messages(含多轮历史)原样发给 /chat/completions."""
    import urllib.request
    from pro.llm_client import LLMClient

    client = LLMClient(log=lambda *_: None)
    captured = {}

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"a\": 1}"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Resp()

    original = urllib.request.urlopen
    urllib.request.urlopen = _fake_urlopen
    try:
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
        out = client._request(messages, 10.0)
    finally:
        urllib.request.urlopen = original
    assert out == {"a": 1}
    assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user", "assistant", "user"]


def test_llm_client_honours_per_call_max_tokens():
    """值班日志调用要能单独放大 max_tokens (默认 2000 会被推理吃光)."""
    import urllib.request
    from pro.llm_client import LLMClient

    client = LLMClient(log=lambda *_: None)
    captured = {}

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": 1}"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    original = urllib.request.urlopen
    urllib.request.urlopen = lambda request, timeout=None: (captured.update(
        body=json.loads(request.data.decode("utf-8"))) or _Resp())
    try:
        client._request([{"role": "user", "content": "x"}], 10.0, 6000)
    finally:
        urllib.request.urlopen = original
    assert captured["body"]["max_tokens"] == 6000
    assert client.max_tokens == 2000          # 默认值不受影响


def test_duty_history_is_trimmed_but_state_survives():
    """历史超预算时丢弃最早的轮次, 但要把上一版全量状态作为继承放在最前面."""
    from pro import agent as pro_agent

    agent = _duty_agent()
    agent.duty_state = {"report_utc": ["2026-10-02T01:30:00Z"], "notes": "carried"}
    agent.duty_history = [
        {"role": "user", "content": "x" * 30000},
        {"role": "assistant", "content": "y" * 30000},
        {"role": "user", "content": "z" * 30000},
    ]
    agent._trim_duty_history()
    assert agent._duty_chars() <= pro_agent.DUTY_HISTORY_CHARS + len(
        json.dumps({"carried_state": agent.duty_state, "note": "更早的原始日志已省略; 请以上一版全量状态为准继续累积"},
                   ensure_ascii=False, separators=(",", ":")))
    assert agent.duty_history[0]["role"] == "user"
    assert "carried_state" in agent.duty_history[0]["content"]


def test_llm_extract_json_handles_reasoning_models():
    """推理型模型 content 为空、答案在 reasoning_content 时也要能解析出来."""
    from pro.llm_client import _extract_json

    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('前言 {"a": 1} 后语') == {"a": 1}
    # 推理文本里混着多个 JSON 片段: 取最后一个能解析的
    messy = '我先列个草稿 {"draft": 1} 不对; 最终答案 {"report_utc": ["2026-10-02T01:30:00Z"]}'
    assert _extract_json(messy) == {"report_utc": ["2026-10-02T01:30:00Z"]}
    assert _extract_json("没有 JSON") is None
    assert _extract_json("") is None
    # 字符串里有裸换行 + 对象尾部多逗号 -> 修复后仍能解析 (LLM 最常见的两种坏 JSON)
    broken = '{"report_utc": [], "notes": "a\nb",}'
    parsed = _extract_json(broken)
    assert parsed is not None and parsed["report_utc"] == [] and parsed["notes"] == "a b"


def test_llm_client_uses_reasoning_content_when_content_empty():
    """端到端: content 为空 + reasoning_content 有 JSON -> 仍然拿到答案."""
    import urllib.request
    from pro.llm_client import LLMClient

    client = LLMClient(log=lambda *_: None)

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": "",
                "reasoning_content": '思考中... 最终 {"report_utc": ["2026-10-02T01:30:00Z"]}',
            }}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    original = urllib.request.urlopen
    urllib.request.urlopen = lambda request, timeout=None: _Resp()
    try:
        out = client._request([{"role": "system", "content": "S"}], 10.0)
    finally:
        urllib.request.urlopen = original
    assert out == {"report_utc": ["2026-10-02T01:30:00Z"]}


def test_duty_llm_sees_run_state_and_can_demand_report():
    """值班日志 LLM 不仅读日志, 还能看到现场态势并直接要求"立即报修"."""
    agent = _duty_agent()
    snapshot = agent._run_state_snapshot({"now_utc": "2026-10-02T01:00:00Z"})
    for key in ("now_utc", "nights_left", "reports", "quality", "notices_now"):
        assert key in snapshot
    assert snapshot["quality"]["scale_now"] > 0.0

    # LLM 判定"现在是仪器故障" -> 直接触发报修
    agent._apply_duty({"report_now": True, "fault_likely": 0.9})
    out = agent._maybe_report(100.0, {"now_utc": "2026-10-02T01:10:00Z"})
    assert out is not None and out["action"] == "report"
    # 只消费一次, 不会连环报修
    assert agent._maybe_report(101.0, {"now_utc": "2026-10-02T01:11:00Z"}) is None

    # 把握度不足 -> 忽略
    agent._apply_duty({"report_now": True, "fault_likely": 0.1})
    agent.last_report_hours = -1e9
    assert agent._maybe_report(102.0, {"now_utc": "2026-10-02T01:12:00Z"}) is None


def test_duty_simple_retry_recovers_report_times():
    """全量解析失败时, 降级重试("只问报修时刻")仍能把时刻拿回来."""
    agent = _duty_agent([
        None,                                      # 第一次(全量)调用失败
        {"report_utc": ["2026-10-02T01:30:00Z"]},  # 降级重试成功
    ])
    payload = {"now_utc": "2026-10-02T00:00:00Z"}
    agent.duty_pending = [{"issued_at_utc": "2026-10-02T00:00:00Z", "duty_log": "X" * 120}]
    agent._duty_tick(payload)          # 提交全量调用
    assert agent.client.requests[-1][0]["role"] == "system"
    agent._duty_tick(payload)          # 收到失败 -> 自动降级重试
    assert agent.duty_simple is True
    agent._duty_tick(payload)          # 收到降级答案
    assert [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in agent.duty_times] == ["2026-10-02T01:30:00Z"]


def test_llm_extract_json_picks_the_real_answer_not_a_draft():
    """v12 的真实故障: 推理文本里混着空 {} / 提示词模板, 必须挑出真正含报修时刻的对象."""
    from pro.llm_client import _extract_json

    reasoning = (
        "先看格式 {} ; 模板是 {\"report_utc\": [], \"no_report_utc\": []} ; "
        "草稿 {\"report_utc\": [\"2026-10-04T00:15:00Z\"]} 不对 ; "
        "最终 {\"report_utc\": [\"2026-10-02T01:30:00Z\", \"2026-10-05T02:30:00Z\"], \"notes\": \"ok\"}"
    )
    assert _extract_json(reasoning) == {
        "report_utc": ["2026-10-02T01:30:00Z", "2026-10-05T02:30:00Z"], "notes": "ok"}
    # 诚实的空答案仍然要能返回 (不能因为"没内容"就丢掉)
    assert _extract_json('{"report_utc": []}') == {"report_utc": []}
    # 其它环节的答案字段也要认得
    assert _extract_json('思考... {"bad_night": true, "avoid_directions": ["SW"]}') == {
        "bad_night": True, "avoid_directions": ["SW"]}
