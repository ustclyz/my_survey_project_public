#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""protocol.py - 协议适配层 (adapter).

职责 (单一职责原则, 官方协议一旦变动只改这里):

    1. 读取 stdin 上的 JSON Lines 消息 (initialize / decision_request / finish);
    2. 解析 ``initialize.payload`` 与 ``decision_request.payload`` 为本项目模型;
    3. 构造合法的 ``decision_response`` 并写 stdout.

字段来源与对照集中在 ``schemas/protocol.md``. 本文件只做"翻译", 不含任何决策
逻辑, 也不直接依赖 LLM. 输出严格裁剪到官方允许的字段集, 避免 ``agent_error``.

关键纪律 (来自任务卡"常见错误"与官方 protocol.py):
    * **标准输出只能写 JSON 回复**, 所有日志写标准错误 (stderr);
    * **必须回填 decision_sequence**;
    * **不得加未知字段**, 否则运行以 agent_error 结束.

协议版本: ``participant-agent-protocol-v4`` (jsonl-v4).
来源: 官方 ``资源/python.zip`` 的 ``agent_core/protocol.py`` 与
``docs/participant-guide.{zh,en}.md``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

from models import Action

PROTOCOL_VERSION = "participant-agent-protocol-v4"

# 官方 protocol.py 的 _RESPONSE_FIELDS: 每种动作允许的字段.
# 来源: 规则原文 / 官方示例 agent_core/protocol.py.
_RESPONSE_FIELDS: Dict[str, set] = {
    "observe": {"pointing", "assignments", "duration_seconds", "program"},
    "wait": {"duration_seconds", "until_utc"},
    "report": set(),
    "finish": set(),
}
_OPTIONAL_FIELDS = {"reason", "decision_source"}

PROGRAMS = ("DARK", "BRIGHT", "BACKUP")

# 协议输出流: 默认 None (运行时取当前 sys.stdout, 便于测试捕获). agent.py 启动时
# 调用 bind_protocol_stdout() 保存真实 stdout, 并把 sys.stdout 重定向到 stderr,
# 从而保证**只有**协议消息写到 stdout, 任何意外 print / 第三方库输出都落入 stderr.
_PROTOCOL_STREAM = None


def bind_protocol_stdout(stream) -> None:
    """绑定协议输出到指定的真实 stdout 流 (agent 启动时调用)."""
    global _PROTOCOL_STREAM
    _PROTOCOL_STREAM = stream



# ---------------------------------------------------------------------------
# 日志 (stderr)
# ---------------------------------------------------------------------------


def log(text: str) -> None:
    """写一行诊断日志到 stderr. 绝不抛异常, 绝不触碰 stdout."""
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def read_messages(stream: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """逐行解析 JSON 对象; 空行跳过, 非法 JSON 记录后跳过 (不崩溃).

    平台不应发送非法行, 但一个坏行不能使智能体在后续请求前就退出.
    """
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            log(f"protocol: 无法解析输入行 ({exc}); 跳过")
            continue
        if not isinstance(message, dict):
            log("protocol: 输入行不是 JSON 对象; 跳过")
            continue
        yield message


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    """解析 ISO-8601 UTC 时间串 (兼容结尾 Z)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# initialize 解析
# ---------------------------------------------------------------------------


class InitializeData:
    """``initialize.payload`` 的轻量解析结果.

    只保留规划/决策必需的字段; 原始 payload 一并保留供深入分析.
    """

    def __init__(self, message: Dict[str, Any]) -> None:
        self.raw = message
        payload = message.get("payload") or {}
        self.payload = payload
        site = payload.get("site") or {}
        survey = payload.get("survey") or {}
        instrument = payload.get("instrument") or {}
        limits = payload.get("limits") or {}
        scoring = payload.get("scoring") or {}

        self.lat_deg = float(site.get("latitude_deg", 0.0))
        self.lon_deg = float(site.get("longitude_deg", 0.0))
        self.min_altitude_deg = float(site.get("minimum_altitude_deg", 30.0))
        self.sun_altitude_limit_deg = float(site.get("sun_altitude_limit_deg", -18.0))

        self.start_utc = parse_utc(survey.get("start_utc"))
        self.end_utc = parse_utc(survey.get("end_utc"))
        self.slot_seconds = int(survey.get("slot_seconds", 900))
        self.nights: List[Dict[str, Any]] = list(survey.get("nights") or [])

        self.n_fibers = int(instrument.get("n_fibers", 16))
        self.grid_side = int(instrument.get("grid_side", round(self.n_fibers ** 0.5)))
        exposure = instrument.get("exposure") or {}
        self.min_exposure_seconds = int(exposure.get("min_duration_seconds", 60))
        self.max_exposure_seconds = int(exposure.get("max_duration_seconds", 3600))
        self.instrument = instrument

        self.scoring = scoring
        self.limits = limits
        self.global_wallclock_seconds = float(limits.get("global_wallclock_seconds", 900))
        self.response_max_bytes = int(limits.get("response_max_bytes", 524288))
        self.max_consecutive_reports = int(
            (scoring.get("reporting") or {}).get(
                "max_consecutive_reports", limits.get("max_consecutive_reports", 32)
            )
        )

        self.task_card = payload.get("task_card") or {}
        self.footprint = list(payload.get("footprint") or [])

        targets = payload.get("targets") or {}
        self.target_columns: List[str] = list(targets.get("columns") or [])
        self.target_rows: List[List[Any]] = list(targets.get("rows") or [])

    @property
    def card_id(self) -> str:
        return str(self.task_card.get("card_id", ""))


# ---------------------------------------------------------------------------
# decision_request 解析
# ---------------------------------------------------------------------------


class DecisionRequest:
    """``decision_request`` 的轻量解析结果 (仅协议字段, 不含决策语义)."""

    def __init__(self, message: Dict[str, Any]) -> None:
        self.raw = message
        self.decision_sequence = int(message.get("decision_sequence", 0))
        payload = message.get("payload") or {}
        self.payload = payload
        self.schema_version = str(payload.get("schema_version", ""))
        self.now_utc = parse_utc(payload.get("now_utc"))
        self.survey_end_utc = parse_utc(payload.get("survey_end_utc"))
        self.observe_action_index = payload.get("observe_action_index")
        self.running_total = payload.get("running_total")
        self.wallclock: Dict[str, Any] = payload.get("wallclock") or {}
        self.latest_bulletin = payload.get("latest_bulletin")
        self.latest_forecast = payload.get("latest_forecast")
        self.active_requests: List[Dict[str, Any]] = list(payload.get("active_requests") or [])
        self.new_messages: List[Dict[str, Any]] = list(payload.get("new_messages") or [])
        self.last_result = payload.get("last_result")

    # -- wallclock 便捷访问 ------------------------------------------------
    def remaining_seconds(self) -> float:
        return _as_float(self.wallclock.get("remaining_seconds"), float("inf"))

    def remaining_real_cpu_seconds(self) -> float:
        value = self.wallclock.get("remaining_real_cpu_seconds")
        return _as_float(value, self.remaining_seconds())

    def wall_remaining_seconds(self) -> float:
        value = self.wallclock.get("wall_remaining_seconds")
        return _as_float(value, self.remaining_seconds())


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _log_response(action: Action, envelope: Dict[str, Any]) -> None:
    """把完整回复写 stderr, 便于主办方复现核验 (可审计)."""
    log(f"protocol: decision_response seq={envelope.get('decision_sequence')} -> {json.dumps(envelope, ensure_ascii=False, separators=(',', ':'))}")


def send_response(decision_sequence: int, action: Action) -> None:
    """把一个 :class:`Action` 编成 ``decision_response`` 写 stdout.

    严格裁剪字段: 仅保留该动作允许的字段 + 可选的 reason/decision_source.
    编码失败时退化为最安全的 ``wait``, 避免卡死或污染输出.
    """
    try:
        action_fields = action.to_protocol_fields()
    except Exception as exc:  # noqa: BLE001 - 绝不让构造失败终止运行
        log(f"protocol: 动作无法序列化 ({exc}); 发送安全 wait")
        action_fields = {"action": "wait", "duration_seconds": 900, "reason": "encode-error-fallback",
                         "decision_source": "fallback"}

    kind = action_fields.get("action")
    allowed = _RESPONSE_FIELDS.get(kind, set()) | _OPTIONAL_FIELDS | {"action"}
    envelope: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "decision_response",
        "decision_sequence": decision_sequence,
    }
    for key, value in action_fields.items():
        if key in allowed and value is not None:
            envelope[key] = value

    # 响应体大小保护: 超过 response_max_bytes 时退化为 wait.
    # 使用 ensure_ascii=True: 输出纯 ASCII JSON (Unicode 转义), 规避任何终端/管道
    # 编码问题, 且始终是合法 JSON; 平台按 JSON 解析, 转义不影响语义.
    encoded = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 512_000:
        log("protocol: 响应过大; 退化为 wait")
        envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "decision_response",
            "decision_sequence": decision_sequence,
            "action": "wait",
            "duration_seconds": 900,
            "reason": "oversize-fallback",
        }
        encoded = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))

    try:
        # 显式以 UTF-8 字节写**绑定的协议流**(真实 stdout), 不依赖 locale 编码
        # (Windows GBK 下也安全). 不直接写 sys.stdout —— 后者可能已被重定向到
        # stderr 以避免非协议输出污染 stdout.
        data = (encoded + "\n").encode("utf-8")
        stream = _PROTOCOL_STREAM if _PROTOCOL_STREAM is not None else sys.stdout
        buf = getattr(stream, "buffer", None)
        if buf is not None:
            buf.write(data)
            buf.flush()
        else:
            # 流没有 buffer (如测试捕获的 StringIO) 时回退到 write
            try:
                stream.write(encoded + "\n")
                stream.flush()
            except Exception:
                print(encoded, flush=True)
    except Exception as exc:  # noqa: BLE001
        log(f"protocol: 写 stdout 失败 ({exc})")
    _log_response(action, envelope)


def build_response_object(decision_sequence: int, action: Action) -> Dict[str, Any]:
    """构造 (但不写出) decision_response, 供单元测试复用.

    与 :func:`send_response` 的裁剪逻辑保持一致.
    """
    action_fields = action.to_protocol_fields()
    kind = action_fields.get("action")
    allowed = _RESPONSE_FIELDS.get(kind, set()) | _OPTIONAL_FIELDS | {"action"}
    envelope: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "decision_response",
        "decision_sequence": decision_sequence,
    }
    for key, value in action_fields.items():
        if key in allowed and value is not None:
            envelope[key] = value
    return envelope


# ---------------------------------------------------------------------------
# 向后兼容别名 (任务书要求 skill/接口名友好)
# ---------------------------------------------------------------------------


def parse_initialize(message: Dict[str, Any]) -> InitializeData:
    return InitializeData(message)


def parse_decision_request(message: Dict[str, Any]) -> DecisionRequest:
    return DecisionRequest(message)
