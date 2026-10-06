#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""models.py - 智能体的数据模型层.

本模块集中定义运行期使用的数据结构, 与协议层的字段名解耦:

    * :class:`LLMConfig`   - LLM 客户端配置 (来源: 环境变量, 见 config.py)
    * :class:`Target`      - 单个观测目标 (复用 preplan.Target 的字段约定)
    * :class:`Action`      - 一次决策动作 (observe/wait/report/finish)
    * :class:`NightPlan`   - 一夜的任务规划结果 (LLM 环节A 的输出)
    * :class:`DecisionState` - 一次 decision_request 压缩后的决策状态

设计要点:
    * 仅依赖 Python 标准库 (dataclasses / typing / datetime), 便于在 2 核 2GB
      评测容器中运行;
    * ``Target`` 直接从 ``preplan`` 的冻结 dataclass 转换而来 (见 from_preplan),
      从而**复用现有规划逻辑**, 不重写.

协议来源: 官方示例 ``agent_core/state.py`` 与 ``docs/participant-guide`` 的
``initialize`` / ``decision_request`` 结构. 详细字段对照见 ``schemas/protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM 客户端配置.

    所有字段均从环境变量读取, **绝不硬编码密钥** (见 config.py).
    """

    api_key: str = ""                                   # 来源: OPENAI_API_KEY / KIMI_API_KEY
    base_url: str = "https://api.openai.com/v1"         # 来源: OPENAI_BASE_URL
    model: str = "gpt-4o-mini"                          # 来源: OPENAI_MODEL
    timeout_seconds: float = 30.0
    max_tokens: int = 1500
    temperature: float = 0.2
    provider: str = "openai"                            # 用于日志标注服务商
    enabled: bool = True                                # False 时强制静态回退

    @property
    def usable(self) -> bool:
        """是否具备真正调用 LLM 的条件 (有密钥且未被禁用)."""
        return bool(self.api_key.strip()) and self.enabled


# ---------------------------------------------------------------------------
# 观测目标 (复用 preplan 的字段约定)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """单个观测目标.

    字段与 ``preplan.Target`` 完全一致, 便于相互转换; 用 ``from_preplan`` 从
    ``preplan.Target`` 构造 (避免重复解析 CSV).
    """

    target_id: str
    ra_deg: float
    dec_deg: float
    target_class: str = ""
    feature_flux: float = 0.0
    science_weight: float = 0.0
    required: bool = False

    @classmethod
    def from_preplan(cls, t: Any) -> "Target":
        """从 ``preplan.Target`` (或其他具有相同属性的对象) 构造."""
        return cls(
            target_id=str(t.target_id),
            ra_deg=float(t.ra_deg),
            dec_deg=float(t.dec_deg),
            target_class=str(getattr(t, "target_class", "") or ""),
            feature_flux=float(getattr(t, "feature_flux", 0.0) or 0.0),
            science_weight=float(getattr(t, "science_weight", 0.0) or 0.0),
            required=bool(getattr(t, "required", False)),
        )

    @classmethod
    def from_protocol_row(cls, columns: List[str], row: List[Any]) -> "Target":
        """从 ``initialize.payload.targets`` 的 ``columns`` + ``rows`` 构造.

        兼容官方列名: target_id, ra_deg, dec_deg, target_class, feature_flux,
        science_weight, required. 缺失列使用安全默认值.
        """
        col = {name: idx for idx, name in enumerate(columns)}

        def get(name: str, default: Any) -> Any:
            idx = col.get(name)
            if idx is None or idx >= len(row):
                return default
            value = row[idx]
            return default if value is None else value

        return cls(
            target_id=str(get("target_id", "")),
            ra_deg=float(get("ra_deg", 0.0)),
            dec_deg=float(get("dec_deg", 0.0)),
            target_class=str(get("target_class", "") or ""),
            feature_flux=float(get("feature_flux", 0.0) or 0.0),
            science_weight=float(get("science_weight", 0.0) or 0.0),
            required=bool(get("required", False)),
        )


# ---------------------------------------------------------------------------
# 动作 (protocol 层构造, 决策内核产出)
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """一次决策动作.

    统一表示四种动作, 未使用的字段为 ``None`` / 空. 由 :mod:`protocol` 负责
    序列化为面向平台的 ``decision_response`` (裁剪到合法字段集).

    ``Action`` 与平台协议的映射见 ``schemas/protocol.md``.
    """

    type: str  # "observe" | "wait" | "report" | "finish"
    ra_deg: Optional[float] = None            # observe: 视场中心赤经 (由 alt/az 反算)
    dec_deg: Optional[float] = None           # observe: 视场中心赤纬
    pointing: Optional[Dict[str, float]] = None   # observe: {"alt_deg","az_deg"}
    assignments: Dict[str, str] = field(default_factory=dict)  # observe: {纤维:"目标ID"}
    exposure_seconds: Optional[int] = None    # observe: 曝光秒数 60-3600
    program: Optional[str] = None             # observe: DARK/BRIGHT/BACKUP
    duration_seconds: Optional[int] = None    # wait: 等待秒数
    until_utc: Optional[str] = None           # wait: 等到某时刻 (以 Z 结尾)
    reason: str = ""
    decision_source: str = "static"           # "llm" | "static" | "fallback"

    def to_protocol_fields(self) -> Dict[str, Any]:
        """转换为协议动作字段 (不含外层 envelope, 由 protocol.py 补齐).

        返回的键严格限定在官方 ``_RESPONSE_FIELDS`` 允许集合内.
        """
        if self.type == "observe":
            fields: Dict[str, Any] = {
                "action": "observe",
                "pointing": self.pointing or {},
                "assignments": dict(self.assignments),
                "duration_seconds": self.exposure_seconds,
                "program": self.program or "BACKUP",
            }
        elif self.type == "wait":
            fields = {"action": "wait"}
            if self.until_utc:
                fields["until_utc"] = self.until_utc
            elif self.duration_seconds is not None:
                fields["duration_seconds"] = int(self.duration_seconds)
        elif self.type == "report":
            fields = {"action": "report"}
        elif self.type == "finish":
            fields = {"action": "finish"}
        else:
            raise ValueError(f"未知动作类型: {self.type!r}")
        if self.reason:
            fields["reason"] = self.reason
        if self.decision_source:
            fields["decision_source"] = self.decision_source
        return fields


# ---------------------------------------------------------------------------
# 一夜规划 (LLM 环节A 输出)
# ---------------------------------------------------------------------------


@dataclass
class NightPlan:
    """本夜观测规划.

    由 :func:`llm.plan_night` (LLM 环节A·任务规划) 产出, 供
    :func:`llm.decide_action` (环节B·行动决策) 与静态规划器参考.

    协议不要求下发 NightPlan, 它纯属智能体内部状态.
    """

    targets: List[str] = field(default_factory=list)  # 本夜候选目标 ID
    program: str = "BACKUP"                            # DARK | BRIGHT | BACKUP
    strategy: str = ""                                 # 一句话策略 (供审计)
    avoid_directions: List[str] = field(default_factory=list)   # N..NW, 本夜规避方向
    duration_scale: float = 1.0                        # 曝光时长缩放 0.7-1.4
    source: str = "static"                             # "llm" | "static"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "targets": list(self.targets),
            "program": self.program,
            "strategy": self.strategy,
            "avoid_directions": list(self.avoid_directions),
            "duration_scale": self.duration_scale,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 决策状态 (一次 decision_request 的压缩视图)
# ---------------------------------------------------------------------------


@dataclass
class DecisionState:
    """一次 ``decision_request`` 压缩后的决策状态.

    只保留决策内核与 LLM 提示词所需的字段, **不塞入全部数万目标**; 目标清单
    以 ``target_count`` / ``required_remaining`` 等统计量 + ``candidate_targets``
    少量候选的形式提供.
    """

    decision_sequence: int = 0
    now_utc: Optional[datetime] = None
    survey_end_utc: Optional[datetime] = None
    remaining_seconds: float = float("inf")            # wallclock.remaining_seconds
    remaining_real_cpu_seconds: float = float("inf")   # remaining_real_cpu_seconds
    wall_remaining_seconds: float = float("inf")       # wall_remaining_seconds
    night_index: Optional[int] = None                  # 当前夜序号, None 表示白天
    night_start_utc: Optional[datetime] = None
    night_end_utc: Optional[datetime] = None
    is_new_night: bool = False
    site_closed: bool = False                          # 简报 rain/storm ALL
    notices: List[Dict[str, Any]] = field(default_factory=list)
    observation_requests: List[Dict[str, Any]] = field(default_factory=list)
    abnormal: bool = False                             # 上次观测结果明显异常
    abnormal_reason: str = ""

    # -- 目标统计 (避免把全部目标塞进提示词) ------------------------------
    target_count: int = 0
    required_count: int = 0
    required_remaining: int = 0
    candidate_targets: List[str] = field(default_factory=list)
    last_hit_rate: Optional[float] = None

    def compact_summary(self, max_targets: int = 40) -> Dict[str, Any]:
        """生成用于 LLM 提示词的紧凑摘要 (统计量 + 有限候选)."""
        return {
            "decision_sequence": self.decision_sequence,
            "now_utc": self.now_utc.isoformat() if self.now_utc else None,
            "night_index": self.night_index,
            "is_new_night": self.is_new_night,
            "remaining_seconds": round(self.remaining_seconds, 1)
            if self.remaining_seconds != float("inf") else None,
            "wall_remaining_seconds": round(self.wall_remaining_seconds, 1)
            if self.wall_remaining_seconds != float("inf") else None,
            "site_closed": self.site_closed,
            "notices": self.notices[:8],
            "active_requests": [
                {
                    "request_id": r.get("request_id"),
                    "minimum_completed": r.get("minimum_completed"),
                    "remaining_count": r.get("remaining_count"),
                    "completion_reward": r.get("completion_reward"),
                    "deadline_utc": r.get("deadline_utc"),
                    "target_ids": (r.get("target_ids") or [])[:10],
                }
                for r in self.observation_requests[:4]
            ],
            "abnormal": self.abnormal,
            "abnormal_reason": self.abnormal_reason,
            "target_count": self.target_count,
            "required_count": self.required_count,
            "required_remaining": self.required_remaining,
            "last_hit_rate": self.last_hit_rate,
            "candidate_targets": self.candidate_targets[:max_targets],
        }
