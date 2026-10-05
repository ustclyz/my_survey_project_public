#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config.py - 运行配置.

从环境变量读取 LLM 密钥与模型配置, **绝不把密钥写入源码**. 支持多组服务商
前缀, 按优先级选择第一组可用的:

    通用 (优先):
        OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    可选服务商 (次选, 任一可用即可):
        KIMI_API_KEY / KIMI_BASE_URL / KIMI_MODEL
        DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
        MOONSHOT_API_KEY / MOONSHOT_BASE_URL / MOONSHOT_MODEL

未配置任何密钥时, 智能体仍可运行 (静态规划 + 启发式决策), 并在 stderr 明确
提示 "未启用 LLM, 仅静态模式".

平台评测容器会设置 HTTPS_PROXY, 常见 SDK 无需额外配置.

参考 (官方示例 llm_client.py):
    * base_url 需以 /v1 之类结尾, 客户端拼接 ``{base_url}/chat/completions``;
    * 评测可设 ``OBSERVER_MODEL_DISABLED=1`` 表示"无模型"评测, 此时无需密钥.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from models import LLMConfig

# 服务商前缀, 按优先级排列. 第一个提供了 API key 的胜出.
_PROVIDER_PREFIXES: Tuple[str, ...] = ("OPENAI", "KIMI", "DEEPSEEK", "MOONSHOT")

# 各组前缀的默认 base_url / model (当对应环境变量缺失时使用).
_DEFAULTS = {
    "OPENAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "KIMI": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "DEEPSEEK": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "MOONSHOT": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def model_disabled() -> bool:
    """平台是否显式要求禁用模型 (无模型评测).

    对应官方示例的 ``OBSERVER_MODEL_DISABLED=1``.
    """
    return os.environ.get("OBSERVER_MODEL_DISABLED", "").strip() == "1"


@dataclass
class RuntimeConfig:
    """智能体运行配置 (不含任何明文密钥, 密钥只经环境变量传递)."""

    llm: LLMConfig
    static_only: bool                     # True = 无密钥或平台禁用, 走纯静态
    openai_package_available: bool = False
    config_source: str = ""               # 胜出的服务商前缀, 便于审计

    @property
    def llm_enabled(self) -> bool:
        return self.llm.usable and not self.static_only

    def describe(self) -> str:
        """返回一段可安全写日志的描述 (不含密钥本身)."""
        if not self.llm_enabled:
            reason = "OBSERVER_MODEL_DISABLED=1" if model_disabled() else "未检测到 API 密钥"
            return f"未启用 LLM, 仅静态模式 ({reason})"
        return (
            f"LLM 已启用: provider={self.llm.provider} model={self.llm.model} "
            f"base_url={self.llm.base_url} (key=***{self.llm.api_key[-4:] if len(self.llm.api_key) >= 4 else '****'})"
        )


def _openai_available() -> bool:
    """检测是否安装了 openai 官方 SDK (缺失时 llm.py 自动用 urllib 后备)."""
    try:
        import openai  # noqa: F401

        return True
    except Exception:
        return False


def load_config() -> RuntimeConfig:
    """从环境变量加载运行配置.

    Returns:
        :class:`RuntimeConfig`. 无密钥时 ``static_only=True``, 程序照常运行.
    """
    chosen_prefix = ""
    api_key = ""
    for prefix in _PROVIDER_PREFIXES:
        key = _env(f"{prefix}_API_KEY")
        if key:
            chosen_prefix, api_key = prefix, key
            break

    # 允许 KIMI_API_KEY 作为 OPENAI_API_KEY 的后备 (官方示例行为)
    if not api_key:
        api_key = _env("KIMI_API_KEY")

    default_base, default_model = _DEFAULTS.get(chosen_prefix, _DEFAULTS["OPENAI"])
    # base_url / model: 优先用与胜出前缀匹配的变量, 其次用通用 OPENAI_*
    base_url = _env(f"{chosen_prefix}_BASE_URL") or _env("OPENAI_BASE_URL", default_base) if chosen_prefix else _env("OPENAI_BASE_URL", default_base)
    model = _env(f"{chosen_prefix}_MODEL") or _env("OPENAI_MODEL", default_model) if chosen_prefix else _env("OPENAI_MODEL", default_model)

    timeout = _safe_float(_env("LLM_TIMEOUT_SECONDS"), 30.0)

    llm = LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/") or default_base,
        model=model or default_model,
        timeout_seconds=max(5.0, min(timeout, 120.0)),
        provider=(chosen_prefix or "openai").lower(),
        enabled=True,
    )

    disabled = model_disabled()
    static_only = disabled or not llm.usable

    return RuntimeConfig(
        llm=llm,
        static_only=static_only,
        openai_package_available=_openai_available(),
        config_source=chosen_prefix or ("OPENAI" if _env("OPENAI_API_KEY") else ""),
    )


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":  # pragma: no cover - 便于人工排查配置
    import sys

    cfg = load_config()
    print(cfg.describe(), file=sys.stderr)
    print(f"openai 包可用: {cfg.openai_package_available}", file=sys.stderr)
