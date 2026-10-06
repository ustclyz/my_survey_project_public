#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config.py 单测: 服务商前缀发现 (平台"添加模型服务"支持自定义前缀).

回归背景: 队伍在平台上把新 key 存成了 ``SOAD_*`` 前缀, 而 config.py 只认
OPENAI/KIMI/DEEPSEEK/MOONSHOT, 结果整场静默退化为无 LLM (放弃评奖硬门槛)。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

_ALL = ("OPENAI", "KIMI", "DEEPSEEK", "MOONSHOT", "SOAD", "USTC", "ANTHROPIC")


def _clear(monkeypatch) -> None:
    for prefix in _ALL:
        for suffix in ("_API_KEY", "_BASE_URL", "_MODEL"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    monkeypatch.delenv("OBSERVER_MODEL_DISABLED", raising=False)


def test_custom_prefix_is_discovered(monkeypatch):
    """自定义前缀 (平台自定义模型服务) 必须被识别, 不能静默忽略。"""
    _clear(monkeypatch)
    monkeypatch.setenv("SOAD_API_KEY", "sk-test-abcd1234")
    monkeypatch.setenv("SOAD_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("SOAD_MODEL", "deepseek-flash")

    cfg = config.load_config()
    assert cfg.llm_enabled, "自定义前缀有 key 时必须启用 LLM"
    assert cfg.llm.provider == "soad"
    assert cfg.llm.base_url == "https://api.deepseek.com"
    assert cfg.llm.model == "deepseek-flash"
    assert cfg.llm.api_key == "sk-test-abcd1234"


def test_known_prefix_wins_over_custom(monkeypatch):
    """已知前缀优先级高于自定义前缀。"""
    _clear(monkeypatch)
    monkeypatch.setenv("SOAD_API_KEY", "sk-soad")
    monkeypatch.setenv("SOAD_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    cfg = config.load_config()
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o-mini"


def test_anthropic_prefix_is_not_treated_as_openai_compatible(monkeypatch):
    """Anthropic 的接口不是 OpenAI 兼容, 不应被自动发现后误用。"""
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert "ANTHROPIC" not in config.provider_candidates()
    cfg = config.load_config()
    assert not cfg.llm_enabled, "只有 Anthropic key 时应退化为静态模式"


def test_no_key_falls_back_to_static(monkeypatch):
    """完全没有密钥时不报错, 退化为静态模式。"""
    _clear(monkeypatch)
    cfg = config.load_config()
    assert not cfg.llm_enabled
    assert cfg.static_only
