#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent.py - 智能体入口.

仓库里有两条决策内核:

* ``pro``  —— 官方 python-pro 参考实现的本地移植 (纯标准库; 平台榜单里
              "official examples (pro)" 的 baseline 约 29,710 分)。**线上运行内核**,
              协议循环在 :mod:`pro.agent`。
* ``planner`` / ``preplan`` / ``llm`` / ``protocol`` / ``models`` —— 我们自己的静态
              内核与协议层, 保留用于本地实验与单元测试, 线上不再走它。

入口本身只做一件与安全有关的事: **stdout 硬化** —— 把 ``sys.stdout`` 重定向到
``sys.stderr``, 并把真实 stdout 绑定为协议输出流。这样任何第三方库或调试输出都
不会污染平台解析的协议通道; 所有协议回复由 :func:`pro.agent._emit` 显式写到真实
stdout。
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 9):  # pragma: no cover - 平台固定 python:3.12-slim
    sys.stderr.write("agent: 需要 Python 3.9 或更高版本\n")
    raise SystemExit(3)


def _harden_stdout():
    """保存真实 stdout 作为协议流, 再把 ``sys.stdout`` 指向 stderr."""
    real_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
    except Exception:  # noqa: BLE001 - 极端环境下重定向失败也不影响协议流
        pass
    return real_stdout


_REAL_STDOUT = _harden_stdout()

from pro import agent as _pro_agent  # noqa: E402 - 必须在 stdout 硬化之后导入

_pro_agent.bind_stdout(_REAL_STDOUT)


def main() -> int:
    return _pro_agent.main()


if __name__ == "__main__":
    raise SystemExit(main())
