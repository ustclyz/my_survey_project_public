"""pro - 官方 python-pro 参考实现的本地移植包.

来源: GOSIM survey26 官方示例项目 ``examples/python-pro`` (标准库实现, 平台
榜单 baseline pro 约 29,710 分)。本包**原样保留**其规划/学习算法, 仅做三处最小
适配, 以便在本仓库内运行:

1. 把模块级 ``from skymath import ...`` 改为包内绝对导入 ``from pro.skymath ...``;
2. ``llm_client`` 的密钥发现改为与 ``config.provider_candidates()`` 同源的多前缀
   逻辑 (平台生效前缀是 ``SOAD_*``, 官方只认 ``OPENAI_*``/``KIMI_*``);
3. 入口 ``agent.py`` 复用本仓库已有的 stdout 硬化 (协议流与日志流分离), 并在
   无密钥时退化为"仅规则"而不是直接退出。

算法细节见各模块 docstring 与 README。
"""
