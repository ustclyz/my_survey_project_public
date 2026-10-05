#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pack_agent.py - 把本项目打包成平台可提交的 ZIP.

    python pack_agent.py [--out ../my-agent.zip]

遵循平台项目规则 (参考官方示例):
    * ``observer.project.json`` 保持在 ZIP 根, 声明 ``"protocol": "jsonl-v4"``;
    * ``.env`` 绝不打包 (平台拒绝含 .env 的 ZIP, 密钥应在参赛页设置);
    * 跳过 ``__pycache__`` / ``.git`` / ``.pytest_cache`` / 编辑器杂物;
    * 默认输出到项目**外部**, 避免把 ZIP 打进自身.

仅使用标准库.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "observer.project.json"
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv", "run_output", ".pytest_cache",
                 ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
                 "tools", "sim_data"}   # 本地开发工具与合成数据: 不进提交包
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".tar", ".gz", ".7z"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


def collect(root: Path):
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if not path.is_file() or path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name not in ENV_TEMPLATES):
            continue
        files.append(path)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "my-agent.zip", help="输出 ZIP 路径 (默认项目外)")
    args = parser.parse_args(argv)

    manifest_path = ROOT / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"缺少 {MANIFEST_NAME} (应位于 {ROOT})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "jsonl-v4":
        raise SystemExit(f'{MANIFEST_NAME} 必须声明 "protocol": "jsonl-v4"')

    out = args.out.resolve()
    if ROOT in out.parents or out.parent == ROOT:
        raise SystemExit("请把 ZIP 输出到项目文件夹之外, 否则会打包自身")

    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect(ROOT)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())

    size = out.stat().st_size
    print(f"打包 {len(files)} 个文件: {ROOT} -> {out} ({size} bytes = {size / 1024 / 1024:.2f} MB)")
    print(f"  清单: image={manifest.get('image')} run={' '.join(manifest.get('run', []))}")
    if (ROOT / ".env").is_file():
        print("  注意: 检测到 .env 但已排除 (绝不上传; 请在参赛页设置密钥)")
    if size >= MAX_SIZE_BYTES:
        print(f"  [错误] ZIP 超过 50MB 上限 ({size / 1024 / 1024:.2f} MB)")
        return 4
    print("  下一步: 作为完整项目上传该 ZIP, 或把该目录推到 GitHub 仓库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
