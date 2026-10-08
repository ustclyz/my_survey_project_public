#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pack_min.py - 只打包"运行真正需要"的文件的 ZIP (agent.py + pro/ + 清单).

用途: GitHub 推送不可用时, 走 `survey26 project upload <zip>` 通道提交同一份代码。
ZIP 条目一律用正斜杠 (Windows 反斜杠会被平台判为非法路径)。

用法: py tools/pack_min.py <repo_root> <out.zip>
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    files = [root / "agent.py", root / "observer.project.json"]
    files += sorted((root / "pro").glob("*.py"))
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.relative_to(root).as_posix())
    print(f"packed {len(files)} files -> {out} ({out.stat().st_size} bytes)")
    for path in files:
        print("   ", path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
