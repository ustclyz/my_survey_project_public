#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""card_static_check.py - GOSIM 巡天智能体任务卡静态校验脚本.

该脚本对 ``cards/`` 目录下的全部任务卡进行纯静态检查, 不启动模拟器、
不调用任何 LLM, 也不修改智能体主程序. 功能包括:

1. 遍历 ``cards/`` 下全部卡片目录;
2. 加载 ``config/`` 下的 JSON 配置, 捕获文件缺失、JSON 解析异常与关键字段缺失;
3. 统计每张卡的核心信息:
   - 总目标数
   - 必观测目标数 (``required == true``)
   - 天区面积 (由 footprint 多边形球面面积估算)
   - 光纤数量 (``field.n_fibers``)
   - 计分基准参数 (``flux_zero_point``、``exposure_zero_point_seconds``)
4. 将完整报告写入项目根目录 ``static_check_report.txt`` (UTF-8);
5. 在控制台打印每张卡的校验状态与核心统计值.

运行方式::

    py card_static_check.py

依赖: 仅使用 Python 标准库.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
CARDS_DIR: Path = PROJECT_ROOT / "cards"
REPORT_PATH: Path = PROJECT_ROOT / "static_check_report.txt"

# 每张卡必须包含的相对文件路径
REQUIRED_FILES: Tuple[str, ...] = (
    "config/v4_fiber_config.json",
    "config/v4_score_config.json",
    "public/targets.csv",
    "public/footprint.csv",
)

# 各配置文件中必须存在且非空的关键字段 (点号表示嵌套路径)
FIBER_REQUIRED_KEYS: Tuple[str, ...] = (
    "field.n_fibers",
    "field.fiber_area_deg2",
    "exposure.min_duration_seconds",
    "exposure.max_duration_seconds",
)
SCORE_REQUIRED_KEYS: Tuple[str, ...] = (
    "flux_zero_point",
    "exposure_zero_point_seconds",
    "airmass_exponent",
)


# ---------------------------------------------------------------------------
# JSON 工具
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """安全加载 JSON 文件.

    Args:
        path: JSON 文件路径.

    Returns:
        二元组 ``(data, error)``. 成功时 ``error`` 为 ``None``;
        失败时 ``data`` 为 ``None`` 并给出人类可读的错误信息.
    """
    if not path.is_file():
        return None, f"文件缺失: {path.name}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh), None
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析异常: {exc}"
    except OSError as exc:  # 权限 / IO 错误
        return None, f"文件读取失败: {exc}"


def get_nested(data: Dict[str, Any], dotted_key: str) -> Any:
    """按点号路径读取嵌套字典的值.

    Args:
        data: 目标字典.
        dotted_key: 形如 ``"field.n_fibers"`` 的路径.

    Returns:
        命中的值, 若任一层缺失则返回 ``None``.
    """
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def missing_keys(data: Dict[str, Any], keys: Tuple[str, ...]) -> List[str]:
    """返回 ``keys`` 中在 ``data`` 内缺失或为空的值对应路径列表."""
    missing: List[str] = []
    for key in keys:
        if get_nested(data, key) is None:
            missing.append(key)
    return missing


# ---------------------------------------------------------------------------
# CSV / 几何工具
# ---------------------------------------------------------------------------


def count_rows(path: Path, has_header: bool = True) -> Tuple[Optional[int], Optional[str]]:
    """统计 CSV 数据行数 (不含表头).

    Args:
        path: CSV 文件路径.
        has_header: 是否包含表头行.

    Returns:
        二元组 ``(row_count, error)``.
    """
    if not path.is_file():
        return None, f"文件缺失: {path.name}"
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
    except OSError as exc:
        return None, f"文件读取失败: {exc}"
    data_rows = max(len(rows) - (1 if has_header and rows else 0), 0)
    return data_rows, None


def count_required_targets(path: Path) -> Tuple[Optional[int], Optional[str]]:
    """统计 ``required`` 列为真的目标数量.

    Args:
        path: ``targets.csv`` 路径.

    Returns:
        二元组 ``(required_count, error)``.
    """
    if not path.is_file():
        return None, f"文件缺失: {path.name}"
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "required" not in reader.fieldnames:
                return None, "targets.csv 缺少 'required' 列"
            for row in reader:
                value = (row.get("required") or "").strip().lower()
                if value in ("true", "1", "yes", "y", "t"):
                    count += 1
    except OSError as exc:
        return None, f"文件读取失败: {exc}"
    return count, None


def spherical_polygon_area_deg2(vertices: List[Tuple[float, float]]) -> float:
    """估算球面多边形的面积 (平方度).

    使用球面 excess 公式: 将顶点按参考点做等距方位投影到切平面,
    再套用平面鞋带公式近似. 对于巡天天区的小尺度分量, 该近似误差极小.

    Args:
        vertices: ``[(ra_deg, dec_deg), ...]`` 顶点序列 (闭合前).

    Returns:
        面积 (平方度, 非负).
    """
    if len(vertices) < 3:
        return 0.0

    # 参考点取顶点均值
    ref_ra = sum(v[0] for v in vertices) / len(vertices)
    ref_dec = sum(v[1] for v in vertices) / len(vertices)
    ref_ra_rad = math.radians(ref_ra)
    ref_dec_rad = math.radians(ref_dec)

    projected: List[Tuple[float, float]] = []
    for ra, dec in vertices:
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)
        d_ra = ra_rad - ref_ra_rad
        # 处理经度环绕
        if d_ra > math.pi:
            d_ra -= 2.0 * math.pi
        elif d_ra < -math.pi:
            d_ra += 2.0 * math.pi
        # 等距方位投影
        cos_c = (
            math.sin(ref_dec_rad) * math.sin(dec_rad)
            + math.cos(ref_dec_rad) * math.cos(dec_rad) * math.cos(d_ra)
        )
        cos_c = max(-1.0, min(1.0, cos_c))
        c = math.acos(cos_c)
        if c < 1e-12:
            projected.append((0.0, 0.0))
            continue
        k = c / math.sin(c)
        x = k * math.cos(dec_rad) * math.sin(d_ra)
        y = k * (
            math.cos(ref_dec_rad) * math.sin(dec_rad)
            - math.sin(ref_dec_rad) * math.cos(dec_rad) * math.cos(d_ra)
        )
        projected.append((x, y))

    # 平面鞋带公式 (弧度平面, 结果单位为 sr, 再转 deg^2)
    area = 0.0
    n = len(projected)
    for i in range(n):
        x1, y1 = projected[i]
        x2, y2 = projected[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    return area * (180.0 / math.pi) ** 2


def compute_footprint_area(path: Path) -> Tuple[Optional[float], Optional[str]]:
    """计算 ``footprint.csv`` 覆盖的天区面积 (平方度).

    文件可能包含多个 component (``component_id``), 各分量面积求和.

    Args:
        path: ``footprint.csv`` 路径.

    Returns:
        二元组 ``(area_deg2, error)``.
    """
    if not path.is_file():
        return None, f"文件缺失: {path.name}"

    components: Dict[str, List[Tuple[float, float]]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            expected = {"component_id", "ra_deg", "dec_deg"}
            if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
                return None, "footprint.csv 缺少必要列 (component_id/ra_deg/dec_deg)"
            for row in reader:
                cid = row["component_id"]
                try:
                    ra = float(row["ra_deg"])
                    dec = float(row["dec_deg"])
                except (TypeError, ValueError):
                    return None, "footprint.csv 存在无法解析的坐标值"
                components.setdefault(cid, []).append((ra, dec))
    except OSError as exc:
        return None, f"文件读取失败: {exc}"

    if not components:
        return None, "footprint.csv 无有效顶点"

    total = sum(spherical_polygon_area_deg2(v) for v in components.values())
    return total, None


# ---------------------------------------------------------------------------
# 单卡校验
# ---------------------------------------------------------------------------


def check_card(card_dir: Path) -> Dict[str, Any]:
    """对单张任务卡执行完整静态校验.

    Args:
        card_dir: 卡片目录 (例如 ``cards/alpha``).

    Returns:
        包含校验状态与统计信息的字典.
    """
    result: Dict[str, Any] = {
        "name": card_dir.name,
        "path": str(card_dir),
        "errors": [],
        "warnings": [],
        "n_targets": None,
        "n_required": None,
        "footprint_area_deg2": None,
        "n_fibers": None,
        "flux_zero_point": None,
        "exposure_zero_point_seconds": None,
        "n_components": None,
    }

    # 1) 必需文件存在性
    for rel in REQUIRED_FILES:
        if not (card_dir / rel).is_file():
            result["errors"].append(f"必需文件缺失: {rel}")

    # 2) fiber 配置
    fiber_path = card_dir / "config" / "v4_fiber_config.json"
    fiber_data, fiber_err = load_json(fiber_path)
    if fiber_err:
        result["errors"].append(f"v4_fiber_config.json -> {fiber_err}")
    else:
        missing = missing_keys(fiber_data, FIBER_REQUIRED_KEYS)
        if missing:
            result["errors"].append(
                "v4_fiber_config.json 关键字段缺失: " + ", ".join(missing)
            )
        result["n_fibers"] = get_nested(fiber_data, "field.n_fibers")

    # 3) score 配置
    score_path = card_dir / "config" / "v4_score_config.json"
    score_data, score_err = load_json(score_path)
    if score_err:
        result["errors"].append(f"v4_score_config.json -> {score_err}")
    else:
        missing = missing_keys(score_data, SCORE_REQUIRED_KEYS)
        if missing:
            result["errors"].append(
                "v4_score_config.json 关键字段缺失: " + ", ".join(missing)
            )
        result["flux_zero_point"] = get_nested(score_data, "flux_zero_point")
        result["exposure_zero_point_seconds"] = get_nested(
            score_data, "exposure_zero_point_seconds"
        )

    # 4) targets.csv
    targets_path = card_dir / "public" / "targets.csv"
    n_targets, tgt_err = count_rows(targets_path)
    if tgt_err:
        result["errors"].append(f"targets.csv -> {tgt_err}")
    else:
        result["n_targets"] = n_targets
    n_req, req_err = count_required_targets(targets_path)
    if req_err:
        result["warnings"].append(f"targets.csv required 统计 -> {req_err}")
    else:
        result["n_required"] = n_req

    # 5) footprint.csv 面积
    footprint_path = card_dir / "public" / "footprint.csv"
    area, fp_err = compute_footprint_area(footprint_path)
    if fp_err:
        result["errors"].append(f"footprint.csv -> {fp_err}")
    else:
        result["footprint_area_deg2"] = area
    # 统计分量数量
    if footprint_path.is_file():
        try:
            with footprint_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                result["n_components"] = len(
                    {row["component_id"] for row in reader if row.get("component_id")}
                )
        except OSError:
            pass

    result["ok"] = not result["errors"]
    return result


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    """将数值格式化为固定小数位字符串; ``None`` 显示为 'N/A'."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(results: List[Dict[str, Any]]) -> str:
    """生成完整校验报告文本.

    Args:
        results: 各卡校验结果列表.

    Returns:
        报告字符串.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GOSIM 巡天智能体任务卡 - 静态校验报告")
    lines.append(f"生成时间: {now}")
    lines.append(f"卡片目录: {CARDS_DIR}")
    lines.append(f"卡片总数: {len(results)}")
    lines.append("=" * 72)
    lines.append("")

    # 汇总表
    lines.append("-" * 72)
    lines.append("汇总表")
    lines.append("-" * 72)
    header = (
        f"{'卡片':<8} {'状态':<6} {'目标数':>8} {'必观测':>7} "
        f"{'天区面积(deg^2)':>16} {'光纤数':>7} {'flux_zp':>8} {'exp_zp(s)':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        lines.append(
            f"{r['name']:<8} {status:<6} {_fmt(r['n_targets']):>8} "
            f"{_fmt(r['n_required']):>7} {_fmt(r['footprint_area_deg2']):>16} "
            f"{_fmt(r['n_fibers']):>7} {_fmt(r['flux_zero_point']):>8} "
            f"{_fmt(r['exposure_zero_point_seconds']):>10}"
        )
    lines.append("")

    # 各卡明细
    for r in results:
        lines.append("-" * 72)
        lines.append(f"卡片: {r['name']}   [{ 'PASS' if r['ok'] else 'FAIL' }]")
        lines.append(f"路径: {r['path']}")
        lines.append("-" * 72)
        lines.append(f"  总目标数            : {_fmt(r['n_targets'])}")
        lines.append(f"  必观测目标数        : {_fmt(r['n_required'])}")
        lines.append(f"  天区面积 (deg^2)    : {_fmt(r['footprint_area_deg2'])}")
        lines.append(f"  天区分量数          : {_fmt(r['n_components'])}")
        lines.append(f"  光纤数量            : {_fmt(r['n_fibers'])}")
        lines.append(f"  flux_zero_point     : {_fmt(r['flux_zero_point'])}")
        lines.append(
            f"  exposure_zero_point_seconds : "
            f"{_fmt(r['exposure_zero_point_seconds'])}"
        )
        if r["errors"]:
            lines.append("  [错误]")
            for err in r["errors"]:
                lines.append(f"    - {err}")
        else:
            lines.append("  [错误] 无")
        if r["warnings"]:
            lines.append("  [警告]")
            for warn in r["warnings"]:
                lines.append(f"    - {warn}")
        lines.append("")

    n_pass = sum(1 for r in results if r["ok"])
    lines.append("=" * 72)
    lines.append(f"校验结束: {n_pass}/{len(results)} 张卡片通过")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def discover_cards(cards_dir: Path) -> List[Path]:
    """发现 ``cards/`` 下的所有卡片目录 (按名称排序)."""
    if not cards_dir.is_dir():
        return []
    return sorted(p for p in cards_dir.iterdir() if p.is_dir())


def main() -> int:
    """脚本入口. 返回进程退出码 (0 正常)."""
    if not CARDS_DIR.is_dir():
        print(f"[错误] 未找到卡片目录: {CARDS_DIR}", file=sys.stderr)
        return 1

    card_dirs = discover_cards(CARDS_DIR)
    if not card_dirs:
        print(f"[错误] {CARDS_DIR} 下没有卡片目录", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    for card_dir in card_dirs:
        res = check_card(card_dir)
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        print(
            f"[{status}] {res['name']:<8} "
            f"目标={_fmt(res['n_targets'])} "
            f"必观测={_fmt(res['n_required'])} "
            f"天区面积={_fmt(res['footprint_area_deg2'])}deg^2 "
            f"光纤={_fmt(res['n_fibers'])} "
            f"flux_zp={_fmt(res['flux_zero_point'])} "
            f"exp_zp={_fmt(res['exposure_zero_point_seconds'])}s"
        )
        for err in res["errors"]:
            print(f"        ! {err}")

    report = render_report(results)
    try:
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\n完整报告已写入: {REPORT_PATH}")
    except OSError as exc:
        print(f"[错误] 无法写入报告: {exc}", file=sys.stderr)
        return 2

    n_pass = sum(1 for r in results if r["ok"])
    return 0 if n_pass == len(results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
