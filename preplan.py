#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preplan.py - GOSIM 巡天智能体独立预规划模块 (纯静态计算).

设计约束 (严格遵守):
    * 纯静态计算: 不接入模拟器、不调用 LLM、不修改任何已有 agent 主程序;
    * 仅依赖 Python 标准库;
    * 既可命令行独立运行, 也可作为模块被智能体主程序 ``import preplan`` 调用.

核心能力:
    1. 读取指定任务卡的全部目标数据与配置;
    2. 目标优先级打分:
       - 必观测目标 (``required == true``) 为最高优先级 (强制前置);
       - 次级权重综合科学权重 (science_weight)、目标亮度 (feature_flux)
         与天区分布均匀度 (赤经 / 赤纬分箱后稀疏区域加权);
    3. 光纤分配算法:
       - 输入候选目标集合, 输出最多 ``n_fibers`` 个目标;
       - 满足方形 (默认 4x4) 光纤方格的空间匹配约束;
       - 优先保证高优先级 (尤其必观测) 目标入选;
    4. 输出候选观测目标列表: 目标 ID、坐标、优先级、建议曝光时长参考值.

典型用法 (命令行)::

    py preplan.py --card alpha
    py preplan.py --card cardB --max-targets 16 --top 20
    py preplan.py --path cards/cardD

典型用法 (作为库)::

    from preplan import CardData, plan_card

    card = CardData.from_card("alpha")
    result = plan_card(card)
    for item in result.selected:
        print(item.target_id, item.priority, item.exposure_seconds)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
CARDS_DIR: Path = PROJECT_ROOT / "cards"

# 光纤方格边长 (4x4 = 16 根光纤, 与练习卡一致); 当卡内光纤数不同时按最接近的方阵处理.
DEFAULT_GRID_SIDE: int = 4

# 优先级打分权重
REQUIRED_BASE_SCORE: float = 1_000.0   # 必观测基础分, 确保强制前置
SCIENCE_WEIGHT_SCALE: float = 10.0     # 科学权重贡献系数
FLUX_WEIGHT_SCALE: float = 2.0         # 亮度贡献系数
UNIFORMITY_WEIGHT_SCALE: float = 1.0   # 天区均匀度贡献系数
DEFAULT_SCIENCE_WEIGHT: float = 1.0
DEFAULT_FEATURE_FLUX: float = 0.5

# 建议曝光时长: 以配置中的计分基准曝光时长为锚点, 按亮度做温和缩放.
MIN_EXPOSURE_FACTOR: float = 0.5
MAX_EXPOSURE_FACTOR: float = 2.0

# 天区均匀度分箱宽度 (度)
UNIFORMITY_BIN_DEG: float = 10.0


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """单个观测目标."""

    target_id: str
    ra_deg: float
    dec_deg: float
    target_class: str
    feature_flux: float
    science_weight: float
    required: bool

    @property
    def is_required(self) -> bool:
        """是否为必观测目标."""
        return self.required


@dataclass
class CardData:
    """一张任务卡解析后的静态数据集合."""

    name: str
    root: Path
    targets: List[Target] = field(default_factory=list)
    fiber_config: Dict[str, Any] = field(default_factory=dict)
    score_config: Dict[str, Any] = field(default_factory=dict)
    footprint_vertices: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def n_fibers(self) -> int:
        """光纤数量 (来自 fiber 配置)."""
        return int(self.fiber_config.get("field", {}).get("n_fibers", 0))

    @property
    def fiber_area_deg2(self) -> float:
        """单根光纤覆盖面积 (平方度)."""
        return float(self.fiber_config.get("field", {}).get("fiber_area_deg2", 0.0))

    @property
    def min_exposure_seconds(self) -> float:
        """允许的最短曝光时长 (秒)."""
        return float(
            self.fiber_config.get("exposure", {}).get("min_duration_seconds", 60)
        )

    @property
    def max_exposure_seconds(self) -> float:
        """允许的最长曝光时长 (秒)."""
        return float(
            self.fiber_config.get("exposure", {}).get("max_duration_seconds", 3600)
        )

    @property
    def exposure_zero_point_seconds(self) -> float:
        """计分基准曝光时长 (秒)."""
        return float(self.score_config.get("exposure_zero_point_seconds", 900))

    @property
    def flux_zero_point(self) -> float:
        """计分基准流量零点."""
        return float(self.score_config.get("flux_zero_point", 0.5))

    @property
    def required_targets(self) -> List[Target]:
        """必观测目标子集."""
        return [t for t in self.targets if t.is_required]

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_card(cls, card: str) -> "CardData":
        """从卡片名或路径加载卡片数据.

        Args:
            card: 卡片名称 (如 ``"alpha"``) 或卡片目录路径.

        Returns:
            解析后的 :class:`CardData`.

        Raises:
            FileNotFoundError: 卡片目录或必需文件不存在.
            json.JSONDecodeError: 配置 JSON 无法解析.
        """
        root = resolve_card_path(card)
        fiber_config = _load_json(root / "config" / "v4_fiber_config.json")
        score_config = _load_json(root / "config" / "v4_score_config.json")
        targets = load_targets(root / "public" / "targets.csv")
        footprint = load_footprint(root / "public" / "footprint.csv")
        return cls(
            name=root.name,
            root=root,
            targets=targets,
            fiber_config=fiber_config,
            score_config=score_config,
            footprint_vertices=footprint,
        )


@dataclass
class PlannedTarget:
    """预规划输出的候选观测目标."""

    target_id: str
    ra_deg: float
    dec_deg: float
    target_class: str
    required: bool
    priority: float
    exposure_seconds: int
    fiber_index: int = -1
    notes: str = ""


@dataclass
class PlanResult:
    """单张卡的预规划结果."""

    card_name: str
    n_candidates: int
    n_fibers: int
    selected: List[PlannedTarget] = field(default_factory=list)
    rejected: List[PlannedTarget] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        """入选目标数量."""
        return len(self.selected)


# ---------------------------------------------------------------------------
# IO 工具
# ---------------------------------------------------------------------------


def resolve_card_path(card: str) -> Path:
    """将卡片名或路径解析为实际目录.

    Args:
        card: 卡片名称或目录路径.

    Returns:
        存在的卡片目录 :class:`Path`.

    Raises:
        FileNotFoundError: 无法定位卡片目录.
    """
    candidate = Path(card)
    if candidate.is_dir():
        return candidate.resolve()
    named = CARDS_DIR / card
    if named.is_dir():
        return named.resolve()
    raise FileNotFoundError(f"无法定位任务卡: {card} (尝试过 '{candidate}' 与 '{named}')")


def _load_json(path: Path) -> Dict[str, Any]:
    """加载 JSON 配置文件."""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件缺失: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_bool(value: Optional[str]) -> bool:
    """将 CSV 中的布尔字符串转换为 ``bool``."""
    return (value or "").strip().lower() in ("true", "1", "yes", "y", "t")


def _to_float(value: Optional[str], default: float = 0.0) -> float:
    """安全地将字符串转为 ``float``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_targets(path: Path) -> List[Target]:
    """加载 ``targets.csv``.

    Args:
        path: ``targets.csv`` 路径.

    Returns:
        目标列表.

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError: 缺少必要列.
    """
    if not path.is_file():
        raise FileNotFoundError(f"目标列表缺失: {path}")
    targets: List[Target] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required_cols = {"target_id", "ra_deg", "dec_deg"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(f"targets.csv 缺少必要列: {required_cols}")
        for row in reader:
            targets.append(
                Target(
                    target_id=str(row.get("target_id", "")).strip(),
                    ra_deg=_to_float(row.get("ra_deg")),
                    dec_deg=_to_float(row.get("dec_deg")),
                    target_class=str(row.get("target_class", "")).strip(),
                    feature_flux=_to_float(
                        row.get("feature_flux"), DEFAULT_FEATURE_FLUX
                    ),
                    science_weight=_to_float(
                        row.get("science_weight"), DEFAULT_SCIENCE_WEIGHT
                    ),
                    required=_to_bool(row.get("required")),
                )
            )
    return targets


def load_footprint(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """加载 ``footprint.csv`` 并按分量分组返回顶点.

    Args:
        path: ``footprint.csv`` 路径.

    Returns:
        ``{component_id: [(ra_deg, dec_deg), ...]}``.
    """
    components: Dict[str, List[Tuple[float, float]]] = {}
    if not path.is_file():
        return components
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return components
        for row in reader:
            cid = str(row.get("component_id", "")).strip()
            if not cid:
                continue
            components.setdefault(cid, []).append(
                (_to_float(row.get("ra_deg")), _to_float(row.get("dec_deg")))
            )
    return components


# ---------------------------------------------------------------------------
# 天区均匀度
# ---------------------------------------------------------------------------


def _uniformity_bins(
    targets: Sequence[Target], bin_deg: float = UNIFORMITY_BIN_DEG
) -> Dict[Tuple[int, int], int]:
    """统计目标在 (RA, Dec) 分箱中的数量分布."""
    bins: Dict[Tuple[int, int], int] = {}
    for t in targets:
        key = (int(t.ra_deg // bin_deg), int(t.dec_deg // bin_deg))
        bins[key] = bins.get(key, 0) + 1
    return bins


def uniformity_scores(
    targets: Sequence[Target], bin_deg: float = UNIFORMITY_BIN_DEG
) -> Dict[str, float]:
    """计算每个目标所在天区稀疏度 (越稀疏分越高).

    稀疏度定义为 ``1 / sqrt(该分箱目标数)``, 归一化到 ``[0, 1]`` 区间.
    稀疏的天区被赋予更高权重, 从而提升天区分布均匀度.

    Args:
        targets: 目标序列.
        bin_deg: 分箱宽度 (度).

    Returns:
        ``{target_id: uniformity_score}``.
    """
    bins = _uniformity_bins(targets, bin_deg)
    if not bins:
        return {}
    raw: Dict[str, float] = {}
    for t in targets:
        key = (int(t.ra_deg // bin_deg), int(t.dec_deg // bin_deg))
        raw[t.target_id] = 1.0 / math.sqrt(bins.get(key, 1))
    lo = min(raw.values()) if raw else 0.0
    hi = max(raw.values()) if raw else 0.0
    if hi - lo < 1e-12:
        return {tid: 1.0 for tid in raw}
    return {tid: (v - lo) / (hi - lo) for tid, v in raw.items()}


# ---------------------------------------------------------------------------
# 优先级打分
# ---------------------------------------------------------------------------


def compute_priorities(card: CardData) -> Dict[str, float]:
    """为卡片内所有目标计算优先级分数.

    打分规则:
        * 必观测目标: ``REQUIRED_BASE_SCORE`` 基础分 (强制前置), 叠加次级项;
        * 次级权重 = 科学权重 * ``SCIENCE_WEIGHT_SCALE``
                     + 亮度 * ``FLUX_WEIGHT_SCALE``
                     + 均匀度 * ``UNIFORMITY_WEIGHT_SCALE``.

    Args:
        card: 已加载的卡片数据.

    Returns:
        ``{target_id: priority}``.
    """
    scores: Dict[str, float] = {}
    uniformity = uniformity_scores(card.targets)
    for t in card.targets:
        secondary = (
            max(t.science_weight, 0.0) * SCIENCE_WEIGHT_SCALE
            + max(t.feature_flux, 0.0) * FLUX_WEIGHT_SCALE
            + uniformity.get(t.target_id, 0.0) * UNIFORMITY_WEIGHT_SCALE
        )
        base = REQUIRED_BASE_SCORE if t.is_required else 0.0
        scores[t.target_id] = base + secondary
    return scores


def suggest_exposure_seconds(
    target: Target, card: CardData
) -> int:
    """为单个目标建议曝光时长 (秒).

    以计分基准曝光时长为锚点, 依据相对亮度做温和缩放: 越暗曝光越久.
    结果被裁剪到配置允许的 ``[min_duration_seconds, max_duration_seconds]``.

    Args:
        target: 目标.
        card: 卡片数据.

    Returns:
        建议曝光秒数 (整数).
    """
    anchor = card.exposure_zero_point_seconds
    flux_ref = max(card.flux_zero_point, 1e-9)
    flux = max(target.feature_flux, 1e-9)
    # 亮度越低 -> ratio 越大 -> 曝光越长
    ratio = flux_ref / flux
    factor = math.sqrt(max(ratio, 1e-9))
    factor = max(MIN_EXPOSURE_FACTOR, min(MAX_EXPOSURE_FACTOR, factor))
    seconds = anchor * factor
    seconds = max(card.min_exposure_seconds, min(card.max_exposure_seconds, seconds))
    return int(round(seconds))


# ---------------------------------------------------------------------------
# 光纤方格空间匹配
# ---------------------------------------------------------------------------


@dataclass
class FiberGrid:
    """方形光纤网格 (默认 4x4).

    使用切平面近似: 以候选目标集的中心为切点, 将 (RA, Dec) 投影为
    局部 (east, north) 偏移 (度), 并映射到方格中的光纤单元.
    """

    side: int
    n_fibers: int
    fiber_side_deg: float  # 单根光纤边长 (度)
    gap_deg: float = 0.0

    @property
    def pitch_deg(self) -> float:
        """相邻光纤中心间距 (度)."""
        return self.fiber_side_deg + self.gap_deg

    @property
    def fov_side_deg(self) -> float:
        """整个视场边长 (度)."""
        return self.side * self.fiber_side_deg + (self.side - 1) * self.gap_deg

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """返回 (row, col) 光纤单元的局部中心偏移 (east, north), 单位度."""
        middle = (self.side - 1) / 2.0
        east = (col - middle) * self.pitch_deg
        north = (row - middle) * self.pitch_deg
        return east, north

    def classify(self, east: float, north: float) -> Tuple[int, float]:
        """将局部偏移映射到光纤索引.

        Args:
            east: 向东偏移 (度).
            north: 向北偏移 (度).

        Returns:
            二元组 ``(fiber_index, margin)``; ``fiber_index`` 为 ``-1`` 表示
            超出视场, ``margin`` 为到光纤玻璃边缘的余量 (度).
        """
        half = self.fov_side_deg / 2.0
        if abs(east) > half or abs(north) > half:
            return -1, -1.0
        middle = self.side / 2.0
        col = int(math.floor(east / self.pitch_deg + middle))
        row = int(math.floor(north / self.pitch_deg + middle))
        col = min(max(col, 0), self.side - 1)
        row = min(max(row, 0), self.side - 1)
        fiber = row * self.side + col
        c_east, c_north = self.cell_center(row, col)
        margin = self.fiber_side_deg / 2.0 - max(
            abs(east - c_east), abs(north - c_north)
        )
        return (fiber, margin) if margin >= 0.0 else (-1, margin)


def build_fiber_grid(card: CardData, side: Optional[int] = None) -> FiberGrid:
    """根据卡片配置构建光纤网格.

    单根光纤边长由 ``fiber_area_deg2`` 开平方得到 (方形近似), 方格边长
    ``side`` 默认为 4 (即 4x4), 或按 ``n_fibers`` 选择最接近的完全平方数.

    Args:
        card: 卡片数据.
        side: 指定方格边长; ``None`` 时自动推断.

    Returns:
        构建好的 :class:`FiberGrid`.
    """
    n = card.n_fibers or (DEFAULT_GRID_SIDE ** 2)
    if side is None:
        root = int(round(math.sqrt(n))) if n > 0 else DEFAULT_GRID_SIDE
        side = max(1, root)
    fiber_side = math.sqrt(max(card.fiber_area_deg2, 1e-9))
    gap = float(card.fiber_config.get("field", {}).get("gap_deg", 0.0))
    return FiberGrid(side=side, n_fibers=n, fiber_side_deg=fiber_side, gap_deg=gap)


def _tangent_offsets(
    ra_deg: float, dec_deg: float, center_ra: float, center_dec: float
) -> Tuple[float, float]:
    """将目标投影到以 center 为切点的局部 (east, north) 平面偏移 (度).

    采用标准切平面 (gnomonic) 近似, 适用于巡天小视场.
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    cra = math.radians(center_ra)
    cdec = math.radians(center_dec)
    d_ra = ra - cra
    if d_ra > math.pi:
        d_ra -= 2.0 * math.pi
    elif d_ra < -math.pi:
        d_ra += 2.0 * math.pi
    cos_dec = math.cos(dec)
    sin_dec = math.sin(dec)
    cos_cdec = math.cos(cdec)
    sin_cdec = math.sin(cdec)
    cos_dra = math.cos(d_ra)
    sin_dra = math.sin(d_ra)
    denom = sin_dec * sin_cdec + cos_dec * cos_cdec * cos_dra
    if denom <= 1e-12:
        return 0.0, 0.0
    east = math.degrees(cos_dec * sin_dra / denom)
    north = math.degrees(
        (sin_dec * cos_cdec - cos_dec * sin_cdec * cos_dra) / denom
    )
    return east, north


def _fill_pointing(
    grid: FiberGrid,
    ranked: Sequence[Target],
    priorities: Dict[str, float],
    center_ra: float,
    center_dec: float,
    limit: int,
) -> Tuple[List[Tuple[Target, int, float]], float]:
    """在给定视场中心下, 按优先级贪心填充光纤方格.

    Args:
        grid: 光纤网格.
        ranked: 已按优先级降序的目标序列.
        priorities: 优先级映射.
        center_ra: 视场中心赤经 (度).
        center_dec: 视场中心赤纬 (度).
        limit: 本视场最多入选目标数.

    Returns:
        二元组 ``(placement, total_priority)``; ``placement`` 为
        ``[(target, fiber_index, margin), ...]``.
    """
    placement: List[Tuple[Target, int, float]] = []
    used_fibers: Dict[int, str] = {}
    total = 0.0
    for t in ranked:
        if len(placement) >= limit:
            break
        east, north = _tangent_offsets(t.ra_deg, t.dec_deg, center_ra, center_dec)
        fiber, margin = grid.classify(east, north)
        if fiber < 0 or fiber in used_fibers:
            continue
        used_fibers[fiber] = t.target_id
        placement.append((t, fiber, margin))
        total += priorities.get(t.target_id, 0.0)
    return placement, total


def assign_fibers(
    card: CardData,
    priorities: Dict[str, float],
    max_targets: Optional[int] = None,
    n_anchors: int = 64,
) -> Tuple[List[PlannedTarget], List[PlannedTarget]]:
    """执行光纤分配: 输出最多 ``n_fibers`` 个满足方格约束的目标.

    算法 (滑动视场聚类搜索):
        1. 按优先级降序排序全部候选 (必观测天然最高);
        2. 取前 ``n_anchors`` 个高优先级目标, 以及必观测目标的球面重心,
           分别作为候选视场中心;
        3. 在每个候选中心下, 将邻近目标投影到切平面, 按优先级贪心填充
           方形光纤方格 (每个光纤单元至多一个目标);
        4. 选择总优先级最高的视场作为最终点规划结果;
        5. 若该视场未填满 ``max_targets``, 用剩余最高优先级目标继续在
           其它可行视场中补充, 直至达到上限或无可用目标.

    Args:
        card: 卡片数据.
        priorities: ``{target_id: priority}``.
        max_targets: 入选目标上限; ``None`` 时取 ``n_fibers``.
        n_anchors: 用作视场中心的高优先级目标数量上限.

    Returns:
        二元组 ``(selected, rejected)``, 均为 :class:`PlannedTarget` 列表.
    """
    grid = build_fiber_grid(card)
    limit = max_targets if max_targets is not None else (card.n_fibers or grid.n_fibers)
    if limit <= 0:
        limit = grid.side * grid.side

    ranked = sorted(
        card.targets, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True
    )
    by_id = {t.target_id: t for t in card.targets}

    def make_planned(t: Target, fiber: int, note: str) -> PlannedTarget:
        return PlannedTarget(
            target_id=t.target_id,
            ra_deg=t.ra_deg,
            dec_deg=t.dec_deg,
            target_class=t.target_class,
            required=t.is_required,
            priority=priorities.get(t.target_id, 0.0),
            exposure_seconds=suggest_exposure_seconds(t, card),
            fiber_index=fiber,
            notes=note,
        )

    if not ranked:
        return [], []

    # -- 构造候选视场中心 (anchors) -------------------------------------
    anchor_points: List[Tuple[float, float]] = []
    required = [t for t in ranked if t.is_required]
    if required:
        anchor_points.append(_centroid(required))
    anchor_points.append(_centroid(ranked[:1]))
    for t in ranked[:n_anchors]:
        anchor_points.append((t.ra_deg, t.dec_deg))

    # -- 评估每个候选中心, 取总优先级最优的视场 -------------------------
    best_placement: List[Tuple[Target, int, float]] = []
    best_total = -1.0
    for center_ra, center_dec in anchor_points:
        placement, total = _fill_pointing(
            grid, ranked, priorities, center_ra, center_dec, limit
        )
        # 优先保证必观测数量, 其次看总优先级
        req_count = sum(1 for t, _, _ in placement if t.is_required)
        best_req = sum(1 for t, _, _ in best_placement if t.is_required)
        score_key = (req_count, total)
        best_key = (best_req, best_total)
        if score_key > best_key:
            best_placement = placement
            best_total = total

    selected = [make_planned(t, f, f"margin={m:.4f}deg") for t, f, m in best_placement]
    chosen_ids = {t.target_id for t, _, _ in best_placement}

    # -- 补充: 若最优视场未填满上限, 依次尝试其它中心继续纳入 -----------
    if len(selected) < limit:
        for center_ra, center_dec in anchor_points:
            if len(selected) >= limit:
                break
            placement, _ = _fill_pointing(
                grid, ranked, priorities, center_ra, center_dec, limit
            )
            for t, f, m in placement:
                if len(selected) >= limit:
                    break
                if t.target_id in chosen_ids:
                    continue
                chosen_ids.add(t.target_id)
                selected.append(
                    make_planned(t, f, f"补充视场 margin={m:.4f}deg")
                )

    # -- 落选目标 (仅记录前若干, 避免内存/输出过大) ---------------------
    rejected: List[PlannedTarget] = []
    for t in ranked:
        if t.target_id in chosen_ids:
            continue
        if len(rejected) >= 200:
            break
        rejected.append(make_planned(t, -1, "未进入任一可行视场/光纤"))

    # 按优先级排序输出, 便于阅读
    selected.sort(key=lambda p: p.priority, reverse=True)
    return selected, rejected


def _centroid(targets: Sequence[Target]) -> Tuple[float, float]:
    """计算一组目标的球面重心 (RA, Dec), 单位度."""
    x = y = z = 0.0
    for t in targets:
        ra = math.radians(t.ra_deg)
        dec = math.radians(t.dec_deg)
        x += math.cos(dec) * math.cos(ra)
        y += math.cos(dec) * math.sin(ra)
        z += math.sin(dec)
    n = len(targets) or 1
    x, y, z = x / n, y / n, z / n
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.atan2(z, math.hypot(x, y)))
    return ra, dec


# ---------------------------------------------------------------------------
# 顶层接口
# ---------------------------------------------------------------------------


def plan_card(
    card: CardData, max_targets: Optional[int] = None
) -> PlanResult:
    """对单张卡执行完整预规划.

    这是供智能体主程序调用的主要接口.

    Args:
        card: 已加载的 :class:`CardData`.
        max_targets: 入选目标上限; ``None`` 时取 ``n_fibers``.

    Returns:
        :class:`PlanResult`.
    """
    priorities = compute_priorities(card)
    selected, rejected = assign_fibers(card, priorities, max_targets=max_targets)
    return PlanResult(
        card_name=card.name,
        n_candidates=len(card.targets),
        n_fibers=card.n_fibers,
        selected=selected,
        rejected=rejected,
    )


def plan_by_name(
    card: str, max_targets: Optional[int] = None
) -> PlanResult:
    """便捷接口: 按卡片名/路径加载并规划.

    Args:
        card: 卡片名称或路径.
        max_targets: 入选目标上限.

    Returns:
        :class:`PlanResult`.
    """
    return plan_card(CardData.from_card(card), max_targets=max_targets)


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def render_plan_text(result: PlanResult, top: int = 16) -> str:
    """将预规划结果渲染为可读文本.

    Args:
        result: 预规划结果.
        top: 最多展示的候选目标数.

    Returns:
        文本字符串.
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"预规划结果 - 卡片: {result.card_name}")
    lines.append("=" * 72)
    lines.append(f"候选目标总数    : {result.n_candidates}")
    lines.append(f"光纤总数        : {result.n_fibers}")
    lines.append(f"入选目标数      : {result.n_selected}")
    lines.append(f"落选目标数      : {len(result.rejected)}")
    lines.append("-" * 72)
    header = (
        f"{'目标ID':<12} {'RA(deg)':>10} {'Dec(deg)':>10} {'类型':<6} "
        f"{'必观测':>6} {'优先级':>10} {'曝光(s)':>8} {'光纤':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for p in result.selected[:top]:
        lines.append(
            f"{p.target_id:<12} {p.ra_deg:>10.4f} {p.dec_deg:>10.4f} "
            f"{p.target_class:<6} {('是' if p.required else '否'):>6} "
            f"{p.priority:>10.2f} {p.exposure_seconds:>8} {p.fiber_index:>5}"
        )
    if result.n_selected > top:
        lines.append(f"... 以及其余 {result.n_selected - top} 个入选目标")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description="GOSIM 任务卡独立预规划模块 (纯静态计算)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--card", "-c", help="卡片名称, 例如 alpha / cardB")
    group.add_argument("--path", "-p", help="卡片目录路径")
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="入选目标上限 (默认取光纤数量)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=16,
        help="控制台最多展示的目标数",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="可选: 将完整结果写入指定文本文件 (UTF-8)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口. 返回进程退出码."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    card_ref = args.card if args.card else args.path

    try:
        card = CardData.from_card(card_ref)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] 加载卡片失败: {exc}", file=sys.stderr)
        return 1

    result = plan_card(card, max_targets=args.max_targets)
    text = render_plan_text(result, top=args.top)
    try:
        print(text)
    except UnicodeEncodeError:
        # 兼容 GBK 控制台: 用一个安全编码回退, 避免脚本崩溃.
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))

    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"[信息] 结果已写入: {args.out}")
        except OSError as exc:
            print(f"[警告] 无法写入输出文件: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
