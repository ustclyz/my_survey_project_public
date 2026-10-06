#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""planner.py - 决策内核.

本模块把现有 ``preplan.py`` **封装成可调用的规划工具**, 并提供:
    * 目标可见性/几何计算 (RA/Dec <-> alt/az, 太阳/月亮, 4x4 光纤方格);
    * 把"候选目标集"落成一个合法的 ``observe`` 动作 (pointing + 16 光纤分配 + 曝光 + program);
    * observe/wait/report/finish 的启发式决策;
    * 与 LLM 环节协作: LLM 给高层意向, 本模块负责落成合法动作.

**注意: 不重写规划逻辑** —— 目标优先级打分与光纤分配复用
``preplan.compute_priorities`` / ``preplan.assign_fibers`` / ``preplan.plan_card``.

预算保护 (900s CPU 成败项):
    * 环节A (``plan_night``) 仅在新夜调用并缓存;
    * 环节B (``decide_action``) **绝不每轮调用**, 只在关键点或按预算自适应的周期
      调用 (:meth:`Planner._should_call_llm_decide`);
    * 其余轮次用静态内核落成动作, 保证巡天能在预算内完成.

观测记忆:
    * 记录已得分目标 (:attr:`Planner.observed_ids`, 强惩罚) 与已尝试目标
      (:attr:`Planner.attempted_ids`, 弱惩罚), 避免重复曝光 (规则 5.4);
    * 记录上次指向并对重复视场施加惩罚 (:meth:`Planner._repeat_pointing_penalty`).

几何公式来源: 官方 ``docs/participant-guide`` 的 Geometry 章节与官方示例
``agent_core/geometry.py`` (公有公式, 不依赖任何隐藏数据).
"""

from __future__ import annotations

import math
import time
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import preplan
from models import Action, DecisionState, NightPlan, Target
from protocol import log
from skyindex import HAVE_NUMPY, TargetIndex

# ---------------------------------------------------------------------------
# 天文几何 (标准公式, 与官方示例一致)
# ---------------------------------------------------------------------------

SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0
DEFAULT_MIN_ALTITUDE_DEG = 30.0
ALT_MARGIN_DEG = 1.0  # 规划时目标需高于最低高度角的安全余量
# 曝光时长估算使用的"典型 seeing"参考值 (官方隐藏真值不可知). 取偏保守 (较大) 的
# 值, 使必观测目标的完成曝光偏长, 从而在常见/偏差天气下更稳地达到完成因子 0.5.
SEEING_REF_FOR_SCORING = 1.1

# ---------------------------------------------------------------------------
# 性能预算调参 (只影响"每轮算多少", 不改变动作合法性)
# ---------------------------------------------------------------------------
# 背景: 每轮决策原本要对全部 N 个目标重算可见性/优先级, 并评估 ~192 个候选
# 视场中心; 在 3 万目标 x 数千轮决策的规模下会吃光 900s CPU 预算。下面这些
# 上限把"每轮成本"与卡规模解耦 (本机实测卡 A 单决策 ~500ms -> 目标 <150ms)。
MAX_CENTERS_PER_DECISION = 64     # 每轮候选视场中心的硬上限
PROBE_MAX = 60                    # 稠密探测阶段最多评估的锚点数
FINE_FIBER_MAX = 16               # 精细搜索时每个锚点采样的光纤中心数
MAX_REFINE_CENTERS = 32           # 精细搜索阶段最多评估的中心数 (须给后续阶段留额度)
MAX_REQUIRED_ANCHOR_CENTERS = 24  # 必观测锚定每轮最多生成的候选中心数
MAX_CONTINUITY_CENTERS = 8        # 连续性种子每轮最多生成的候选中心数
REQUEST_BONUS = 800.0             # 限时观测请求目标的优先级加成
REQUEST_BONUS_URGENT = 1200.0     # 距截止不足 1 夜时的加成

# 曝光估算的保守系数: 公开模型缺 η/τ/K/seeing 真值, 实测(卡B)表明原估计偏乐观
# 15~40% (106 个必观测卡在 g∈[0.40,0.50)), 这里把估计质量乘以该系数,
# 等价于把"达标所需曝光"放大约 1/0.65 ≈ 1.54 倍。
COMPLETION_Q_SCALE = 0.65

# 必观测视场的"硬优先"权重: 只要视场含 1 个未完成必观测, 其质量就必须压过任何
# 不含必观测的视场 (含"重试堆"视场, 后者 16x(11+300) ≈ 5000), 否则必观测会被
# 反复失败的目标挤出局 (实测卡B: 161 个必观测全场可见却一次都没被指派)。
REQUIRED_FIELD_DOMINANCE = 100000.0
# 重试加成只给前 N 次尝试; 之后转为抑制, 避免失败目标长期挤占新鲜目标。
RETRY_BONUS = 300.0
RETRY_BONUS_MAX_ATTEMPTS = 2
STALE_PENALTY = 200.0
# 从未尝试过的可见目标获得加成 (覆盖优先, 防止"只啃硬骨头")。
FRESH_BONUS = 150.0
# 必观测锚定冷却: 刚锚定过的目标让位给其它未完成必观测 (轮转, 避免饿死)。
ANCHOR_COOLDOWN_DECISIONS = 3
# 每个目标的尝试次数硬上限: 反复失败后不再参与本轮选择, 把时间让给新鲜目标。
# (实测卡A1: 11 个目标被拍 ≥50 次而 2.6 万个目标从没被碰过 -> 只覆盖 3,445 个目标)
MAX_ATTEMPTS_NORMAL = 3
MAX_ATTEMPTS_REQUIRED = 8      # 必观测单个值 50 分, 允许更多次尝试


def _julian_date(moment: datetime) -> float:
    return moment.timestamp() / 86400.0 + 2440587.5


def local_sidereal_deg(moment: datetime, longitude_deg: float) -> float:
    days = _julian_date(moment) - 2451545.0
    return (280.46061837 + 360.98564736629 * days + longitude_deg) % 360.0


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def radec_to_altaz(ra_deg: float, dec_deg: float, lst_deg: float, lat_deg: float) -> Tuple[float, float]:
    ha = math.radians(wrap180(lst_deg - ra_deg))
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_alt = max(1e-12, math.cos(alt))
    sin_az = -math.sin(ha) * math.cos(dec) / cos_alt
    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (cos_alt * max(1e-12, math.cos(lat)))
    return math.degrees(alt), math.degrees(math.atan2(sin_az, cos_az)) % 360.0


def altaz_to_radec(alt_deg: float, az_deg: float, lst_deg: float, lat_deg: float) -> Tuple[float, float]:
    alt, az = math.radians(alt_deg), math.radians(az_deg)
    lat = math.radians(lat_deg)
    sin_dec = math.sin(alt) * math.sin(lat) + math.cos(alt) * math.cos(lat) * math.cos(az)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))
    cos_dec = max(1e-12, math.cos(dec))
    sin_h = -math.sin(az) * math.cos(alt) / cos_dec
    cos_h = (math.sin(alt) - math.sin(dec) * math.sin(lat)) / (cos_dec * max(1e-12, math.cos(lat)))
    hour = math.degrees(math.atan2(sin_h, cos_h))
    return (lst_deg - hour) % 360.0, math.degrees(dec)


def shift_altaz(alt_deg: float, az_deg: float, d_north: float, d_east: float) -> Tuple[float, float]:
    alt, az = math.radians(alt_deg), math.radians(az_deg)
    point = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
    north = (-math.sin(alt) * math.cos(az), -math.sin(alt) * math.sin(az), math.cos(alt))
    east = (-math.sin(az), math.cos(az), 0.0)
    dn, de = math.radians(d_north), math.radians(d_east)
    x, y, z = (p + dn * n + de * e for p, n, e in zip(point, north, east))
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return math.degrees(math.asin(max(-1.0, min(1.0, z / norm)))), math.degrees(math.atan2(y, x)) % 360.0


def tangent_offsets(t_alt: float, t_az: float, c_alt: float, c_az: float) -> Optional[Tuple[float, float]]:
    """目标相对视场中心的切平面偏移, 返回 ``(north, east)`` (单位: 度).

    与官方 ``skymath.tangent_offsets`` 一致 (官方同样返回 ``(north, east)``).
    调用方注意: ``preplan.FiberGrid.classify`` 的参数顺序是 ``(east, north)``,
    必须先解包再按该顺序传入, 否则会把光纤方位写反.
    """
    alt, az = math.radians(t_alt), math.radians(t_az)
    calt, caz = math.radians(c_alt), math.radians(c_az)
    t = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
    c = (math.cos(calt) * math.cos(caz), math.cos(calt) * math.sin(caz), math.sin(calt))
    north = (-math.sin(calt) * math.cos(caz), -math.sin(calt) * math.sin(caz), math.cos(calt))
    east = (-math.sin(caz), math.cos(caz), 0.0)
    depth = t[0] * c[0] + t[1] * c[1] + t[2] * c[2]
    if depth <= 0.0:
        return None
    return (
        math.degrees((t[0] * north[0] + t[1] * north[1] + t[2] * north[2]) / depth),
        math.degrees((t[0] * east[0] + t[1] * east[1] + t[2] * east[2]) / depth),
    )


def normalized_airmass(alt_deg: float) -> float:
    """Kasten-Young 大气质量 (相对天顶)."""
    if alt_deg <= 0.0:
        return float("inf")
    zenith = 90.0 - alt_deg
    raw = 1.0 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    return raw / (1.0 / (1.0 + 0.50572 * 96.07995 ** -1.6364))


def _sun_radec(moment: datetime) -> Tuple[float, float]:
    days = _julian_date(moment) - 2451545.0
    mean_longitude = (280.460 + 0.9856474 * days) % 360.0
    anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    longitude = math.radians((mean_longitude + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly)) % 360.0)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return (
        math.degrees(math.atan2(math.cos(obliquity) * math.sin(longitude), math.cos(longitude))) % 360.0,
        math.degrees(math.asin(math.sin(obliquity) * math.sin(longitude))),
    )


def _moon_radec(moment: datetime) -> Tuple[float, float]:
    days = _julian_date(moment) - 2451545.0
    mean_longitude = math.radians((218.316 + 13.176396 * days) % 360.0)
    anomaly = math.radians((134.963 + 13.064993 * days) % 360.0)
    arg_latitude = math.radians((93.272 + 13.229350 * days) % 360.0)
    longitude = mean_longitude + math.radians(6.289) * math.sin(anomaly)
    latitude = math.radians(5.128) * math.sin(arg_latitude)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    x = math.cos(longitude) * math.cos(latitude)
    y = math.sin(longitude) * math.cos(latitude) * math.cos(obliquity) - math.sin(latitude) * math.sin(obliquity)
    z = math.sin(longitude) * math.cos(latitude) * math.sin(obliquity) + math.sin(latitude) * math.cos(obliquity)
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))


def _separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def angular_separation_altaz(alt1: float, az1: float, alt2: float, az2: float) -> float:
    """两个地平坐标方向之间的球面角距 (度).

    直接使用球面余弦定理: 把 (az, alt) 视作球面坐标的一对分量即可, 与赤道坐标
    的角距公式同形. 仅用于判断指向是否重合, 不涉及时角/赤经转换.
    """
    return _separation_deg(az1, alt1, az2, alt2)


class _Moon:
    """月球位置与光照 (供 program 估计)."""

    def __init__(self, moment: datetime, lst_deg: float, latitude_deg: float):
        self.ra, self.dec = _moon_radec(moment)
        sun_ra, sun_dec = _sun_radec(moment)
        self.illumination = (1.0 - math.cos(math.radians(_separation_deg(sun_ra, sun_dec, self.ra, self.dec)))) / 2.0
        self.alt, _ = radec_to_altaz(self.ra, self.dec, lst_deg, latitude_deg)


def _lunar_factor(moon: "_Moon", ra_deg: float, dec_deg: float, model: Dict[str, Any]) -> float:
    """公开的月球质量因子 (1.0 = 无惩罚)."""
    if moon.alt <= 0.0:
        return 1.0
    separation = _separation_deg(ra_deg, dec_deg, moon.ra, moon.dec)
    try:
        penalty = (
            float(model.get("maximum_penalty", 0.5))
            * moon.illumination
            * math.sin(math.radians(max(0.0, moon.alt))) ** float(model.get("altitude_exponent", 1.0))
            * math.exp(-separation / float(model.get("angular_decay_scale_deg", 40.0)))
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return max(0.0, min(1.0, 1.0 - penalty))


def max_hour_angle_deg(dec_deg: float, lat_deg: float, min_alt_deg: float) -> float:
    denominator = math.cos(math.radians(lat_deg)) * math.cos(math.radians(dec_deg))
    if abs(denominator) < 1e-12:
        return 0.0
    value = (math.sin(math.radians(min_alt_deg)) - math.sin(math.radians(lat_deg)) * math.sin(math.radians(dec_deg))) / denominator
    if value >= 1.0:
        return 0.0
    if value <= -1.0:
        return 180.0
    return math.degrees(math.acos(value))


# ---------------------------------------------------------------------------
# SportBoard: 规划工具 (封装 preplan.py)
# ---------------------------------------------------------------------------


class PlannerTool:
    """把 ``preplan.py`` 封装为 LLM 可调用的"规划工具".

    LLM 环节可通过 :meth:`describe` 了解工具能力, 决策内核通过
    :meth:`candidates_from_targets` / :meth:`plans_by_name` 复用其算法.
    """

    def __init__(self, card: Optional[preplan.CardData] = None):
        self.card = card

    def describe(self) -> str:
        cards = "alpha/beta/gamma/delta/cardA..cardD"
        return (
            f"规划工具 preplan.py: 提供优先级打分(必观测强制最高)、4x4 光纤方格分配、"
            f"建议曝光时长; 可用卡片: {cards}. "
            f"当前卡: {self.card.name if self.card else '(未加载, 仅用协议目标)'}"
        )

    def plan_for_card(self, max_targets: int = 16):
        """对已加载卡片执行 preplan.plan_card (复用现有规划逻辑)."""
        if self.card is None:
            return None
        return preplan.plan_card(self.card, max_targets=max_targets)

    def ranked_candidate_ids(self, limit: int = 64) -> List[str]:
        """用 preplan 的优先级对卡片目标排序, 返回前 limit 个目标 ID."""
        if self.card is None:
            return []
        priorities = preplan.compute_priorities(self.card)
        ranked = sorted(self.card.targets, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True)
        return [t.target_id for t in ranked[:limit]]

    def assign_for_targets(
        self,
        targets: Sequence[Target],
        priorities: Optional[Dict[str, float]] = None,
        max_targets: int = 16,
    ):
        """对任意目标子集执行 preplan 的光纤分配 (复用 assign_fibers).

        Args:
            targets: 目标子集 (本项目 ``models.Target``).
            priorities: 可选的优先级覆盖; 缺省按子集内部计算.
            max_targets: 上限 (<= 光纤数).

        Returns:
            ``(selected, rejected)``, 其中 selected 为 ``preplan.PlannedTarget`` 列表.
        """
        if not targets:
            return [], []
        # 构造一个临时 CardData 以复用 preplan 的算法与配置
        card = self._temp_card(targets)
        if priorities is None:
            priorities = preplan.compute_priorities(card)
        return preplan.assign_fibers(card, priorities, max_targets=max_targets)

    def _temp_card(self, targets: Sequence[Target]) -> preplan.CardData:
        """用给定目标构造临时 CardData (复用原卡配置, 保证算法一致)."""
        converted = [
            preplan.Target(
                target_id=t.target_id,
                ra_deg=t.ra_deg,
                dec_deg=t.dec_deg,
                target_class=t.target_class,
                feature_flux=t.feature_flux,
                science_weight=t.science_weight,
                required=t.required,
            )
            for t in targets
        ]
        if self.card is not None:
            return preplan.CardData(
                name=self.card.name,
                root=self.card.root,
                targets=converted,
                fiber_config=self.card.fiber_config,
                score_config=self.card.score_config,
                footprint_vertices=self.card.footprint_vertices,
            )
        # 无卡片: 使用协议/默认配置
        return preplan.CardData(
            name="runtime",
            root=preplan.PROJECT_ROOT,
            targets=converted,
            fiber_config={"field": {"n_fibers": 16, "fiber_area_deg2": 0.4, "gap_deg": 0.0},
                          "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}},
            score_config={},
            footprint_vertices={},
        )


# ---------------------------------------------------------------------------
# 决策内核
# ---------------------------------------------------------------------------


@dataclass
class _TargetTable:
    """运行期目标表 (从协议或卡片加载)."""

    targets: List[Target] = field(default_factory=list)
    index: Dict[str, Target] = field(default_factory=dict)

    def add(self, t: Target) -> None:
        self.targets.append(t)
        self.index[t.target_id] = t


class Planner:
    """决策内核: 协作 LLM 环节与 preplan 工具, 产出合法动作."""

    DURATIONS = (300, 600, 900, 1200, 1800, 2400, 3600)

    def __init__(
        self,
        init_data,
        planner_tool: PlannerTool,
        llm_planner=None,
        log_fn=log,
    ) -> None:
        self.init = init_data
        self.tool = planner_tool
        self.llm = llm_planner
        self.log = log_fn

        self.lat = init_data.lat_deg
        self.lon = init_data.lon_deg
        self.min_alt = init_data.min_altitude_deg
        self.min_exposure = init_data.min_exposure_seconds
        self.max_exposure = init_data.max_exposure_seconds
        self.n_fibers = init_data.n_fibers
        self.grid_side = init_data.grid_side
        self.slot_seconds = init_data.slot_seconds

        # 平台正式赛下发的 card_id 是 A/B/C/D/A1…, 而仓库目录名是 cardA/cardB/…:
        # agent 侧可能解析不到卡片文件。此时用 initialize payload 里的仪器与计分参数
        # 构造一个"运行期卡片", 保证光纤几何与曝光基准取自真实卡配置, 而不是退回
        # 硬编码的 0.4 deg² (曾使 B/C/D 及 A1-D1 的视场大小算错)。
        if self.tool.card is None:
            self.tool.card = self._runtime_card_from_payload()

        self.targets = _TargetTable()
        self._load_targets()

        self.consecutive_reports = 0
        self.reports_made = 0
        self.hits_total = 0
        self.assigned_total = 0
        self.last_night_index: Optional[int] = None
        self.night_plan: Optional[NightPlan] = None
        self.duration_scale = 1.0
        self.extra_avoid: List[str] = []
        self.report_fault_hint = False
        self._fiber_grid = self._build_grid()
        self._priorities_all: Optional[Dict[str, float]] = None  # 全体目标优先级缓存

        # -- 每轮共享的预计算 (性能关键) --------------------------------------
        # 夜间起止时间: 原本每轮都要对全部夜重新做 ISO 解析 (卡 D 每轮 1800+ 次),
        # 这里一次性解析好; 索引与 init.nights 一一对应 (解析失败处为 None)。
        self._nights: List[Tuple[Optional[datetime], Optional[datetime]]] = [
            (_parse_utc(n.get("observing_start_utc")), _parse_utc(n.get("observing_end_utc")))
            for n in self.init.nights
        ]
        # 目标可见性/地平坐标索引: 一次性算好 h_max 等, 之后每轮只做数组比较。
        self._sky = TargetIndex(self.targets.targets, self.lat, self.min_alt + ALT_MARGIN_DEG)
        # 每轮只算一次的可见性快照 (供 _build_state 与 _plan_observe 复用)
        self._snapshot: Optional[Tuple[float, List[Target], Dict[str, Tuple[float, float]]]] = None
        # 必观测集合/达标阈值: 与时间无关, 预计算一次 (原实现每轮重建一个数万条的集合)
        self._required_ids = {t.target_id for t in self.targets.targets if t.required}
        try:
            self._required_threshold = float(
                (init_data.scoring or {}).get("required", {}).get("observed_factor_threshold", 0.5)
            )
        except (AttributeError, TypeError, ValueError):
            self._required_threshold = 0.5
        if self._required_threshold <= 0.0:
            self._required_threshold = 0.5

        # -- 性能计数 (只写 stderr 日志 / 本地预算仿真, 不影响决策) -------------
        self.perf_fill_calls = 0      # 本轮 _fill_pointing 调用次数
        self.perf_visible_n = 0       # 本轮可见目标数
        self._cpu_mark: Optional[float] = None
        self.decision_cpu_total = 0.0
        self.decision_cpu_count = 0

        # -- 观测记忆: 已观测/已得分目标 (避免重复曝光同一视场) --------------
        # 规则: 每个目标只算它最好的一次曝光, 多次曝光不累加 (规则 5.4).
        self.observed_ids: set = set()          # 已成功得分的目标 ID
        self.attempted_ids: set = set()         # 已尝试观测 (无论是否命中) 的目标 ID
        self.last_pointing: Optional[Tuple[float, float]] = None  # 上次指向 (alt, az)
        self.repeat_pointings = 0               # 与上次几乎重合的指向计数

        # -- 扫描连续性 / 收敛 (让相邻曝光形成连贯扫描而非乱跳) ----------------
        # 相邻轮指向平均跳变 28°, 会让望远镜"瞬移"且破坏天区均匀覆盖. 这里引入
        # 迟滞 (hysteresis) 与惯性 (momentum): 在总优先级相近的候选视场中, 优先
        # 选择离上次指向更近、且延续既有扫描方向的视场, 形成平滑扫天.
        self.prev_pointing: Optional[Tuple[float, float]] = None   # 上一成功曝光的指向
        self.sweep_altaz: Optional[Tuple[float, float]] = None     # 扫描方向 (alt/az 增量, 度)
        self.continuity_bonus_max = 60.0        # 空间连续性最大加分 (连续性子目标)
        self.continuity_scale_deg = 6.0         # 加分随角距衰减的尺度 (度)
        self.momentum_weight = 0.35             # 惯性: 奖励延续扫描方向的权重
        # 质量带: 仅当候选质量与最优相差不超过此带时, 才用连续性/惯性择优.
        # 相对带 (fraction) 与绝对带 (abs) 取较大者, 适配不同任务卡的计分尺度.
        # 实测 (alpha): 0.20 可把相邻指向中位跳变 17.6°->~13°, 大跳变次数下降,
        # 且 RA 条带均匀度改善; 带宽过大会牺牲覆盖, 故取较保守值.
        self.continuity_band_fraction = 0.20    # 相对最优质量的 20%
        self.continuity_band_abs = 1000.0       # 或绝对 1000 分
        self.recent_fields: List[Tuple[float, float, float]] = []  # (alt, az, seq) 近期指向
        self.recent_field_horizon = 6           # 回访去重的时间窗 (轮)
        # 必观测保障 (漏一个 -50, 是首要得分项): 未完成必观测集合, 以及"视场覆盖到
        # 未完成必观测目标"的硬加成. 该加成直接进入视场质量, 不参与连续性折中.
        self._required_unfinished: set = set()
        # 每个未完成必观测目标的视场加成: 取"支配性"数值, 保证含必观测的视场必胜
        self.required_field_bonus = REQUIRED_FIELD_DOMINANCE
        # 时序门控: 未完成必观测仅在"当前高度角 >= 中天高度角的该比例"时才优先锚定
        # (把低仰角窗口让给普通目标), 但重试目标不受限. 实测该门控在部分卡上会推迟
        # 必观测、反而增加漏失, 故默认关闭 (0.0); 保留参数便于按卡调参.
        self.required_defer_frac = 0.0
        # 必观测锚定使用的中心光纤 (靠近网格中心的若干根), 用于反推指向.
        side = self.grid_side
        mid_lo = (side - 1) // 2
        mid_hi = side // 2
        self._anchor_fibers = [r * side + c for r in (mid_lo, mid_hi) for c in (mid_lo, mid_hi)]

        # -- 时序感知调度 (time-aware scheduling) ----------------------------
        # 单个目标的得分随时间近似"抛物线"(过中天/transit 时最佳, 高度角最高、
        # 大气质量最小; 参见 Cao 2025 AJ 170,88). 因此对必观测目标应尽量安排在其
        # **最佳时段**(中天附近 + 暗夜), 而非"可见即观测". 同时记录尝试次数与已达成
        # 的完成因子, 对"尝试过但未完成"的目标进行**重试**并加大曝光.
        self.required_attempts: Dict[str, int] = {}     # 必观测目标已尝试次数
        self.attempt_count: Dict[str, int] = {}         # 全部目标已尝试次数 (重试封顶用)
        self._last_anchor: Dict[str, int] = {}          # 目标最近一次被必观测锚定的轮次
        # 全局"已达成完成因子上界"记忆 (所有目标): score/w = g·m 是 g 的上界; 用于
        # 判断目标是否已达标 (g>=threshold) 或需要重试 (上界仍 < threshold).
        self.achieved_g: Dict[str, float] = {}
        self._required_transit_alt: Dict[str, float] = {}  # 目标当日中天高度角 (缓存)
        self._required_transit_lst: Dict[str, float] = {}  # 目标中天时的 LST
        # 时序加成的权重 (叠加到必观测锚点的视场质量上, 使其在最佳时段更易胜出).
        self.time_aware_weight = 600.0

        # -- LLM 调用节流 (900s 预算保护, 任务书第 6.3 条成败项) --------------
        # decide_action(环节B) 不是每轮都调: 仅在关键点或周期性调用, 其余轮次用
        # 静态内核落成动作; 由性能预算与决策序号共同控制.
        self.decisions_seen = 0                 # 已处理的 decision 数
        self.llm_decide_calls = 0               # 环节B 实际调用次数
        self.last_llm_decide_seq = -10**9       # 上次环节B调用的 decision_sequence
        self.llm_decide_interval = 5            # 默认每隔 5 轮调一次 (pace 会覆盖)
        self.llm_decide_max_total = 120         # 一次运行环节B调用总次数上限
        self.llm_decide_max_fraction = 0.30     # 每轮平均最多消耗的 CPU 秒占比

        # 锚点搜索参数 (预算与质量的折中); 会按剩余 CPU 预算自适应缩小
        self.anchor_pool = 160          # 快速探测的锚点上限
        self.refine_anchors = 4         # 精细搜索的最密锚点数
        self._center_fiber = 5          # 快速探测使用的中心光纤 (靠中间)
        self.pace_level = 0             # 0=充裕 1=适中 2=紧张
        self.per_decision_cpu = float("inf")  # 每决策可用的 CPU 秒 (预算保护)
        self._empty_waits_in_night = 0  # 本夜连续"无目标"等待轮数 (用于跳夜判据)
        self._now_utc = None            # 当前决策时刻 (供曝光估算使用)

    # -- 初始化 ------------------------------------------------------------
    def _load_targets(self) -> None:
        """优先用协议 payload, 回退到 preplan 卡片数据."""
        loaded = 0
        if self.init.target_rows:
            for row in self.init.target_rows:
                t = Target.from_protocol_row(self.init.target_columns, row)
                if t.target_id:
                    self.targets.add(t)
                    loaded += 1
        if loaded == 0 and self.tool.card is not None:
            for pt in self.tool.card.targets:
                self.targets.add(Target.from_preplan(pt))
                loaded += 1
        self.log(f"planner: 载入 {loaded} 个目标 (协议={bool(self.init.target_rows)}, 卡片={self.tool.card.name if self.tool.card else None})")

    def _runtime_card_from_payload(self) -> preplan.CardData:
        """用 initialize payload 的仪器/计分参数构造"运行期卡片".

        正式赛的 card_id (A/B/C/D/A1…) 与仓库目录名 (cardA/cardB/…) 不一致,
        或该卡本身未随仓库下发 (A1-D1) 时, 用它替代缺失的卡片文件, 使
        ``build_fiber_grid`` / ``suggest_exposure_seconds`` 仍使用**真实**的
        光纤布局与计分基准 (fiber_area_deg2 / grid_side / flux_zero_point /
        exposure_zero_point_seconds), 而不是硬编码默认值。

        目标仍以协议 payload 为准 (这里 ``targets`` 留空, 由 ``_load_targets``
        从 payload 填入), 因此不会改变目标集合。
        """
        instrument = self.init.instrument if isinstance(self.init.instrument, dict) else {}
        exposure = instrument.get("exposure") or {}
        try:
            area = float(instrument.get("fiber_area_deg2", 0.4) or 0.4)
        except (TypeError, ValueError):
            area = 0.4
        try:
            gap = float(instrument.get("gap_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            gap = 0.0
        try:
            min_exp = int(exposure.get("min_duration_seconds", self.min_exposure))
        except (TypeError, ValueError):
            min_exp = self.min_exposure
        try:
            max_exp = int(exposure.get("max_duration_seconds", self.max_exposure))
        except (TypeError, ValueError):
            max_exp = self.max_exposure
        fiber_config = {
            "schema_version": "v4-fiber-map-v1",
            "site": {"latitude_deg": self.lat, "longitude_deg": self.lon},
            "field": {"n_fibers": self.n_fibers, "fiber_area_deg2": area, "gap_deg": gap},
            "exposure": {"min_duration_seconds": min_exp, "max_duration_seconds": max_exp},
        }
        score_config = self.init.scoring if isinstance(self.init.scoring, dict) else {}
        name = self.init.card_id or "runtime"
        self.log(f"planner: 无卡片文件, 用 payload 构造运行期卡片 {name!r} "
                 f"(光纤 {self.n_fibers} 根, 单根 {area} deg^2, 网格 {self.grid_side}x{self.grid_side})")
        return preplan.CardData(
            name=name,
            root=preplan.PROJECT_ROOT,
            targets=[],
            fiber_config=fiber_config,
            score_config=score_config,
            footprint_vertices={},
        )

    def _build_grid(self) -> preplan.FiberGrid:
        card = self.tool.card
        if card is not None:
            return preplan.build_fiber_grid(card)
        # 理论上不会走到这里 (__init__ 已用 payload 兜底); 兜底时也按 payload 计算,
        # 不硬编码 0.4, 否则非 4x4 / 非 0.4 deg² 的卡几何会算错。
        instrument = self.init.instrument if isinstance(self.init.instrument, dict) else {}
        try:
            area = float(instrument.get("fiber_area_deg2", 0.4) or 0.4)
        except (TypeError, ValueError):
            area = 0.4
        try:
            gap = float(instrument.get("gap_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            gap = 0.0
        return preplan.FiberGrid(
            side=self.grid_side,
            n_fibers=self.n_fibers,
            fiber_side_deg=math.sqrt(max(area, 1e-9)),
            gap_deg=gap,
        )

    # -- 决策入口 ----------------------------------------------------------
    def decide(self, req) -> Action:
        """对一次 decision_request 产出动作.

        流程:
            1. 计算当前夜 / 可见性 / 异常;
            2. 关键决策点调 LLM:
               (a) 新一轮观测窗口 -> plan_night (环节A);
               (b) 结果明显异常 -> diagnose_abnormal (环节C);
               (c) 行动决策 -> decide_action (环节B), 但仅在不平凡时;
            3. 由本内核把 LLM 意向或启发式结果落成合法动作.
        """
        now = req.now_utc
        if now is None:
            return Action(type="wait", duration_seconds=900, reason="无 now_utc", decision_source="fallback")
        self._now_utc = now

        # -- 性能计数: 两次 decide 之间的 CPU 时间 ≈ 上一轮的成本 -------------
        cpu = time.process_time()
        if self._cpu_mark is not None:
            self.decision_cpu_total += cpu - self._cpu_mark
            self.decision_cpu_count += 1
        self._cpu_mark = cpu
        self.perf_fill_calls = 0
        self._snapshot = None       # 每轮重新取快照 (同轮内多次调用会命中缓存)

        self._update_result(req)
        # 预算档位提前计算: 供后续等待动作做预算自适应 (跳过空转 slot).
        self._update_pace(req)

        # 白天: 等到下一夜
        night = self._current_night(now)
        if night is None:
            nxt = self._next_night_start(now)
            if nxt is None:
                return Action(type="finish", reason="无剩余观测夜", decision_source="static")
            return Action(type="wait", until_utc=_format_utc(nxt), reason="白天: 等到下一夜", decision_source="static")
        night_index, night_start, night_end = night

        # 站点因雨/暴风关闭
        if self._site_closed(req):
            return self._wait_action(now, night_start, night_end, "简报: 全天天雨/暴风")

        # 夜晚即将结束
        if (night_end - now).total_seconds() < self.min_exposure:
            nxt = self._next_night_start(now)
            if nxt is None:
                return Action(type="finish", reason="巡天结束", decision_source="static")
            return Action(type="wait", until_utc=_format_utc(nxt), reason="本夜将尽", decision_source="static")

        is_new_night = night_index != self.last_night_index
        self.last_night_index = night_index
        if is_new_night:
            self._empty_waits_in_night = 0

        self.decisions_seen += 1
        # 每轮只算一次"可见目标 + 地平坐标"(供 _build_state 与 _plan_observe 复用)
        lst = local_sidereal_deg(now, self.lon)
        visible, altaz = self._snapshot_sky(lst)
        # 每轮只算一次有效优先级与排序 (含所有加成)
        ranked, priorities, request_targets = self._prepare_round(visible, req.active_requests)
        state = self._build_state(req, night_index, night_start, night_end, is_new_night,
                                  lst=lst, visible=visible,
                                  candidates=[t.target_id for t in ranked[:96]])
        if self.decision_cpu_count and self.decision_cpu_count % 200 == 0:
            avg_ms = self.decision_cpu_total / self.decision_cpu_count * 1000.0
            self.log(f"planner: CPU 平均 {avg_ms:.0f} ms/决策 (已 {self.decision_cpu_count} 轮); "
                     f"本轮可见 {len(visible)} 目标, 视场中心评估 {self.perf_fill_calls} 次, pace={self.pace_level}")

        # 环节A·任务规划: 仅在新夜 (关键决策点) 调用
        if is_new_night or self.night_plan is None:
            if self.llm is not None:
                self.night_plan = self.llm.plan_night(state)
            else:
                self.night_plan = NightPlan(targets=state.candidate_targets, source="static")
            self.duration_scale = self.night_plan.duration_scale
            self.extra_avoid = list(self.night_plan.avoid_directions)
            self.log(f"planner: 本夜规划 source={self.night_plan.source} program={self.night_plan.program} "
                     f"strategy={self.night_plan.strategy!r}")

        # 环节C·计划自适应: 结果异常时
        if state.abnormal and self.llm is not None:
            diag = self.llm.diagnose_abnormal(state)
            if diag:
                try:
                    self.duration_scale = max(0.7, min(1.4, float(diag.get("duration_scale", self.duration_scale))))
                except (TypeError, ValueError):
                    pass
                extra = [str(d).upper() for d in (diag.get("avoid_directions") or [])]
                self.extra_avoid = list({*self.extra_avoid, *extra})
                self.report_fault_hint = bool(diag.get("report_fault"))

        # 故障报告 (确定性判据 + LLM 深诊提示)
        report = self._maybe_report(state)
        if report is not None:
            return report

        # 环节B·行动决策: LLM 给高层意向. **不是每轮都调** (900s 预算成败项):
        # 仅在新夜/新请求/异常等关键点, 或按预算自适应的周期上调用; 其余轮次由
        # 静态内核落成 observe, 保证巡天能在 CPU 预算内完成.
        llm_action = None
        if self.llm is not None and self._should_call_llm_decide(req, is_new_night):
            llm_action = self.llm.decide_action(state, self.night_plan)
            self.llm_decide_calls += 1
            self.last_llm_decide_seq = req.decision_sequence
        if llm_action is not None and llm_action.type == "report":
            if self._can_report():
                return self._make_report("LLM 高层决策")
        if llm_action is not None and llm_action.type == "finish":
            # 关键保护: 绝不因 LLM 一句话就提前结束整个巡天. 只有"确实没有剩余观测
            # 时间"时才接受 finish; 否则忽略之, 继续用静态内核观测/等待.
            # (历史教训: card C 首轮 LLM 幻觉输出 {"action":"finish"} 直接终止了整场.)
            if self._survey_effectively_over(now, night_end):
                return Action(type="finish", reason=llm_action.reason, decision_source="llm")
            self.log("planner: 忽略 LLM 的提前 finish (仍有剩余观测时间)")
        if llm_action is not None and llm_action.type == "wait":
            return self._wait_action(now, night_start, night_end, llm_action.reason)

        # 落成 observe (LLM 意向或纯静态)
        action = self._plan_observe(now, night_end, night_index, state,
                                    lst=lst, visible=visible, altaz=altaz,
                                    ranked=ranked, priorities=priorities,
                                    request_targets=request_targets)
        if action is None:
            return self._wait_action(now, night_start, night_end, "暂无可观测目标")
        return action

    def _should_call_llm_decide(self, req, is_new_night: bool) -> bool:
        """判断本轮是否调用环节B (行动决策) 的 LLM.

        预算保护的核心: decide_action 单次约数秒, 若每轮都调会在 900s CPU 预算
        与 30 分钟墙钟上限内耗光, 导致最后阶段无预算. 这里只在**关键决策点**或
        **周期性**调用, 并按剩余预算自适应放宽/收紧周期.

        关键点: 新夜第一轮 / 收到新的限时请求 / 结果异常 / 距上次调用超过周期.
        硬约束: 调用总次数与每轮平均 CPU 占比均有上限.
        """
        if self.llm is None:
            return False
        if self.llm_decide_calls >= self.llm_decide_max_total:
            return False
        seq = req.decision_sequence
        # 关键点 (忽略周期, 但避免连续两轮重复调用)
        keypoint = is_new_night or self._has_new_observation_request(req) or self._state_abnormal(req)
        if keypoint:
            return (seq - self.last_llm_decide_seq) >= 1
        # 周期性调用: 由性能预算决定间隔
        return (seq - self.last_llm_decide_seq) >= max(1, self.llm_decide_interval)

    @staticmethod
    def _has_new_observation_request(req) -> bool:
        try:
            return any(str(m.get("record_type") or m.get("message_type") or "") == "observation_request"
                       for m in (req.new_messages or []))
        except Exception:  # noqa: BLE001
            return False

    def _state_abnormal(self, req) -> bool:
        lr = req.last_result
        if not (lr and lr.get("action") == "observe"):
            return False
        assigned = int(lr.get("assigned_count", 0) or 0)
        hit = int(lr.get("hit_count", 0) or 0)
        return assigned >= 8 and hit == 0

    # -- 状态与统计 --------------------------------------------------------
    def _update_result(self, req) -> None:
        lr = req.last_result
        if lr and lr.get("action") == "observe":
            assigned = int(lr.get("assigned_count", 0) or 0)
            hit = int(lr.get("hit_count", 0) or 0)
            self.assigned_total += assigned
            self.hits_total += hit
            # 记忆命中 (已得分) 目标; 同时记录本轮的 assignments 为"已尝试".
            hits = lr.get("hits") or []
            for h in hits:
                tid = h.get("target_id") if isinstance(h, dict) else None
                if tid:
                    tid = str(tid)
                    self.observed_ids.add(tid)
                    # 记录该目标已达成的"完成因子上界"估计: 引擎返回
                    # hits[].score = c = w·g·m (m∈[1.0,1.2]), 故 score/w = g·m >= g,
                    # 即 score/w 是 g 的上界. 若 score/w < 阈值, 则**必定** g < 阈值
                    # (一定未达标), 需要重试; 对所有目标都跟踪 (不只必观测).
                    t = self.targets.index.get(tid)
                    if t is not None:
                        try:
                            score = float(h.get("score", 0.0))
                        except (TypeError, ValueError):
                            score = 0.0
                        g_upper = max(0.0, score / max(1e-6, t.science_weight))
                        self.achieved_g[tid] = max(self.achieved_g.get(tid, 0.0), g_upper)
            for tid in (lr.get("assigned_target_ids") or []):
                self.attempted_ids.add(str(tid))
        if lr and lr.get("action") == "report":
            correct = bool(lr.get("correct"))
            self.log(f"planner: report 结果 correct={correct} delta={lr.get('score_delta')}")
            self.consecutive_reports = 1 if correct else self.consecutive_reports

    def _update_pace(self, req) -> None:
        """按剩余 CPU 预算与剩余决策量选择计算档位 (预算保护).

        以 ``remaining_real_cpu_seconds`` 为准 (回退 remaining_seconds), 估算剩余
        decision 数 (夜剩余时长 / 700s), 得到每决策可用的 CPU 秒, 据此收紧锚点搜索.
        """
        cpu_left = req.remaining_real_cpu_seconds()
        if cpu_left == float("inf"):
            self.pace_level = 0
            return
        # 估算剩余 decision 数
        decisions_left = 1.0
        if req.now_utc is not None:
            for start, end in self._nights:
                if end and end > req.now_utc:
                    eff_start = max(start, req.now_utc) if start else req.now_utc
                    decisions_left += max(0.0, (end - eff_start).total_seconds()) / 700.0
        per_decision = cpu_left / max(1.0, decisions_left)
        self.per_decision_cpu = per_decision
        # 阈值: 每决策可用 CPU 秒
        level = 0 if per_decision > 0.35 else 1 if per_decision > 0.10 else 2
        if level != self.pace_level:
            self.log(f"planner: pace level {level} (每决策可用 {per_decision * 1000:.0f} ms CPU, 剩余 {cpu_left:.0f}s)")
        self.pace_level = level

        # 环节B 的 LLM 调用周期 (预算保护): 估算一次 LLM 调用约 5s CPU,
        # 只允许消耗每轮可用 CPU 预算的一小部分, 从而得出最小调用间隔.
        # 预算充裕时才频繁调用 (但仍非每轮), 紧张时拉长间隔; 上限保守为 20 轮,
        # 保证仍能定期获得 LLM 的高层决策 (同时关键点始终触发).
        approx_llm_cpu = 5.0
        if per_decision <= 1e-6:
            self.llm_decide_interval = 20  # 几乎没有预算: 拉到最大间隔
        else:
            budget_for_llm = per_decision * self.llm_decide_max_fraction
            interval = int(math.ceil(approx_llm_cpu / max(budget_for_llm, 1e-6)))
            self.llm_decide_interval = max(1, min(interval, 20))

    def _current_night(self, now):
        for idx, (start, end) in enumerate(self._nights):
            if start and end and start <= now < end:
                return idx, start, end
        return None

    def _next_night_start(self, now):
        best = None
        for start, _end in self._nights:
            if start and start > now and (best is None or start < best):
                best = start
        return best

    def _to_next_slot(self, now, night_start) -> int:
        into = (now - night_start).total_seconds() % self.slot_seconds
        return int(max(60, min(3600, self.slot_seconds - into if into else self.slot_seconds)))

    def _wait_action(self, now, night_start, night_end, reason: str) -> Action:
        """预算自适应的等待动作 (避免把 CPU 预算耗在空转的逐 slot wait 上).

        在"当前无可见目标 / 站点关闭"等**本就不观测**的情形下等待. 当每决策可用
        CPU 预算紧张时, 一次等待跨多个 slot (甚至整夜), 把预算省给真正能观测的时刻;
        预算充裕时仍逐 slot 走, 不错过任何机会.
        """
        cpu = self.per_decision_cpu
        self._empty_waits_in_night += 1
        if cpu == float("inf") or cpu >= 0.20:
            mult = 1
        elif cpu >= 0.10:
            mult = 2
        elif cpu >= 0.05:
            mult = 4
        else:
            mult = 8
        # 本夜剩余 slot 数
        secs_left = max(0.0, (night_end - now).total_seconds())
        slots_available = max(1, int(secs_left // self.slot_seconds))
        slots = max(1, min(slots_available, mult))
        # 若预算很紧、本夜剩余时间还很长、且**本夜已连续多轮无目标可观测**, 才跳到
        # 下一夜 (避免刚入夜、目标尚未升起就误跳整夜).
        if (cpu < 0.05 and secs_left > self.slot_seconds * 12
                and self._empty_waits_in_night >= 3):
            nxt = self._next_night_start(now)
            if nxt is not None:
                return Action(type="wait", until_utc=_format_utc(nxt),
                              reason=f"{reason} (预算紧张, 跳到下一夜)", decision_source="static")
        duration = int(max(60, min(3600, slots * self.slot_seconds)))
        return Action(type="wait", duration_seconds=duration, reason=reason, decision_source="static")

    def _survey_effectively_over(self, now, night_end) -> bool:
        """巡天是否确实已无可观测时间 (用于决定是否接受 LLM 的 finish).

        仅当当前夜已结束**且**没有未来的观测夜时才算结束. 在夜内一律不认为结束
        (防止 LLM 幻觉提前终止整场巡天).
        """
        if now < night_end:
            return False
        return self._next_night_start(now) is None

    def _site_closed(self, req) -> bool:
        """当前是否因雨/暴风**全天关闭** (只依据当前简报).

        重要: 只用 ``latest_bulletin`` (逐 slot 下发, 代表"此刻") 判定是否关闭。
        ``latest_forecast`` 是**一周展望**, 其 notice 带 ``nights[]`` (例如
        "12-04 那一夜可能下雨"), 而且会持续数日; 若拿它来关闭站点, agent 会在
        整周内一直等待、一枪不发 (公开测试卡 PT 实测: 253 轮全 wait、0 观测)。

        预报仍会通过 :meth:`_all_notices` 交给规划/LLM (见 ``_build_state``),
        只是**不再触发"立刻关闭"**。
        """
        bulletin = req.latest_bulletin
        if not isinstance(bulletin, dict):
            return False
        for n in (bulletin.get("notices") or []):
            if not isinstance(n, dict):
                continue
            if n.get("event_kind") in ("rain", "storm") and n.get("direction") == "ALL":
                return True
        return False

    @staticmethod
    def _all_notices(req) -> List[Dict[str, Any]]:
        """汇总简报与预报的 notices (供规划/LLM 参考, **不**用于判定"此刻关闭").

        预报 notice 常带 ``nights[]`` 与覆盖时段, 描述的是未来一周的展望, 不能
        等价于"现在关闭"; 关闭判定请用 :meth:`_site_closed` (仅看简报)。
        """
        notices: List[Dict[str, Any]] = []
        for src in (req.latest_bulletin, req.latest_forecast):
            if isinstance(src, dict):
                notices.extend(src.get("notices") or [])
        return notices

    def _can_report(self) -> bool:
        return self.consecutive_reports < self.init.max_consecutive_reports

    def _maybe_report(self, state: DecisionState) -> Optional[Action]:
        """启发式故障判据 + LLM 提示.

        只有当 (a) LLM 明确建议 report, 或 (b) 近期命中率极低且未报过太少次 时才报告.
        误报会扣分, 故保守: 默认不报, 除非证据充分.
        """
        if not self._can_report():
            return None
        if self.reports_made >= 2:
            return None
        rate = state.last_hit_rate
        strong = rate is not None and rate <= 0.15 and self.assigned_total >= 48
        if self.report_fault_hint or strong:
            self.report_fault_hint = False
            return self._make_report(f"命中率异常低 ({rate})")
        return None

    def _make_report(self, reason: str) -> Action:
        self.reports_made += 1
        self.consecutive_reports += 1
        source = "llm" if "LLM" in reason else "static"
        self.log(f"planner: 提交 report ({reason})")
        return Action(type="report", reason=reason, decision_source=source)

    # -- 构建 DecisionState ------------------------------------------------
    def _build_state(self, req, night_index, night_start, night_end, is_new_night,
                     lst: Optional[float] = None,
                     visible: Optional[Sequence[Target]] = None,
                     candidates: Optional[List[str]] = None) -> DecisionState:
        now = req.now_utc
        if lst is None:
            lst = local_sidereal_deg(now, self.lon)
        if visible is None:
            visible, _ = self._snapshot_sky(lst)
        # 候选: 由 _prepare_round 的排序结果直接截取 (避免重复排序); 缺省回退到现算。
        if candidates is None:
            candidates = self._visible_candidates(lst, now, visible=visible)
        required_remaining = sum(1 for t in visible if t.required)
        hit_rate = (self.hits_total / self.assigned_total) if self.assigned_total > 0 else None

        # 异常判定: 上次观测有指派但命中率很低
        abnormal = False
        abnormal_reason = ""
        lr = req.last_result
        if lr and lr.get("action") == "observe":
            assigned = int(lr.get("assigned_count", 0) or 0)
            hit = int(lr.get("hit_count", 0) or 0)
            if assigned >= 8 and hit == 0:
                abnormal = True
                abnormal_reason = f"上轮指派 {assigned} 命中 0"
            elif assigned >= 8 and hit / max(1, assigned) < 0.25:
                abnormal = True
                abnormal_reason = f"上轮命中率 {hit}/{assigned}"

        # 新收到的限时请求 (关键决策点)
        received_request = any(m.get("record_type") == "observation_request" for m in req.new_messages)

        return DecisionState(
            decision_sequence=req.decision_sequence,
            now_utc=now,
            survey_end_utc=req.survey_end_utc,
            remaining_seconds=req.remaining_seconds(),
            remaining_real_cpu_seconds=req.remaining_real_cpu_seconds(),
            wall_remaining_seconds=req.wall_remaining_seconds(),
            night_index=night_index,
            night_start_utc=night_start,
            night_end_utc=night_end,
            is_new_night=is_new_night,
            site_closed=self._site_closed(req),
            notices=self._all_notices(req),
            observation_requests=req.active_requests,
            abnormal=abnormal,
            abnormal_reason=abnormal_reason,
            target_count=len(self.targets.targets),
            required_count=sum(1 for t in self.targets.targets if t.required),
            required_remaining=required_remaining,
            candidate_targets=candidates,
            last_hit_rate=hit_rate,
        )

    def _prepare_round(self, visible: Sequence[Target], requests):
        """一轮只算一次: 有效优先级 + 排序 + 未完成必观测集合.

        原实现同一轮里会 (1) 在 ``_build_state`` 对可见集排序取前 96; (2) 在
        ``_plan_observe`` 再排一次; (3) 复制一份 3~5 万条的全局优先级字典; (4) 每轮
        重建 ``required_ids`` 集合。这里统一成一次 O(可见数) 的计算。

        Returns:
            ``(ranked, priorities, request_targets)`` —— ``ranked`` 为按有效优先级
            降序的可见目标; ``request_targets`` 为本轮获得请求加成的目标集合。
        """
        base = self._all_priorities()
        # 尝试上限: 反复失败的目标本轮退出选择, 把预算让给从未尝试过的目标。
        # (只有"重试陷阱"场景才会触发: 正常情况下极少有目标被拍这么多次)
        required_ids = self._required_ids
        attempt_count = self.attempt_count
        visible = [t for t in visible
                   if attempt_count.get(t.target_id, 0)
                   < (MAX_ATTEMPTS_REQUIRED if t.target_id in required_ids else MAX_ATTEMPTS_NORMAL)]
        priorities: Dict[str, float] = {t.target_id: base.get(t.target_id, 0.0) for t in visible}

        # 本夜规划目标的加成 (LLM/静态策略的落点)
        if self.night_plan is not None:
            for tid in self.night_plan.targets:
                if tid in priorities:
                    priorities[tid] += 500.0

        # 观测记忆: 已达标 -> 强惩罚 (不再重复); 未达标 -> 重试加成;
        # 试过但没命中且非必观测 -> 轻微回避。按"可见集"迭代, 成本与可见数同阶,
        # 不随巡天推进而无限增长 (旧实现遍历的是全局 observed/attempted 集合)。
        threshold = self._required_threshold
        observed = self.observed_ids
        attempted = self.attempted_ids
        required_ids = self._required_ids
        for t in visible:
            tid = t.target_id
            if tid in observed:
                g_upper = self.achieved_g.get(tid)
                if g_upper is not None and g_upper >= threshold:
                    priorities[tid] -= 5000.0
                else:
                    # 未达标: 只在前 RETRY_BONUS_MAX_ATTEMPTS 次给重试加成,
                    # 之后转为抑制 —— 否则"失败目标堆"会永久压过新鲜目标。
                    if self.attempt_count.get(tid, 0) <= RETRY_BONUS_MAX_ATTEMPTS:
                        priorities[tid] += RETRY_BONUS
                    else:
                        priorities[tid] -= STALE_PENALTY
            elif tid in attempted:
                if tid not in required_ids:
                    priorities[tid] -= 100.0
            else:
                priorities[tid] += FRESH_BONUS      # 从未尝试过: 覆盖优先

        # 限时观测请求目标加成
        request_targets = self._boost_request_targets(requests, priorities, self._now_utc)

        # 未完成必观测集合 (供视场加成与必观测锚定使用)
        unfinished = set()
        for tid in required_ids:
            g_upper = self.achieved_g.get(tid)
            if g_upper is None:
                if tid not in observed:
                    unfinished.add(tid)
            elif g_upper < threshold:
                unfinished.add(tid)
        self._required_unfinished = unfinished

        ranked = sorted(visible, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True)
        return ranked, priorities, request_targets

    def _boost_request_targets(self, requests, priorities: Dict[str, float],
                               now: datetime) -> set:
        """给"未完成且未过期"的限时观测请求目标加优先级, 返回被加成的目标集合.

        常规 +800; 距 deadline 不足 1 夜 +1200。已完成的目标跳过, 已过截止的
        请求整条跳过 (规则: 只有发布到截止之间的曝光才计入)。被加成的目标还会
        在 :meth:`_choose_duration` 里按"完成因子 0.5"给足曝光 —— 奖励按完成
        因子判定且不含程序加成, 光靠排序加成拿不到分。
        """
        boosted: set = set()
        for r in (requests or []):
            if not isinstance(r, dict):
                continue
            tids = [str(t) for t in (r.get("target_ids") or [])]
            if not tids:
                continue
            done = {str(t) for t in (r.get("completed_target_ids") or [])}
            remaining = r.get("remaining_count")
            try:
                remaining = int(remaining) if remaining is not None else None
            except (TypeError, ValueError):
                remaining = None
            if remaining is None:
                try:
                    remaining = max(0, int(r.get("minimum_completed") or 0) - len(done))
                except (TypeError, ValueError):
                    remaining = 0
            if remaining <= 0:
                continue                       # 已达标: 不必再抢
            deadline = _parse_utc(r.get("deadline_utc"))
            if deadline is not None and now is not None and deadline <= now:
                continue                       # 已过截止: 再观测也不计入
            bonus = REQUEST_BONUS
            if (deadline is not None and now is not None
                    and (deadline - now).total_seconds() <= 86400.0):
                bonus = REQUEST_BONUS_URGENT
            for tid in tids:
                if tid in done or tid not in priorities:
                    continue                    # 不可见/已完成: 跳过
                priorities[tid] += bonus
                boosted.add(tid)
        if boosted:
            self.log(f"planner: 限时请求目标加成 {len(boosted)} 个")
        return boosted

    def _visible(self, t: Target, lst: float) -> bool:
        """单个目标是否可见 (阈值 = minimum_altitude + 安全余量).

        走 :class:`skyindex.TargetIndex` 的时角判据 (与 alt 判据严格等价, 无三角函数)。
        """
        i = self._sky.pos.get(t.target_id)
        if i is None:  # 未进索引 (异常情况): 回退到直接计算
            alt, _ = radec_to_altaz(t.ra_deg, t.dec_deg, lst, self.lat)
            return alt >= self.min_alt + ALT_MARGIN_DEG
        return self._sky.is_visible_index(i, lst)

    def _snapshot_sky(self, lst: float):
        """一轮只算一次的"可见目标 + 地平坐标"快照.

        原本同一轮里 ``_build_state``(候选/必观测统计) 与 ``_plan_observe`` 会各自
        把全部目标扫一遍 (卡 A 实测每轮约 6 万次可见性判断)。这里统一成一次,
        并顺手得到 alt/az 供光纤填充复用。返回 ``(visible_targets, altaz_map)``。
        """
        cached = self._snapshot
        if cached is not None and cached[0] == lst:
            return cached[1], cached[2]
        idx = self._sky.visible_indices(lst)
        targets = self.targets.targets
        visible = [targets[i] for i in idx]
        altaz = self._sky.altaz_map(idx, lst)
        self._snapshot = (lst, visible, altaz)
        self.perf_visible_n = len(visible)
        return visible, altaz

    def _all_priorities(self) -> Dict[str, float]:
        """全体目标的优先级 (只算一次并缓存; 与夜/时间无关).

        原实现的缓存在调用方传入"可见子集"时永不命中, 于是每轮重算两次
        (卡 A 实测 ~76 ms/轮)。这里改为始终基于**全体目标**计算一次 ——
        均匀度按全量归一化 (有意变更), 且优先级跨轮稳定。
        """
        if self._priorities_all is None:
            card = self.tool._temp_card(self.targets.targets)
            self._priorities_all = preplan.compute_priorities(card)
        return self._priorities_all

    def _visible_candidates(self, lst: float, now: datetime, limit: int = 96,
                            visible: Optional[Sequence[Target]] = None) -> List[str]:
        """可见目标按优先级排序, 返回前 limit 个 ID (可复用已算好的可见集)."""
        if visible is None:
            visible, _ = self._snapshot_sky(lst)
        if not visible:
            return []
        priorities = self._all_priorities()
        ranked = sorted(visible, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True)
        return [t.target_id for t in ranked[:limit]]

    def _priorities_for(self, targets: Sequence[Target]) -> Dict[str, float]:
        """按给定目标集计算优先级 (保留给需要"子集口径"的调用方).

        决策主路径请用 :meth:`_all_priorities` (全量缓存, 每轮零成本)。
        """
        if len(targets) == len(self.targets.targets):
            return self._all_priorities()
        card = self.tool._temp_card(targets)
        return preplan.compute_priorities(card)

    # -- observe 规划 ------------------------------------------------------
    def _plan_observe(self, now, night_end, night_index, state: DecisionState,
                      lst: Optional[float] = None,
                      visible: Optional[Sequence[Target]] = None,
                      altaz: Optional[Dict[str, Tuple[float, float]]] = None,
                      ranked: Optional[List[Target]] = None,
                      priorities: Optional[Dict[str, float]] = None,
                      request_targets: Optional[set] = None) -> Optional[Action]:
        """把候选目标集落成一个合法的 observe 动作.

        方法 (复用 preplan 的优先级与 FiberGrid 方格模型, 但不使用其多次指向的
        `assign_fibers` 结果, 因为那样得到的光纤索引不属于同一曝光):

            1. 用 preplan.compute_priorities 对可见目标打分排序;
            2. 取若干高优先级"锚点"目标, 逐一作为视场中心的候选;
            3. 对每个候选中心, 用 FiberGrid.classify 把邻近目标映射到 16 根光纤
               (每根至多一个目标, 且目标需高于最低高度角);
            4. 选总优先级最高的中心作为最终指向, 并选择曝光时长与 program.
        """
        if lst is None:
            lst = local_sidereal_deg(now, self.lon)
        seconds_left = (night_end - now).total_seconds()
        if seconds_left < self.min_exposure:
            return None

        # 填充用: 全部可见目标 (尽可能填满光纤)。可见集与 alt/az 由本轮快照一次算好,
        # 不再在这里对全部目标重扫一遍 (原实现每轮多扫 ~3 万次)。
        if visible is None or altaz is None:
            visible, altaz = self._snapshot_sky(lst)
        if not visible:
            return None
        # 有效优先级/排序/请求加成/未完成必观测集合: 由 _prepare_round 每轮只算一次
        # (见 decide)。这里缺省回退, 以便单独调用本方法时仍可用。
        if ranked is None or priorities is None:
            ranked, priorities, request_targets = self._prepare_round(
                visible, state.observation_requests)
        required_ids = self._required_ids

        # 空间分箱索引 (按 (alt,az) 分箱): 让 _fill_pointing 只遍历视场附近的少量目标,
        # 在大卡 (数万目标 / 100 光纤) 上把每轮决策从 O(N) 降到 O(局部).
        _bin = max(1.0, self._fiber_grid.fov_side_deg)
        index = self._build_spatial_index(ranked, altaz, _bin)
        order = {t.target_id: i for i, t in enumerate(ranked)}
        # cache: 同一轮内按 (alt,az) 分箱缓存"邻域目标", 供几十~上百次候选评估复用
        spatial = {"bin_deg": _bin, "index": index, "order": order, "cache": {}}

        best = None  # 兼容旧变量: 最终为 (quality, c_alt, c_az, assignments)
        seen_centers = set()
        candidates: List[Tuple[float, float, float, Dict[str, str]]] = []  # (quality, alt, az, assignments)
        # 锚点搜索: 参照官方示例的 anchor-search 精神.
        #   1) 先用少量中心光纤快速探测每个锚点的邻域密度, 选最"密"的锚点;
        #   2) 对最优锚点用全部 16 根光纤的中心偏移做精细搜索.
        # 锚点池: 既取最高优先级, 也按步长在全体可见目标中抽样 (兼顾优先级与密度).
        # 预算紧张时整体收缩 (预算保护).
        pool_size = min(PROBE_MAX, {0: self.anchor_pool, 1: 60, 2: 20}[self.pace_level])
        refine = {0: self.refine_anchors, 1: 2, 2: 1}[self.pace_level]
        top = ranked[: pool_size // 2]
        stride = max(1, len(ranked) // max(1, pool_size // 2))
        spread = ranked[::stride][: pool_size // 2]
        seen_ids = set()
        sample = []
        for t in (*top, *spread):
            if t.target_id in seen_ids:
                continue
            seen_ids.add(t.target_id)
            sample.append(t)
        dense_centers = []
        for anchor in sample:
            a_alt, a_az = altaz[anchor.target_id]
            if a_alt < self.min_alt + ALT_MARGIN_DEG:
                continue
            row, col = divmod(self._center_fiber, self.grid_side)
            d_north, d_east = self._fiber_grid.cell_center(row, col)
            c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
            if not (self.min_alt + 1.0 <= c_alt <= 89.0):
                continue
            assignments, total = self._fill_pointing(ranked, priorities, altaz, c_alt, c_az, spatial)
            if assignments:
                dense_centers.append((total + 50.0 * len(assignments), anchor, d_north, d_east))
        if not dense_centers:
            return None
        dense_centers.sort(key=lambda x: -x[0])

        def collect_center(c_alt: float, c_az: float) -> None:
            """评估一个候选中心, 收集其质量分 (供后续带内连续性择优)."""
            if len(candidates) >= MAX_CENTERS_PER_DECISION:
                return                      # 每轮候选中心硬上限 (成本与卡规模解耦)
            if not (self.min_alt + 1.0 <= c_alt <= 89.0):
                return
            key = (round(c_alt, 1), round(c_az, 1))
            if key in seen_centers:
                return
            seen_centers.add(key)
            assignments, total = self._fill_pointing(ranked, priorities, altaz, c_alt, c_az, spatial)
            if not assignments:
                return
            # 必观测完成加成: 该视场能覆盖的"未完成必观测目标"数量, 直接计入质量,
            # 确保必观测 (漏一个 -50) 优先于普通目标的边际增益. 这是**硬优先级**,
            # 不参与后续"质量带"的连续性折中.
            n_req = sum(1 for tid in assignments.values() if tid in self._required_unfinished)
            quality = total + 50.0 * len(assignments) + self.required_field_bonus * n_req
            quality -= self._repeat_pointing_penalty(c_alt, c_az)
            candidates.append((quality, c_alt, c_az, assignments))

        # 精细搜索: 对最密的若干锚点, 尝试光纤中心偏移. 大网格 (如 10x10=100 光纤)
        # 时对光纤采样, 避免 refine x n_fibers 的候选爆炸 (预算保护).
        max_fine_fibers = FINE_FIBER_MAX
        if self.n_fibers <= max_fine_fibers:
            fine_fibers = range(self.n_fibers)
        else:
            fstep = max(1, self.n_fibers // max_fine_fibers)
            fine_fibers = list(range(0, self.n_fibers, fstep))[:max_fine_fibers]
        # 注意: collect_center 有全局硬上限, 且按调用顺序生效 —— 必须给每个阶段
        # 单独定额度, 否则 refine 会把预算吃光, 导致后续"连续性种子"被整段丢弃
        # (连续性机制会失效)。
        used_refine = 0
        for _, anchor, _, _ in dense_centers[:refine]:
            a_alt, a_az = altaz[anchor.target_id]
            for fiber_id in fine_fibers:
                if used_refine >= MAX_REFINE_CENTERS:
                    break
                used_refine += 1
                row, col = divmod(fiber_id, self.grid_side)
                d_north, d_east = self._fiber_grid.cell_center(row, col)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                collect_center(c_alt, c_az)

        # 必观测锚定: 对当前可见的**未完成必观测目标**, 以"使其落入某光纤"为目标
        # 生成候选中心, 保证每个可见的未完成必观测都有机会在本轮被安排.
        used_required = 0
        for c_alt, c_az in self._required_anchor_centers(altaz):
            if used_required >= MAX_REQUIRED_ANCHOR_CENTERS:
                break
            used_required += 1
            collect_center(c_alt, c_az)

        # 连续性种子: 在上次指向附近、以及沿扫描方向继续处布置候选中心,
        # 使"平滑扫天"的期望视场进入候选集 (否则会被锚点池过滤掉).
        used_cont = 0
        for c_alt, c_az in self._continuity_seed_centers(altaz):
            if used_cont >= MAX_CONTINUITY_CENTERS:
                break
            used_cont += 1
            collect_center(c_alt, c_az)

        if not candidates:
            return None
        # 收敛选择: 在"质量带"内用连续性/惯性择优 (尺度无关, 取代绝对加分).
        #   1) 取最高质量 q_max;
        #   2) 保留质量 >= q_max * (1 - band) - band_abs 的候选 (近似同优);
        #   3) 其中选择连续性得分最高者 (最靠近上次指向 / 延续扫描方向).
        best = max(candidates, key=lambda x: x[0])
        q_max = best[0]
        band = self.continuity_band_fraction
        band_abs = self.continuity_band_abs
        threshold = q_max - max(abs(q_max) * band, band_abs)
        band_pool = [c for c in candidates if c[0] >= threshold]
        if band_pool:
            best = max(band_pool, key=lambda c: self._continuity_bonus(c[1], c[2]))

        _, center_alt, center_az, assignments = best
        duration = self._choose_duration(assignments, seconds_left, altaz, lst, now,
                                         extra_must_complete=request_targets)
        # 程序选择: LLM 明确指定则用之; 纯静态时按月光+大气质量模型估计档位
        # (DARK×1.20 / BRIGHT×1.12 / BACKUP×1.06; 声明错档只 ×1.00).
        if self.night_plan is not None and self.night_plan.source == "llm" and self.night_plan.program in ("DARK", "BRIGHT", "BACKUP"):
            program = self.night_plan.program
        else:
            program = self._estimate_program(assignments, altaz, lst, now)
        action = Action(
            type="observe",
            pointing={"alt_deg": round(center_alt, 4), "az_deg": round(center_az, 4) % 360.0},
            assignments=assignments,
            exposure_seconds=duration,
            program=program,
            reason=f"{len(assignments)} 光纤, 程序 {program}",
            decision_source="llm" if (self.night_plan and self.night_plan.source == "llm") else "static",
        )
        center_ra, center_dec = altaz_to_radec(center_alt, center_az, lst, self.lat)
        action.ra_deg = center_ra
        action.dec_deg = center_dec
        # 记录指向与已尝试目标 (下次规划据此避免重复视场 / 重复曝光).
        self.last_pointing = (center_alt, center_az)
        # 更新扫描连续性状态: 记录上一指向、扫描方向 (惯性) 与近期视场.
        if self.prev_pointing is not None:
            p_alt, p_az = self.prev_pointing
            d_alt = center_alt - p_alt
            d_az = wrap180(center_az - p_az)
            # 指数滑动平均更新扫描方向, 抑制单轮噪声.
            if self.sweep_altaz is None:
                self.sweep_altaz = (d_alt, d_az)
            else:
                s_alt, s_az = self.sweep_altaz
                self.sweep_altaz = (
                    0.5 * s_alt + 0.5 * d_alt,
                    0.5 * s_az + 0.5 * d_az,
                )
        self.prev_pointing = (center_alt, center_az)
        self._empty_waits_in_night = 0   # 本轮成功观测: 清空"无目标"计数
        self.recent_fields.append((center_alt, center_az, self.decisions_seen))
        if len(self.recent_fields) > self.recent_field_horizon:
            self.recent_fields = self.recent_fields[-self.recent_field_horizon:]
        for tid in assignments.values():
            tid = str(tid)
            self.attempted_ids.add(tid)
            self.attempt_count[tid] = self.attempt_count.get(tid, 0) + 1
            # 时序感知: 记录必观测目标的尝试次数, 供重试优先级与曝光升级使用.
            if tid in required_ids:
                self.required_attempts[tid] = self.required_attempts.get(tid, 0) + 1
        self.log(f"planner: observe 指派 {len(assignments)} 根光纤, 曝光 {duration}s, 程序 {program}, "
                 f"指向 alt={center_alt:.2f} az={center_az:.2f}")
        return action

    def _continuity_bonus(self, center_alt: float, center_az: float) -> float:
        """"空间连续性 / 扫描惯性"目标函数 (收敛控制, 在质量带内择优使用).

        目的: 让相邻曝光的指向平滑过渡、形成连贯扫天, 而不是每轮在全局重新选址
        导致大幅跳变 (实测平均跳变 28°). 该函数**不再是无条件加到总分上的奖励**,
        而是在"质量带"(:attr:`continuity_band_fraction` / `_abs`) 内、于近似同优的
        候选之间作为**择优依据**, 因此不受计分尺度 (动辄上万) 影响:

            1. 邻近项: 距离上次指向越近, 值越高 (指数衰减, 尺度 continuity_scale);
            2. 惯性项: 若该视场延续既有扫描方向, 额外奖励 (形成有方向的扫描);
            3. 回访去重: 若与近期(<= horizon 轮)某指向几乎重合, 扣分, 避免来回振荡.
        """
        if self.prev_pointing is None:
            return 0.0
        p_alt, p_az = self.prev_pointing
        sep = angular_separation_altaz(center_alt, center_az, p_alt, p_az)

        bonus = self.continuity_bonus_max * math.exp(-sep / max(1e-6, self.continuity_scale_deg))

        # 惯性: 奖励延续扫描方向的候选 (与扫描方向夹角越小奖励越高).
        if self.sweep_altaz is not None:
            s_alt, s_az = self.sweep_altaz
            norm = math.hypot(s_alt, s_az)
            if norm > 1e-6 and sep > 1e-6:
                d_alt = center_alt - p_alt
                d_az = wrap180(center_az - p_az)
                vnorm = math.hypot(d_alt, d_az)
                if vnorm > 1e-6:
                    cosine = (d_alt * s_alt + d_az * s_az) / (norm * vnorm)
                    bonus += self.continuity_bonus_max * self.momentum_weight * max(0.0, cosine)

        # 回访去重: 若与近期视场几乎重合(半个视场内), 扣分, 避免在两地间振荡.
        half = self._fiber_grid.fov_side_deg / 2.0
        for f_alt, f_az, _seq in self.recent_fields:
            if angular_separation_altaz(center_alt, center_az, f_alt, f_az) < half:
                bonus -= self.continuity_bonus_max
                break
        return bonus

    def _transit_altitude(self, dec_deg: float) -> float:
        """目标中天时的最大高度角 = 90 - |lat - dec| (单位度)."""
        return 90.0 - abs(self.lat - dec_deg)

    def _time_quality(self, target: Target, alt_deg: float) -> float:
        """目标当前时刻相对"最佳时段"的质量因子, 取值 (0, 1].

        最佳时段 = 中天附近 (高度角最大, 大气质量最小). 用"当前高度角 / 中天高度角"
        近似接近中天的程度 (越接近 1 越好). 同时用当前高度角惩罚低仰角. 这是公开
        几何量, 不依赖隐藏天气; 用于把必观测目标尽量排在其最佳时段.
        """
        max_alt = self._transit_altitude(target.dec_deg)
        if max_alt <= 0.0:
            return 0.0
        # 归一化高度角: 目标越低 -> 越接近地平 -> 质量越低.
        # 结合"离中天多近"和"绝对高度角"两项.
        near_transit = max(0.0, min(1.0, alt_deg / max(1.0, max_alt)))
        alt_factor = max(0.0, min(1.0, (alt_deg - self.min_alt) / max(1.0, max_alt - self.min_alt)))
        return 0.5 * near_transit + 0.5 * alt_factor

    def _required_anchor_centers(self, altaz) -> List[Tuple[float, float]]:
        """为当前可见的**未完成必观测目标**生成候选视场中心 (时序感知).

        对每个未完成必观测目标, 生成若干"能让它落入某根光纤"的中心 (以其为锚点,
        用光纤中心偏移反推指向). 排序时按"时序质量 × 紧迫度"优先: 越接近其最佳时段
        (中天/高仰角)、以及尝试过但未完成的目标 (需重试), 越优先被锚定.
        """
        return self._required_anchor_centers_impl(altaz)

    def _required_anchor_centers_impl(self, altaz) -> List[Tuple[float, float]]:
        """时序感知的必观测锚定实现.

        排序依据 (从高到低):
            1. 时序质量: 当前时刻越接近目标的中天/高仰角越好 (Cao 2025: 得分随
               高度角近似抛物线);
            2. 重试紧迫度: 尝试过但未完成的目标优先 (需在更好时段重试);
            3. target_id 稳定排序 (确定性).

        每轮最多锚定 ``max_anchor`` 个目标, 控制候选规模与计算预算.
        """
        centers: List[Tuple[float, float]] = []
        if not self._required_unfinished:
            return centers
        # 预算: 每轮最多 MAX_REQUIRED_ANCHOR_CENTERS 个候选中心 -> 目标数 x 中心光纤数
        fiber_ids = self._anchor_fibers[:2]
        max_anchor = max(1, MAX_REQUIRED_ANCHOR_CENTERS // max(1, len(fiber_ids)))
        # 只锚定"当前可见"的未完成必观测目标.
        visible_unfinished = [tid for tid in self._required_unfinished
                              if tid in altaz and altaz[tid][0] >= self.min_alt + ALT_MARGIN_DEG]
        # 轮转: 刚锚定过的目标本轮让位, 否则"按时序质量排序取前 N 个"会让
        # 时序质量偏低的目标 (例如中天高度角较低者) 被永久饿死 —— 实测卡B 有
        # 161 个必观测全场可见 (中位 874 个 slot) 却一次都没被指派。
        cooldown = ANCHOR_COOLDOWN_DECISIONS
        seq = self.decisions_seen
        fresh_pool = [tid for tid in visible_unfinished
                      if seq - self._last_anchor.get(tid, -10 ** 9) > cooldown]
        if fresh_pool:
            visible_unfinished = fresh_pool

        def rank_key(tid: str):
            t = self.targets.index.get(tid)
            alt = altaz[tid][0]
            tq = self._time_quality(t, alt) if t is not None else 0.0
            attempts = self.required_attempts.get(tid, 0)
            g_upper = self.achieved_g.get(tid, 0.0)
            retry = 1.0 if (attempts > 0 and g_upper < 0.5) else 0.0
            return (-(tq + 0.15 * retry), tid)

        visible_unfinished.sort(key=rank_key)
        # 时序感知延迟: 优先只锚定"当前高度角接近其最大值(中天附近)"的目标, 把
        # 得分不佳的低仰角窗口让给普通目标; 但"重试过仍未完成"的目标不受此限制
        # (避免因延迟而永远错过).
        gated = []
        for tid in visible_unfinished:
            t = self.targets.index.get(tid)
            if t is None:
                gated.append(tid)
                continue
            max_alt = self._transit_altitude(t.dec_deg)
            alt = altaz[tid][0]
            near_transit = max_alt <= 0 or alt >= self.required_defer_frac * max_alt
            retry = self.required_attempts.get(tid, 0) > 0 and self.achieved_g.get(tid, 0.0) < 0.5
            if near_transit or retry:
                gated.append(tid)
        # 若门控后为空 (全部目标都处于低仰角), 则回退到全部可见未完成目标.
        candidates_anchor = gated or visible_unfinished

        if len(candidates_anchor) > max_anchor:
            # 优先保留时序质量最高的一批; 其余均匀抽样以覆盖更多目标.
            head = candidates_anchor[: max_anchor // 2]
            rest = candidates_anchor[max_anchor // 2:]
            stride = max(1, len(rest) // max(1, max_anchor - len(head)))
            sampled = rest[::stride][: max_anchor - len(head)]
            candidates_anchor = head + sampled
        for tid in candidates_anchor:
            self._last_anchor[tid] = seq      # 记录锚定时刻, 供下一轮轮转
            a_alt, a_az = altaz[tid]
            for fiber_id in fiber_ids:
                row, col = divmod(fiber_id, self.grid_side)
                d_north, d_east = self._fiber_grid.cell_center(row, col)
                centers.append(shift_altaz(a_alt, a_az, -d_north, -d_east))
        return centers

    def _continuity_seed_centers(self, altaz) -> List[Tuple[float, float]]:
        """生成"延续扫描"的候选视场中心 (收紧搜索范围, 提升收敛性).

        在上次指向附近按若干步长 (几度) 布点, 并沿扫描方向延伸, 使望远镜能够:
            * 小幅推进 -> 平滑扫天;
            * 沿既有方向继续 -> 形成有方向的扫描.
        这些中心仍会经过 _fill_pointing 与连续性打分, 若质量太差则不会胜出.
        """
        seeds: List[Tuple[float, float]] = []
        if self.prev_pointing is None:
            return seeds
        p_alt, p_az = self.prev_pointing
        fov = self._fiber_grid.fov_side_deg
        # 步长: 整视场宽度的若干比例 (保证相邻视场有合理重叠但不完全相同)
        step = max(1.0, fov * 0.75)
        # 扫描方向单位化; 无方向时默认沿方位角正向.
        if self.sweep_altaz is not None and math.hypot(*self.sweep_altaz) > 1e-6:
            s_alt, s_az = self.sweep_altaz
        else:
            s_alt, s_az = 0.0, 1.0
        norm = math.hypot(s_alt, s_az) or 1.0
        u_alt, u_az = s_alt / norm, s_az / norm
        # 1) 沿扫描方向的前进点 (1~3 个步长)
        for k in (1.0, 2.0, 3.0):
            seeds.append(shift_altaz(p_alt, p_az, u_alt * step * k, u_az * step * k))
        # 2) 前进点两侧的横向展开 (覆盖扫描带宽度, 避免只走一条线)
        perp_alt, perp_az = -u_az, u_alt
        for k in (1.0, 2.0):
            for side in (1.0, -1.0):
                seeds.append(shift_altaz(p_alt, p_az,
                                         u_alt * step * k + perp_alt * step * side,
                                         u_az * step * k + perp_az * step * side))
        return seeds

    def _repeat_pointing_penalty(self, center_alt: float, center_az: float) -> float:
        """对与上次几乎重合的指向施加惩罚, 避免连续曝光同一视场 (多次不累加).

        以球面角距衡量; 若与上次指向角距小于视场半径的 1/2, 视为重复视场, 给
        一个足以让其它候选胜出的惩罚.
        """
        if self.last_pointing is None:
            return 0.0
        prev_alt, prev_az = self.last_pointing
        sep = angular_separation_altaz(center_alt, center_az, prev_alt, prev_az)
        if sep < self._fiber_grid.fov_side_deg / 2.0:
            return 10000.0
        return 0.0


    def cpu_summary(self) -> str:
        """一行性能摘要 (agent 在 finish 时写日志 / 本地预算仿真读取)."""
        n = self.decision_cpu_count
        avg_ms = (self.decision_cpu_total / n * 1000.0) if n else 0.0
        return (f"决策 {n} 轮, 平均 {avg_ms:.0f} ms/决策, 累计 {self.decision_cpu_total:.1f}s CPU, "
                f"目标 {len(self.targets.targets)} 个 (本轮可见 {self.perf_visible_n}), "
                f"视场中心评估 {self.perf_fill_calls} 次/轮, numpy={HAVE_NUMPY}")


    def _fill_pointing(self, ranked, priorities, altaz, center_alt, center_az, spatial=None):
        """在给定指向下, 贪心填充光纤, 返回 (assignments, total_priority).

        复用 preplan.FiberGrid.classify; 每根光纤至多一个目标, 目标需全程高于
        最低高度角 (用 ALT_MARGIN 余量近似). 先用 FOV 半径快速排除远处目标,
        再按优先级贪心分配, 保证单次曝光内光纤不重复.

        ``spatial`` 为可选的 (alt,az) 分箱索引: 若提供, 只遍历视场附近的少量目标
        (O(局部) 而非 O(全部目标)), 在 100 光纤 / 数万目标的大卡上显著提速.
        """
        self.perf_fill_calls += 1       # 性能计数 (只用于日志/本地预算仿真)
        grid = self._fiber_grid
        half = grid.fov_side_deg / 2.0
        assignments: Dict[str, str] = {}
        used = set()
        total = 0.0
        if spatial is not None:
            candidates_iter = self._nearby_targets(spatial, center_alt, center_az, half * 1.6)
        else:
            candidates_iter = ranked
        for t in candidates_iter:
            if len(assignments) >= self.n_fibers:
                break
            t_alt, t_az = altaz[t.target_id]
            if t_alt < self.min_alt + ALT_MARGIN_DEG:
                continue
            # 快速球面距离筛选 (避免昂贵的切平面投影)
            d_alt = t_alt - center_alt
            d_az = wrap180(t_az - center_az) * max(0.1, math.cos(math.radians((t_alt + center_alt) / 2)))
            if abs(d_alt) > half * 1.5 or abs(d_az) > half * 1.5:
                continue
            offsets = tangent_offsets(t_alt, t_az, center_alt, center_az)
            if offsets is None:
                continue
            # 注意: tangent_offsets 返回 (north, east); FiberGrid.classify 需要
            # (east, north). 切勿写反 —— 写反会把 16 根光纤的方位整体交换,
            # 导致目标落到别的光纤 (命中率骤降). [已修复的历史缺陷]
            north, east = offsets
            fiber, margin = grid.classify(east, north)
            if fiber < 0 or fiber in used:
                continue
            if margin < 0.03:  # 太靠边, 易脱靶
                continue
            used.add(fiber)
            assignments[str(fiber)] = t.target_id
            total += priorities.get(t.target_id, 0.0)
        return assignments, total

    def _build_spatial_index(self, ranked, altaz, bin_deg: float) -> Dict[Tuple[int, int], list]:
        """把按优先级降序的目标按 (alt, az) 分箱, 供 _nearby_targets 快速取邻域.

        bin_deg 取略大于视场尺寸, 保证视场内目标落在中心 bin 及其相邻 bin.
        每箱内保持原 ranked 顺序 (即优先级降序), 便于贪心填充.
        """
        index: Dict[Tuple[int, int], list] = {}
        for t in ranked:
            pair = altaz.get(t.target_id)
            if pair is None:
                continue
            a, z = pair
            key = (int(a // bin_deg), int((z % 360.0) // bin_deg))
            index.setdefault(key, []).append(t)
        return index

    @staticmethod
    def _nearby_targets(spatial, center_alt: float, center_az: float, radius_deg: float):
        """从分箱索引中取视场附近的目标 (3x3 邻域即可覆盖半径内所有目标).

        性能: 一轮里会评估几十到上百个候选中心, 其中很多落在同一个 (alt,az) 分箱,
        邻域完全相同。这里按分箱缓存结果, 命中时直接返回; 未命中时用 ``heapq.merge``
        合并各 bin (每个 bin 内部已按优先级有序), 避免每调用一次就整体排序。
        """
        b = spatial["bin_deg"]
        # 邻域半径以 bin 为单位 (向上取整, 覆盖 radius_deg)
        r = int(radius_deg // b) + 1
        ca = int(center_alt // b)
        cz = int((center_az % 360.0) // b)
        cache = spatial.get("cache")
        if cache is not None:
            hit = cache.get((ca, cz))
            if hit is not None:
                return hit
        n_az_bins = max(1, int(360.0 // b) + 1)
        order = spatial["order"]
        lists = []
        for da in range(-r, r + 1):
            ai = ca + da
            for dz in range(-r, r + 1):
                zi = (cz + dz) % n_az_bins
                lst = spatial["index"].get((ai, zi))
                if lst:
                    lists.append(lst)
        if not lists:
            seen = []
        elif len(lists) == 1:
            seen = lists[0]
        else:
            # 各 bin 内已按 order 升序 -> 归并即得全局顺序 (O(k log b) 而非整体排序)
            seen = list(heapq.merge(*lists, key=lambda t: order.get(t.target_id, 0)))
        if cache is not None:
            cache[(ca, cz)] = seen
        return seen

    def _estimate_program(self, assignments, altaz, lst, now) -> str:
        """按月亮 + 大气质量模型估计观测程序档位 (纯静态回退时的 DARK/BRIGHT 选择).

        使用公开的 scoring.program.bands 与 lunar_model; 不读取任何隐藏天气真值.
        """
        scoring = self.init.scoring or {}
        program_cfg = scoring.get("program") or {}
        bands = {**{"DARK": 0.65, "BRIGHT": 0.40}, **(program_cfg.get("bands") or {})}
        q0 = float(scoring.get("q0", 1.0)) or 1.0
        beta = float(scoring.get("airmass_exponent", 0.6))
        lunar_model = scoring.get("lunar_model") or {}

        moon = _Moon(now, lst, self.lat)
        values = []
        for tid in assignments.values():
            t = self.targets.index.get(tid)
            if t is None:
                continue
            t_alt = altaz.get(tid, (None, None))[0]
            if t_alt is None:
                continue
            airmass = normalized_airmass(max(t_alt, 1.0))
            lunar = _lunar_factor(moon, t.ra_deg, t.dec_deg, lunar_model)
            quality = (lunar / (q0 * (airmass ** beta))) if airmass > 0 else 0.0
            values.append(quality)
        if not values:
            return "BACKUP"
        mean_q = sum(values) / len(values)
        if mean_q >= bands.get("DARK", 0.65):
            return "DARK"
        if mean_q >= bands.get("BRIGHT", 0.40):
            return "BRIGHT"
        return "BACKUP"

    def _choose_duration(self, assignments, seconds_left: float, altaz, lst, now=None,
                         extra_must_complete: Optional[set] = None) -> int:
        """选择曝光时长.

        策略 (参考 Cao 2025 的曝光时间计算器思想):
            * 对**必须完成**的目标 (必观测 + 限时请求目标), 计算"达到完成因子
              threshold=0.5 所需的最短曝光", 取所需最长者且不缩放;
            * 普通目标用 preplan 的建议曝光 (以计分基准曝光为锚, 按亮度缩放);
            * **预算紧张时** (pace >= 1) 把曝光拉长到"本视场最后一个目标饱和"的
              时长 —— 同一枪内所有目标共享曝光时间, 延长到最暗目标饱和既不掉分,
              又能显著减少决策次数 (这是 CPU 预算的第二杠杆);
            * 最后裁剪到 [min, max]、剩余时间, 以及"目标掉出高度门槛之前"。

        完成因子模型 (公开部分):  g = flux * T * Q / (f0*T0),
        其中 Q ≈ (η·τ·K·L)/(seeing·X^β)/q0. 隐藏项 η·τ·K/s 未知, 这里用一个
        偏保守但合理的公开估计 Q_hat = L/(q0·X^β·seeing_ref) (seeing_ref 取典型值),
        以便在常见天气下把必观测推到 0.5 以上.
        """
        must_complete = set(extra_must_complete or ())
        seconds = []
        required_need = 0        # 本视场中"必须完成"目标所需的最短曝光 (不缩放)
        # 月球只计算一次 (供所有必观测目标的完成因子估算复用).
        moon = None
        scoring = self.init.scoring if isinstance(self.init.scoring, dict) else {}
        f0 = float(scoring.get("flux_zero_point", 0.5)) or 0.5
        t0 = float(scoring.get("exposure_zero_point_seconds", 900.0)) or 900.0
        needs_moon = any(
            (self.targets.index.get(tid) is not None
             and (self.targets.index[tid].required or tid in must_complete))
            for tid in assignments.values()
        )
        if now is not None and needs_moon:
            try:
                moon = _Moon(now, lst, self.lat)
            except Exception:
                moon = None
        for tid in assignments.values():
            t = self.targets.index.get(tid)
            if t is None:
                continue
            suggested = preplan.suggest_exposure_seconds(
                preplan.Target(t.target_id, t.ra_deg, t.dec_deg, t.target_class,
                               t.feature_flux, t.science_weight, t.required),
                self.tool._temp_card([t]),
            )
            if t.required or tid in must_complete:
                need = self._exposure_for_completion(t, altaz.get(tid), moon)
                # 时序感知重试升级: 若该必观测目标此前已尝试但未完成, 说明一次
                # 短曝光在当时的天气/时段下落空; 增大所需曝光 (重试升级), 提高成功
                # 概率. 升级系数随尝试次数增长, 上限 1.5 倍.
                prior = self.required_attempts.get(str(tid), 0)
                g_upper = self.achieved_g.get(str(tid), 0.0)
                if prior > 0 and g_upper < 0.5:
                    need = int(need * min(1.5, 1.0 + 0.25 * prior))
                required_need = max(required_need, need)
            seconds.append(suggested)
        base = max(seconds) if seconds else 900
        normal_duration = int(round(base * self.duration_scale / 30.0) * 30)
        duration = normal_duration
        if required_need > 0:
            # 必观测目标所需时长优先且不缩放; 若 need 已超过 max_exposure 的 60%,
            # 说明该目标较暗, 直接给满 max_exposure, 提高成功概率.
            required_duration = required_need
            if required_need >= 0.6 * self.max_exposure:
                required_duration = self.max_exposure
            duration = max(duration, required_duration)

        # -- 预算紧张: 拉长到"本视场最后一个目标饱和"的时长, 减少决策次数 -------
        if self.pace_level >= 1 and now is not None and assignments:
            t_sat = 0.0
            for tid in assignments.values():
                t = self.targets.index.get(tid)
                if t is None:
                    continue
                q_hat = self._quality_hat(t, altaz.get(tid), moon)
                if q_hat <= 0.0:
                    continue
                t_sat = max(t_sat, f0 * t0 / (max(1e-6, t.feature_flux) * q_hat))
            if t_sat > duration:
                duration = min(self.max_exposure, int(round(t_sat / 30.0) * 30))

        duration = max(self.min_exposure, min(self.max_exposure, duration))
        duration = int(max(self.min_exposure, min(duration, seconds_left)))
        # 预算紧张时才拉长曝光; 既然拉长了, 就用"目标掉出高度门槛之前"给个安全上限
        # (否则一次长曝光可能中途跌破 30 度, 整枪 0 分)。pace 0 (预算充裕) 不启用,
        # 保持与既有节奏一致。
        if self.pace_level >= 1:
            runway = self._altitude_runway_seconds(assignments, lst)
            duration = int(max(self.min_exposure, min(duration, runway, seconds_left)))
        return duration

    def _altitude_runway_seconds(self, assignments, lst: float) -> float:
        """本视场中"最早掉出高度门槛"的目标还能观测多少秒 (留 10% 余量).

        目标是全程 >= 30° 才计分; 这里用 31° (min_alt + 安全余量) 的穿越时刻再乘
        0.9, 因此不会真的把目标拖到门槛以下。
        """
        best = float("inf")
        for tid in assignments.values():
            i = self._sky.pos.get(str(tid))
            if i is None:
                continue
            best = min(best, self._sky.seconds_to_set(i, lst))
        if best == float("inf"):
            return 1e9
        return max(60.0, best * 0.9)

    def _quality_hat(self, target: Target, altaz_pair, moon,
                     conservative: bool = False) -> float:
        """公开可得的天空质量估计 ``Q_hat = L / (q0 * X^beta * seeing_ref)``.

        隐藏项 (η/τ/K/seeing 真值) 不可知, 这里用典型 seeing 取值使估计偏保守
        (宁可曝光略长)。返回 0 表示无法估计。

        ``conservative=True`` 时再乘 ``COMPLETION_Q_SCALE`` (≈0.65), 专用于
        "达标所需曝光"的推算: 平台实测 (卡B) 106 个必观测的最好成绩卡在
        g∈[0.40,0.50) 且曝光远未触顶, 说明原估计偏乐观 15~40%, 直接导致漏失。
        """
        if altaz_pair is None:
            return 0.0
        scoring = self.init.scoring if isinstance(self.init.scoring, dict) else {}
        q0 = float(scoring.get("q0", 1.0)) or 1.0
        beta = float(scoring.get("airmass_exponent", 0.6))
        lunar_model = scoring.get("lunar_model") or {}
        alt = altaz_pair[0]
        airmass = normalized_airmass(max(alt, 1.0))
        lunar = 1.0
        if moon is not None:
            lunar = _lunar_factor(moon, target.ra_deg, target.dec_deg, lunar_model)
        q = lunar / (q0 * (airmass ** beta) * max(1e-6, SEEING_REF_FOR_SCORING))
        return q * (COMPLETION_Q_SCALE if conservative else 1.0)

    def _exposure_for_completion(self, target: Target, altaz_pair, moon, threshold: float = 0.5) -> int:
        """估算某目标达到完成因子 ``threshold`` 所需的曝光秒数 (公开模型, 无隐藏真值).

        g = flux·T·Q/(f0·T0) >= threshold  =>  T >= threshold·f0·T0/(flux·Q).
        Q 用公开几何+月光的估计, 并对 seeing 取一个典型参考值 (隐藏真值不可知),
        使结果在常见天气下偏保守 (宁可曝光略长). ``moon`` 由调用方预先算好并复用.
        """
        if altaz_pair is None:
            return self.min_exposure
        scoring = self.init.scoring if isinstance(self.init.scoring, dict) else {}
        f0 = float(scoring.get("flux_zero_point", 0.5)) or 0.5
        t0 = float(scoring.get("exposure_zero_point_seconds", 900.0)) or 900.0
        # conservative=True: 用打了折的质量估计反解曝光, 保证真实 g 越过 0.5
        q_hat = self._quality_hat(target, altaz_pair, moon, conservative=True)
        flux = max(1e-6, target.feature_flux)
        need = threshold * f0 * t0 / (flux * max(1e-6, q_hat))
        need = max(self.min_exposure, min(self.max_exposure, need))
        return int(round(need / 30.0) * 30)

    @staticmethod
    def _centroid(targets: Sequence[Target]) -> Tuple[float, float]:
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
# 工具函数
# ---------------------------------------------------------------------------


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _format_utc(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_card_for(name_or_path: str) -> Optional[preplan.CardData]:
    """尝试加载任务卡供运行时复用 (失败返回 None, 不崩溃)."""
    try:
        return preplan.CardData.from_card(name_or_path)
    except Exception:
        return None


def card_slug_candidates(card_id: str) -> List[str]:
    """把平台下发的 card_id 映射为可能的卡片目录名 (按优先级去重).

    正式赛下发的 ``card_id`` 是 ``A`` / ``B`` / ``C`` / ``D`` / ``A1`` ...,
    而仓库中的卡片目录名是 ``cardA`` / ``cardB`` / ...; 练习卡则直接叫
    ``alpha`` / ``beta`` / ...。这里同时尝试"原样"与"加 card 前缀"两种写法,
    避免因命名不一致而误判为"找不到任务卡"。
    """
    raw = str(card_id or "").strip()
    if not raw:
        return []
    candidates: List[str] = []
    for variant in (raw, raw.lower(), raw.upper(),
                    "card" + raw, "card" + raw.upper(), "card" + raw.lower()):
        if variant and variant not in candidates:
            candidates.append(variant)
    return candidates


def resolve_card(init_data) -> Optional[preplan.CardData]:
    """按平台 ``card_id`` 定位任务卡 (兼容 ``A``/``A1`` -> ``cardA``/``cardA1``).

    放在 planner 层而非 agent 层, 便于单元测试直接调用 (导入 agent 会触发
    stdout 硬化, 影响 protocol 的 stdout 测试)。找不到返回 ``None``, 由
    :class:`Planner` 用 initialize payload 兜底。
    """
    card_id = getattr(init_data, "card_id", "")
    for slug in card_slug_candidates(card_id):
        card = load_card_for(slug)
        if card is not None:
            log(f"agent: 任务卡已加载 card_id={card_id!r} -> {card.name!r}")
            return card
    log(f"agent: 未找到任务卡 {card_id!r} (尝试过 {card_slug_candidates(card_id)}); "
        f"将使用协议 payload 中的目标与仪器参数")
    return None
