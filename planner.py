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

几何公式来源: 官方 ``docs/participant-guide`` 的 Geometry 章节与官方示例
``agent_core/geometry.py`` (公有公式, 不依赖任何隐藏数据).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import preplan
from models import Action, DecisionState, NightPlan, Target
from protocol import log

# ---------------------------------------------------------------------------
# 天文几何 (标准公式, 与官方示例一致)
# ---------------------------------------------------------------------------

SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0
DEFAULT_MIN_ALTITUDE_DEG = 30.0
ALT_MARGIN_DEG = 1.0  # 规划时目标需高于最低高度角的安全余量


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

        # 锚点搜索参数 (预算与质量的折中); 会按剩余 CPU 预算自适应缩小
        self.anchor_pool = 160          # 快速探测的锚点上限
        self.refine_anchors = 4         # 精细搜索的最密锚点数
        self._center_fiber = 5          # 快速探测使用的中心光纤 (靠中间)
        self.pace_level = 0             # 0=充裕 1=适中 2=紧张

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

    def _build_grid(self) -> preplan.FiberGrid:
        card = self.tool.card
        if card is not None:
            return preplan.build_fiber_grid(card)
        return preplan.FiberGrid(
            side=self.grid_side,
            n_fibers=self.n_fibers,
            fiber_side_deg=math.sqrt(0.4),
            gap_deg=0.0,
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

        self._update_result(req)

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
            return Action(type="wait", duration_seconds=self._to_next_slot(now, night_start),
                          reason="简报: 全天天雨/暴风", decision_source="static")

        # 夜晚即将结束
        if (night_end - now).total_seconds() < self.min_exposure:
            nxt = self._next_night_start(now)
            if nxt is None:
                return Action(type="finish", reason="巡天结束", decision_source="static")
            return Action(type="wait", until_utc=_format_utc(nxt), reason="本夜将尽", decision_source="static")

        is_new_night = night_index != self.last_night_index
        self.last_night_index = night_index

        self._update_pace(req)
        state = self._build_state(req, night_index, night_start, night_end, is_new_night)

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

        # 环节B·行动决策: LLM 给高层意向 (不是每轮都调)
        llm_action = None
        if self.llm is not None:
            llm_action = self.llm.decide_action(state, self.night_plan)
        if llm_action is not None and llm_action.type == "report":
            if self._can_report():
                return self._make_report("LLM 高层决策")
        if llm_action is not None and llm_action.type == "finish":
            return Action(type="finish", reason=llm_action.reason, decision_source="llm")
        if llm_action is not None and llm_action.type == "wait":
            return Action(type="wait", duration_seconds=self._to_next_slot(now, night_start),
                          reason=llm_action.reason, decision_source="llm")

        # 落成 observe (LLM 意向或纯静态)
        action = self._plan_observe(now, night_end, night_index, state)
        if action is None:
            return Action(type="wait", duration_seconds=self._to_next_slot(now, night_start),
                          reason="暂无可观测目标", decision_source="static")
        return action

    # -- 状态与统计 --------------------------------------------------------
    def _update_result(self, req) -> None:
        lr = req.last_result
        if lr and lr.get("action") == "observe":
            assigned = int(lr.get("assigned_count", 0) or 0)
            hit = int(lr.get("hit_count", 0) or 0)
            self.assigned_total += assigned
            self.hits_total += hit
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
            for n in self.init.nights:
                end = _parse_utc(n.get("observing_end_utc"))
                start = _parse_utc(n.get("observing_start_utc"))
                if end and end > req.now_utc:
                    eff_start = max(start, req.now_utc) if start else req.now_utc
                    decisions_left += max(0.0, (end - eff_start).total_seconds()) / 700.0
        per_decision = cpu_left / max(1.0, decisions_left)
        # 阈值: 每决策可用 CPU 秒
        level = 0 if per_decision > 0.35 else 1 if per_decision > 0.10 else 2
        if level != self.pace_level:
            self.log(f"planner: pace level {level} (每决策可用 {per_decision * 1000:.0f} ms CPU, 剩余 {cpu_left:.0f}s)")
        self.pace_level = level

    def _current_night(self, now):
        for idx, n in enumerate(self.init.nights):
            start = _parse_utc(n.get("observing_start_utc"))
            end = _parse_utc(n.get("observing_end_utc"))
            if start and end and start <= now < end:
                return idx, start, end
        return None

    def _next_night_start(self, now):
        best = None
        for n in self.init.nights:
            start = _parse_utc(n.get("observing_start_utc"))
            if start and start > now and (best is None or start < best):
                best = start
        return best

    def _to_next_slot(self, now, night_start) -> int:
        into = (now - night_start).total_seconds() % self.slot_seconds
        return int(max(60, min(3600, self.slot_seconds - into if into else self.slot_seconds)))

    def _site_closed(self, req) -> bool:
        notices = self._all_notices(req)
        for n in notices:
            kind = n.get("event_kind")
            direction = n.get("direction")
            if kind in ("rain", "storm") and direction == "ALL":
                return True
        return False

    @staticmethod
    def _all_notices(req) -> List[Dict[str, Any]]:
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
    def _build_state(self, req, night_index, night_start, night_end, is_new_night) -> DecisionState:
        now = req.now_utc
        lst = local_sidereal_deg(now, self.lon)
        # 候选: 优先级排序 + 可见性过滤, 限制数量
        candidates = self._visible_candidates(lst, now)
        required_remaining = sum(
            1 for t in self.targets.targets
            if t.required and self._visible(t, lst)
        )
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

    def _visible(self, t: Target, lst: float) -> bool:
        alt, _ = radec_to_altaz(t.ra_deg, t.dec_deg, lst, self.lat)
        return alt >= self.min_alt + ALT_MARGIN_DEG

    def _visible_candidates(self, lst: float, now: datetime, limit: int = 96) -> List[str]:
        """可见目标按 preplan 优先级排序, 返回前 limit 个 ID."""
        visible = [t for t in self.targets.targets if self._visible(t, lst)]
        if not visible:
            return []
        priorities = self._priorities_for(visible)
        ranked = sorted(visible, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True)
        return [t.target_id for t in ranked[:limit]]

    def _priorities_for(self, targets: Sequence[Target]) -> Dict[str, float]:
        """调用 preplan 计算优先级 (复用, 不重写).

        若目标是全体目标, 则缓存结果 (优先级只依赖目标集合, 不随夜变化).
        """
        if len(targets) == len(self.targets.targets):
            if self._priorities_all is None:
                card = self.tool._temp_card(targets)
                self._priorities_all = preplan.compute_priorities(card)
            return self._priorities_all
        card = self.tool._temp_card(targets)
        return preplan.compute_priorities(card)

    # -- observe 规划 ------------------------------------------------------
    def _plan_observe(self, now, night_end, night_index, state: DecisionState) -> Optional[Action]:
        """把候选目标集落成一个合法的 observe 动作.

        方法 (复用 preplan 的优先级与 FiberGrid 方格模型, 但不使用其多次指向的
        `assign_fibers` 结果, 因为那样得到的光纤索引不属于同一曝光):

            1. 用 preplan.compute_priorities 对可见目标打分排序;
            2. 取若干高优先级"锚点"目标, 逐一作为视场中心的候选;
            3. 对每个候选中心, 用 FiberGrid.classify 把邻近目标映射到 16 根光纤
               (每根至多一个目标, 且目标需高于最低高度角);
            4. 选总优先级最高的中心作为最终指向, 并选择曝光时长与 program.
        """
        lst = local_sidereal_deg(now, self.lon)
        seconds_left = (night_end - now).total_seconds()
        if seconds_left < self.min_exposure:
            return None

        # 填充用: 全部可见目标 (尽可能填满 16 根光纤); 优先级只在可见集内计算一次.
        visible = [t for t in self.targets.targets if self._visible(t, lst)]
        if not visible:
            return None
        # 本夜规划目标作为优先级加成 (LLM/静态策略的落点)
        wanted = set(tid for tid in (self.night_plan.targets if self.night_plan else []) if tid in self.targets.index)
        priorities = self._priorities_for(visible)
        for tid in wanted:
            if tid in priorities:
                priorities[tid] += 500.0  # 计划优先目标获得显著加成
        ranked = sorted(visible, key=lambda t: priorities.get(t.target_id, 0.0), reverse=True)

        # 预计算每个目标的 alt/az (避免重复计算)
        altaz = {t.target_id: radec_to_altaz(t.ra_deg, t.dec_deg, lst, self.lat) for t in ranked}

        best = None  # (score, center_alt, center_az, assignments)
        seen_centers = set()
        # 锚点搜索: 参照官方示例的 anchor-search 精神.
        #   1) 先用少量中心光纤快速探测每个锚点的邻域密度, 选最"密"的锚点;
        #   2) 对最优锚点用全部 16 根光纤的中心偏移做精细搜索.
        # 锚点池: 既取最高优先级, 也按步长在全体可见目标中抽样 (兼顾优先级与密度).
        # 预算紧张时整体收缩 (预算保护).
        pool_size = {0: self.anchor_pool, 1: 60, 2: 20}[self.pace_level]
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
            assignments, total = self._fill_pointing(ranked, priorities, altaz, c_alt, c_az)
            if assignments:
                dense_centers.append((total + 50.0 * len(assignments), anchor, d_north, d_east))
        if not dense_centers:
            return None
        dense_centers.sort(key=lambda x: -x[0])
        # 精细搜索: 对最密的若干锚点, 尝试全部光纤中心偏移
        for _, anchor, _, _ in dense_centers[:refine]:
            a_alt, a_az = altaz[anchor.target_id]
            for fiber_id in range(self.n_fibers):
                row, col = divmod(fiber_id, self.grid_side)
                d_north, d_east = self._fiber_grid.cell_center(row, col)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (self.min_alt + 1.0 <= c_alt <= 89.0):
                    continue
                key = (round(c_alt, 1), round(c_az, 1))
                if key in seen_centers:
                    continue
                seen_centers.add(key)
                assignments, total = self._fill_pointing(ranked, priorities, altaz, c_alt, c_az)
                if not assignments:
                    continue
                score = total + 50.0 * len(assignments)
                if best is None or score > best[0]:
                    best = (score, c_alt, c_az, assignments)
            if best is not None and len(best[3]) >= self.n_fibers:
                break
        if best is None:
            return None

        _, center_alt, center_az, assignments = best
        duration = self._choose_duration(assignments, seconds_left, altaz, lst)
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
        self.log(f"planner: observe 指派 {len(assignments)} 根光纤, 曝光 {duration}s, 程序 {program}, "
                 f"指向 alt={center_alt:.2f} az={center_az:.2f}")
        return action

    def _fill_pointing(self, ranked, priorities, altaz, center_alt, center_az):
        """在给定指向下, 贪心填充 16 根光纤, 返回 (assignments, total_priority).

        复用 preplan.FiberGrid.classify; 每根光纤至多一个目标, 目标需全程高于
        最低高度角 (用 ALT_MARGIN 余量近似). 先用 FOV 半径快速排除远处目标,
        再按优先级贪心分配, 保证单次曝光内光纤不重复.
        """
        grid = self._fiber_grid
        half = grid.fov_side_deg / 2.0
        assignments: Dict[str, str] = {}
        used = set()
        total = 0.0
        for t in ranked:
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
            east, north = offsets
            fiber, margin = grid.classify(east, north)
            if fiber < 0 or fiber in used:
                continue
            if margin < 0.03:  # 太靠边, 易脱靶
                continue
            used.add(fiber)
            assignments[str(fiber)] = t.target_id
            total += priorities.get(t.target_id, 0.0)
        return assignments, total

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

    def _choose_duration(self, assignments, seconds_left: float, altaz, lst) -> int:
        """选择曝光时长: 以计分基准曝光 * duration_scale 为基准, 裁剪到合法范围与剩余时间.

        暗目标需要更长的曝光才能达到完成因子 1.0; 这里取分配目标中所需最长曝光为
        建议值 (复用 preplan 的建议逻辑思想, 但以实际分配为准).
        """
        suggestions = []
        for tid in assignments.values():
            t = self.targets.index.get(tid)
            if t is None:
                continue
            suggestions.append(
                preplan.suggest_exposure_seconds(
                    preplan.Target(t.target_id, t.ra_deg, t.dec_deg, t.target_class,
                                   t.feature_flux, t.science_weight, t.required),
                    self.tool._temp_card([t]),
                )
            )
        base = max(suggestions) if suggestions else 900
        duration = int(round(base * self.duration_scale / 30.0) * 30)
        duration = max(self.min_exposure, min(self.max_exposure, duration))
        duration = int(max(self.min_exposure, min(duration, seconds_left)))
        return duration

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
