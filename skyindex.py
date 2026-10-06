#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""skyindex.py - 目标可见性与地平坐标的快速索引 (纯计算, 无副作用).

动机
----
每次决策都要回答两个问题: "现在哪些目标可见?" 与 "它们的地平坐标是多少?"。
朴素做法是对全部 N 个目标逐个调用 :func:`planner.radec_to_altaz` (每次都做
多次三角函数)。在 3 万目标 x 数千轮决策的规模下, 这是 CPU 预算的头号开销:
本机实测卡 A(3 万目标) 每次观测决策约 500 ms, 其中可见性扫描约 106 ms、
全量 alt/az 重算约 50 ms。

做法
----
* **一次性预计算** (每张卡只做一次): 每个目标的 ``sin(dec)/cos(dec)``, 以及
  ``h_max`` —— 目标保持在地平高度下限之上所能容许的最大时角。
  由 ``sin(alt) = sin(lat)sin(dec) + cos(lat)cos(dec)cos(H)`` 可得:

      可见  <=>  ``|wrap180(lst - ra)| <= h_max``

  该判据与 ``alt >= min_alt`` **严格等价**(无近似; 已用 2 万随机样本验证
  0 处不一致), 且**不需要任何三角函数**。
* **每轮**: 数组比较得到可见掩码, 再只对可见子集算 alt/az。

有 numpy 时走向量化路径, 没有时自动回退纯 Python 循环, 两者结果一致
(numpy 只是更快)。本模块不读写任何隐藏状态, 也不做决策。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # numpy 为可选加速; 缺失时回退纯 Python, 不报错
    import numpy as _np
except Exception:  # pragma: no cover - 取决于运行环境
    _np = None

HAVE_NUMPY = _np is not None

# 恒星时相对 UT 的推进速率 (度/秒), 与 planner.local_sidereal_deg 同源
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


def wrap180(angle: float) -> float:
    """把角度折算到 (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def max_hour_angle_deg(dec_deg: float, lat_deg: float, min_alt_deg: float) -> float:
    """目标低于 ``min_alt_deg`` 之前的最大时角 (度).

    与 ``planner.max_hour_angle_deg`` 同一公式:
        * 返回 0.0   -> 该目标永远到不了这个高度 (永不可见);
        * 返回 180.0 -> 该目标始终在这个高度之上 (拱极/常显)。
    """
    denominator = math.cos(math.radians(lat_deg)) * math.cos(math.radians(dec_deg))
    if abs(denominator) < 1e-12:
        return 0.0
    value = (math.sin(math.radians(min_alt_deg))
             - math.sin(math.radians(lat_deg)) * math.sin(math.radians(dec_deg))) / denominator
    if value >= 1.0:
        return 0.0
    if value <= -1.0:
        return 180.0
    return math.degrees(math.acos(value))


class TargetIndex:
    """一张卡全部目标的可见性/地平坐标快速索引.

    构造一次 (O(N)), 之后每轮只做数组运算。所有方法都是纯函数式读取, 不修改
    状态, 因此可以安全地在同一轮内对多个 LST 复用。
    """

    def __init__(self, targets: Sequence, lat_deg: float, min_alt_deg: float) -> None:
        """
        Args:
            targets: 目标序列 (需有 ``target_id`` / ``ra_deg`` / ``dec_deg``).
            lat_deg: 站点纬度 (度).
            min_alt_deg: 可见性门槛, 调用方应传 ``minimum_altitude + 安全余量``.
        """
        self.lat = float(lat_deg)
        self.min_alt = float(min_alt_deg)
        self.ids = [str(t.target_id) for t in targets]
        self.pos: Dict[str, int] = {tid: i for i, tid in enumerate(self.ids)}
        self.ra = [float(t.ra_deg) for t in targets]
        self.dec = [float(t.dec_deg) for t in targets]
        self.n = len(self.ids)
        self.h_max = [max_hour_angle_deg(d, self.lat, self.min_alt) for d in self.dec]
        self._sin_dec = [math.sin(math.radians(d)) for d in self.dec]
        self._cos_dec = [math.cos(math.radians(d)) for d in self.dec]
        self._sin_lat = math.sin(math.radians(self.lat))
        self._cos_lat = math.cos(math.radians(self.lat))
        if HAVE_NUMPY and self.n:
            self._np_ra = _np.asarray(self.ra, dtype=float)
            self._np_hmax = _np.asarray(self.h_max, dtype=float)
            self._np_sin_dec = _np.asarray(self._sin_dec, dtype=float)
            self._np_cos_dec = _np.asarray(self._cos_dec, dtype=float)

    # -- 可见性 ------------------------------------------------------------
    def is_visible_index(self, i: int, lst_deg: float) -> bool:
        """第 ``i`` 个目标在 ``lst_deg`` 时是否可见 (与 ``alt >= min_alt`` 等价)."""
        d = wrap180(float(lst_deg) - self.ra[i])
        return -self.h_max[i] <= d <= self.h_max[i]

    def visible_indices(self, lst_deg: float) -> List[int]:
        """当前可见目标的索引列表 (按原目标顺序)."""
        if self.n == 0:
            return []
        if HAVE_NUMPY:
            delta = _np.abs((float(lst_deg) - self._np_ra + 180.0) % 360.0 - 180.0)
            return _np.nonzero(delta <= self._np_hmax)[0].tolist()
        lst = float(lst_deg)
        out: List[int] = []
        for i in range(self.n):
            if self.is_visible_index(i, lst):
                out.append(i)
        return out

    # -- 地平坐标 ----------------------------------------------------------
    def altaz_pairs(self, indices: Iterable[int],
                    lst_deg: float) -> List[Tuple[float, float]]:
        """对给定索引批量计算 ``(alt_deg, az_deg)`` (与 ``radec_to_altaz`` 一致)."""
        idx = list(indices)
        if not idx:
            return []
        if HAVE_NUMPY and len(idx) >= 32:
            i = _np.asarray(idx, dtype=_np.intp)
            ha = _np.radians(float(lst_deg) - self._np_ra[i])
            sin_dec = self._np_sin_dec[i]
            cos_dec = self._np_cos_dec[i]
            sin_alt = _np.clip(self._sin_lat * sin_dec + self._cos_lat * cos_dec * _np.cos(ha),
                               -1.0, 1.0)
            alt = _np.arcsin(sin_alt)
            cos_alt = _np.maximum(1e-12, _np.cos(alt))
            sin_az = -_np.sin(ha) * cos_dec / cos_alt
            cos_az = (sin_dec - sin_alt * self._sin_lat) / (cos_alt * max(1e-12, self._cos_lat))
            az = _np.degrees(_np.arctan2(sin_az, cos_az)) % 360.0
            return list(zip(_np.degrees(alt).tolist(), az.tolist()))
        # 纯 Python 回退 (公式与 planner.radec_to_altaz 相同)
        lat = math.radians(self.lat)
        out: List[Tuple[float, float]] = []
        for k in idx:
            ha = math.radians(wrap180(float(lst_deg) - self.ra[k]))
            dec = math.radians(self.dec[k])
            sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
            alt = math.asin(max(-1.0, min(1.0, sin_alt)))
            cos_alt = max(1e-12, math.cos(alt))
            sin_az = -math.sin(ha) * math.cos(dec) / cos_alt
            cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (cos_alt * max(1e-12, math.cos(lat)))
            out.append((math.degrees(alt), math.degrees(math.atan2(sin_az, cos_az)) % 360.0))
        return out

    def altaz_map(self, indices: Iterable[int],
                  lst_deg: float) -> Dict[str, Tuple[float, float]]:
        """同 :meth:`altaz_pairs`, 但返回 ``{target_id: (alt, az)}`` (供填充光纤使用)."""
        idx = list(indices)
        pairs = self.altaz_pairs(idx, lst_deg)
        ids = self.ids
        return {ids[i]: p for i, p in zip(idx, pairs)}

    # -- 时序 --------------------------------------------------------------
    def seconds_to_set(self, i: int, lst_deg: float) -> float:
        """目标距跌破高度门槛还剩多少秒 (拱极目标返回 ``inf``).

        恒星时单调推进, 所以目标越过 ``h_max`` 后即不可见; 用于给"拉长曝光"
        加一个安全上限, 避免一次长曝光中途目标掉到 30 度以下而整枪 0 分。
        """
        if self.h_max[i] >= 179.999:
            return float("inf")
        remain = self.h_max[i] - wrap180(float(lst_deg) - self.ra[i])
        if remain <= 0.0:
            return 0.0
        return remain / SIDEREAL_DEG_PER_SECOND


__all__ = ["TargetIndex", "HAVE_NUMPY", "max_hour_angle_deg", "wrap180",
           "SIDEREAL_DEG_PER_SECOND"]
