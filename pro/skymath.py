"""Sky geometry for the agent: the same formulas the engine uses, so planned hits are real hits.

All angles in degrees. Azimuth: 0 = north, 90 = east. Standard library only.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def julian_date(moment: datetime) -> float:
    return moment.timestamp() / 86400.0 + 2440587.5


def local_sidereal_deg(moment: datetime, longitude_deg: float) -> float:
    days = julian_date(moment) - 2451545.0
    return (280.46061837 + 360.98564736629 * days + longitude_deg) % 360.0


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def radec_to_altaz(ra_deg: float, dec_deg: float, lst_deg: float, latitude_deg: float) -> tuple[float, float]:
    """Equatorial -> horizontal for a given local sidereal time."""
    hour_angle = math.radians(wrap180(lst_deg - ra_deg))
    lat = math.radians(latitude_deg)
    dec = math.radians(dec_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_alt = max(1e-12, math.cos(alt))
    sin_az = -math.sin(hour_angle) * math.cos(dec) / cos_alt
    cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (cos_alt * max(1e-12, math.cos(lat)))
    return math.degrees(alt), math.degrees(math.atan2(sin_az, cos_az)) % 360.0


def altaz_to_radec(alt_deg: float, az_deg: float, lst_deg: float, latitude_deg: float) -> tuple[float, float]:
    alt = math.radians(alt_deg)
    az = math.radians(az_deg)
    lat = math.radians(latitude_deg)
    sin_dec = math.sin(alt) * math.sin(lat) + math.cos(alt) * math.cos(lat) * math.cos(az)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))
    cos_dec = max(1e-12, math.cos(dec))
    sin_h = -math.sin(az) * math.cos(alt) / cos_dec
    cos_h = (math.sin(alt) - math.sin(dec) * math.sin(lat)) / (cos_dec * max(1e-12, math.cos(lat)))
    hour = math.degrees(math.atan2(sin_h, cos_h))
    return (lst_deg - hour) % 360.0, math.degrees(dec)


def max_hour_angle_deg(dec_deg: float, latitude_deg: float, min_alt_deg: float) -> float:
    """Largest |hour angle| at which a source stays at or above min_alt (0 = never, 180 = always)."""
    lat = math.radians(latitude_deg)
    dec = math.radians(dec_deg)
    denominator = math.cos(lat) * math.cos(dec)
    if abs(denominator) < 1e-12:
        return 0.0
    value = (math.sin(math.radians(min_alt_deg)) - math.sin(lat) * math.sin(dec)) / denominator
    if value >= 1.0:
        return 0.0
    if value <= -1.0:
        return 180.0
    return math.degrees(math.acos(value))


def tangent_offsets(target_alt: float, target_az: float, center_alt: float, center_az: float):
    """(north, east) gnomonic offsets in degrees of a target from a field centre, or None."""
    alt, az = math.radians(target_alt), math.radians(target_az)
    calt, caz = math.radians(center_alt), math.radians(center_az)
    t = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
    c = (math.cos(calt) * math.cos(caz), math.cos(calt) * math.sin(caz), math.sin(calt))
    north = (-math.sin(calt) * math.cos(caz), -math.sin(calt) * math.sin(caz), math.cos(calt))
    east = (-math.sin(caz), math.cos(caz), 0.0)
    depth = t[0] * c[0] + t[1] * c[1] + t[2] * c[2]
    if depth <= 0.0:
        return None
    return (math.degrees((t[0] * north[0] + t[1] * north[1] + t[2] * north[2]) / depth),
            math.degrees((t[0] * east[0] + t[1] * east[1] + t[2] * east[2]) / depth))


def shift_altaz(alt_deg: float, az_deg: float, d_north: float, d_east: float) -> tuple[float, float]:
    """The direction d_north / d_east degrees away on the tangent plane at (alt, az). Works near the zenith."""
    alt, az = math.radians(alt_deg), math.radians(az_deg)
    point = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
    north = (-math.sin(alt) * math.cos(az), -math.sin(alt) * math.sin(az), math.cos(alt))
    east = (-math.sin(az), math.cos(az), 0.0)
    dn, de = math.radians(d_north), math.radians(d_east)
    x, y, z = (p + dn * n + de * e for p, n, e in zip(point, north, east))
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    return math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x)) % 360.0


class FiberGrid:
    """n x n square fibres; fibre 0 bottom-left, rows along +alt, columns along +az."""

    def __init__(self, instrument: dict):
        self.side = int(instrument["grid_side"])
        self.n = int(instrument["n_fibers"])
        # 平台会直接下发几何派生量; 若缺失 (旧协议或本地构造) 则按单格面积与间隙推导,
        # 避免 KeyError 让整场崩溃。
        area = float(instrument.get("fiber_area_deg2", 0.4) or 0.4)
        gap = float(instrument.get("gap_deg", 0.0) or 0.0)
        glass_default = math.sqrt(area)
        pitch_default = glass_default + gap
        self.glass = float(instrument.get("glass_side_deg", glass_default) or glass_default)
        self.pitch = float(instrument.get("pitch_deg", pitch_default) or pitch_default)
        self.fov = float(instrument.get("fov_side_deg", 0.0) or 0.0) or self.side * self.pitch

    def fiber_center(self, fiber: int) -> tuple[float, float]:
        row, col = divmod(fiber, self.side)
        middle = (self.side - 1) / 2.0
        return (row - middle) * self.pitch, (col - middle) * self.pitch

    def classify(self, d_north: float, d_east: float):
        """-> (fiber id, margin in degrees to the glass edge) or (None, negative) when not on glass."""
        half = self.fov / 2.0
        if abs(d_north) > half or abs(d_east) > half:
            return None, -1.0
        middle = self.side / 2.0
        row = min(max(int(math.floor(d_north / self.pitch + middle)), 0), self.side - 1)
        col = min(max(int(math.floor(d_east / self.pitch + middle)), 0), self.side - 1)
        fiber = row * self.side + col
        c_north, c_east = self.fiber_center(fiber)
        margin = self.glass / 2.0 - max(abs(d_north - c_north), abs(d_east - c_east))
        return (fiber, margin) if margin >= 0.0 else (None, margin)


def normalized_airmass(alt_deg: float) -> float:
    if alt_deg <= 0.0:
        return float("inf")
    zenith = 90.0 - alt_deg
    raw = 1.0 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    return raw / (1.0 / (1.0 + 0.50572 * 96.07995 ** -1.6364))


def sun_radec(moment: datetime) -> tuple[float, float]:
    days = julian_date(moment) - 2451545.0
    mean_longitude = (280.460 + 0.9856474 * days) % 360.0
    anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    longitude = math.radians((mean_longitude + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly)) % 360.0)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return (math.degrees(math.atan2(math.cos(obliquity) * math.sin(longitude), math.cos(longitude))) % 360.0,
            math.degrees(math.asin(math.sin(obliquity) * math.sin(longitude))))


def moon_radec(moment: datetime) -> tuple[float, float]:
    days = julian_date(moment) - 2451545.0
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


def separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


class Moon:
    """Moon state at one instant; lunar_factor() is the public lunar model from the score config."""

    def __init__(self, moment: datetime, lst_deg: float, latitude_deg: float, lunar_model: dict):
        self.ra, self.dec = moon_radec(moment)
        sun_ra, sun_dec = sun_radec(moment)
        self.illumination = (1.0 - math.cos(math.radians(separation_deg(sun_ra, sun_dec, self.ra, self.dec)))) / 2.0
        self.alt, _ = radec_to_altaz(self.ra, self.dec, lst_deg, latitude_deg)
        self.model = lunar_model

    def lunar_factor(self, ra_deg: float, dec_deg: float) -> float:
        if self.alt <= 0.0:
            return 1.0
        separation = separation_deg(ra_deg, dec_deg, self.ra, self.dec)
        penalty = (float(self.model["maximum_penalty"]) * self.illumination
                   * math.sin(math.radians(self.alt)) ** float(self.model["altitude_exponent"])
                   * math.exp(-separation / float(self.model["angular_decay_scale_deg"])))
        return max(0.0, min(1.0, 1.0 - penalty))
