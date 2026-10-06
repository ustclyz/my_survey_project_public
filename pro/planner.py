"""Planner of the pro agent: pointing, fibre assignment, duration and program in one search.

Per decision (plan):
1. Visible targets with something left to gain are ranked with a cheap proxy; the best POOL go on.
2. The gain of one exposure of duration T for target i is
       weight * (reach(T) * program multiplier - best so far)
     + REQUIRED_BONUS * P(required target reaches factor 0.5)   (discounted while its sky is far from its best)
     + request bonus  * P(request target reaches its threshold)
   times an urgency factor for targets with few nights left.
3. Candidate fields: the N_ANCHORS best targets centred on each of the 16 fibres, plus the N_DENSE densest
   patches of remaining science. Every fibre gets the target with the largest gain there, for each
   duration; the winner maximises  total gain - lambda * T  (lambda = time price, scaled by the card's
   time scarcity), and is then refined by small pointing shifts.
4. Program: the one with the largest expected score, using a band level fitted to saturated hits.
5. The command is the chosen centre minus the learned pointing offset (Hard-mode cards).

Learning from results: unsaturated hits give the quality level (scale); saturated hits show the program
multiplier exactly and so bracket the band level; E = quality level / band level is the instrument-fault
signal (clean_e); hit/miss patterns reveal the pointing offset. The planner only uses the public catalogue,
the public score formula, bulletins and its own results. Standard library only.
"""
from __future__ import annotations

import bisect
import heapq
import math
import os
from collections import deque
from datetime import datetime, timedelta

from pro.skymath import (
    SIDEREAL_DEG_PER_SECOND,
    FiberGrid,
    Moon,
    local_sidereal_deg,
    max_hour_angle_deg,
    normalized_airmass,
    parse_utc,
    radec_to_altaz,
    shift_altaz,
    tangent_offsets,
    wrap180,
)


def _env(name: str, default):
    return type(default)(os.environ.get(f"PRO_{name}", default))


# --- search -------------------------------------------------------------------------------------------
LAMBDA_FRAC = _env("LAMBDA_FRAC", 0.6)        # price of telescope time, as a share of the recent best gain rate
LAMBDA_EMA = _env("LAMBDA_EMA", 0.03)
SCARCITY_REF = _env("SCARCITY_REF", 0.86)     # tuning constant: scarcity at which time is priced fully
SCARCITY_POWER = _env("SCARCITY_POWER", 1.0)  # time price x min(1, scarcity / SCARCITY_REF) ** power
TYPICAL_Q = 0.6
N_ANCHORS = _env("N_ANCHORS", 12)             # targets tried as field centres per decision (full speed)
N_DENSE = _env("N_DENSE", 20)                  # plus centres in the densest patches of remaining science
DENSE_BIN_DEG = _env("DENSE_BIN_DEG", 2.5)
DENSE_FIBERS = (5, 6, 9, 10)
REFINE = _env("REFINE", 0.1)                 # local pointing search step (deg); 0 = off
REFINE_ROUNDS = _env("REFINE_ROUNDS", 4)
REFINE_FIXED_T = _env("REFINE_FIXED_T", 1)
REFINE_STEPS = tuple((dn * REFINE, de * REFINE) for dn in (-1, 0, 1) for de in (-1, 0, 1) if dn or de) if REFINE > 0 else ()
POOL = _env("POOL", 600)                      # candidates kept after the cheap proxy ranking
NEIGHBOUR_RADIUS_DEG = _env("NEIGHBOUR_RADIUS_DEG", 2.1)
EDGE_MARGIN_DEG = _env("EDGE_MARGIN_DEG", 0.04)   # keep targets this far inside their fibre cell
DURATIONS = (300, 450, 600, 750, 900, 1200, 1500, 1800, 2400, 3000, 3600)
LEVEL_DURATIONS = (DURATIONS, (300, 600, 900, 1200, 1800, 2400, 3600), (450, 900, 1800, 3600), (900, 1800))
MIN_T = _env("MIN_T", 0)                      # shortest exposure considered (unless the night is ending)
MIN_VISIBLE_SECONDS = 600
ALT_MARGIN_DEG = 0.6                          # keep targets this far above the altitude limit
# --- value --------------------------------------------------------------------------------------------
PLAN_FACTOR_SAFETY = _env("PLAN_FACTOR_SAFETY", 0.97)   # plan as if the sky were 3% worse than estimated
REQUIRED_BONUS = _env("REQUIRED_BONUS", 80.0)  # planning value of rescuing one required target (penalty 50)
REQ_P_LO = _env("REQ_P_LO", 0.95)             # P(success) ramps from 0 at this share of the needed reach ...
REQ_P_HI = _env("REQ_P_HI", 1.35)             # ... to 1 at this share
REQUEST_MULT = _env("REQUEST_MULT", 3.0)
REQ_CALIB_POWER = _env("REQ_CALIB_POWER", 0.0)  # per-target calibration after failed tries (0 = off; did not help)
FORECAST_DISCOUNT = _env("FORECAST_DISCOUNT", 0.2)
REQ_CALENDAR = _env("REQ_CALENDAR", 0)        # compare with the best future night (public ephemeris), not an ideal sky
REQ_TIMING = _env("REQ_TIMING", 0.85)         # attempt a required target when its sky model is >= this share of its best ...
REQ_TIMING_NIGHTS = _env("REQ_TIMING_NIGHTS", 5)   # ... unless fewer nights than this are left for it
REQ_TIMING_DISCOUNT = _env("REQ_TIMING_DISCOUNT", 0.3)      # required bonus x this while a better moment will come
URGENCY = _env("URGENCY", 1.5)               # gain x (1 + URGENCY / nights left for the target)
PARTIAL_DISCOUNT = _env("PARTIAL_DISCOUNT", 1.0)   # <1: discount partial exposures of targets the season plan will complete (off: helped on 4 cards, hurt on 8 others)
PARTIAL_DONE = _env("PARTIAL_DONE", 0.9)      # reach below this counts as partial
PARTIAL_NIGHTS = _env("PARTIAL_NIGHTS", 3)    # no discount when fewer nights than this are left for the target
PLAN_UTIL = _env("PLAN_UTIL", 0.55)           # share of fibre-time that ends up useful (for the season plan)
PLAN_Q = _env("PLAN_Q", 0.7)                  # typical quality for the season plan
KAPPA = _env("KAPPA", 1.0)                    # convex shaping of science value (1 = linear)
REQUIRED_SAFE_FACTOR = 0.62                   # a required target counts as safe at this estimated factor
DONE_FACTOR = 0.95                            # other targets are done at this factor
# --- pointing offset (Hard-mode cards: a fixed, unannounced offset; its size is not published) ---
OFFSET_STEPS = 16                             # coarse grid half-width in steps of pitch/12 (about a third of the field);
                                              # the grid widens by half whenever the best offset sits on its edge
OFFSET_MIN_MISSES = 6                         # start estimating after this many assigned-but-missed targets
OFFSET_REFINE_EVERY = 10                      # observes between fine searches
OFFSET_FINE_EVIDENCE = 150                    # observes used by the fine search
OFFSET_MARGIN = 8                             # adopt an offset only if it explains this many more outcomes
# --- learning -----------------------------------------------------------------------------------------
SKY_MEMORY_HOURS = 2.0                        # quality samples older than this are stale
BAND_OPT = _env("BAND_OPT", 1.0)              # declare programs as if the band were this much better (higher programs pay more)
BAND_HALF_LIFE = _env("BAND_HALF_LIFE", 0.0)  # hours; recent saturated hits count more when fitting the band (0 = equal)
BAND_MEMORY_HOURS = 2.0                       # saturated hits used to fit the band level
BAND_FALLBACK = _env("BAND_FALLBACK", 0)      # with few recent saturated hits, fit the last 8 within BAND_FALLBACK_HOURS
BAND_FALLBACK_HOURS = _env("BAND_FALLBACK_HOURS", 12.0)
BAND_CONT = _env("BAND_CONT", 0)              # tie-break the band fit towards the previous fitted level
CLOSED_KINDS = {"rain", "storm"}
SKY_WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}
WEATHER_GATE = _env("WEATHER_GATE", 0)        # no fault evidence from hours with announced all-sky weather
NEUTRAL_KINDS = {"earthquake"}                # announced; per the guide it lowers instrument efficiency, not the sky
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


class Planner:
    def __init__(self, init: dict, log=lambda text: None):
        self.log = log
        site = init["site"]
        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site["minimum_altitude_deg"])
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"])) for n in init["survey"]["nights"]]
        self.survey_end = parse_utc(init["survey"]["end_utc"])
        self.slot_seconds = int(init["survey"]["slot_seconds"])
        instrument = init["instrument"]
        self.grid = FiberGrid(instrument)
        self.min_exposure = int(instrument["exposure"]["min_duration_seconds"])
        self.max_exposure = int(instrument["exposure"]["max_duration_seconds"])
        score = init["scoring"]
        self.f0t0 = float(score["flux_zero_point"]) * float(score["exposure_zero_point_seconds"])
        self.q0 = float(score["q0"])
        self.airmass_exponent = float(score["airmass_exponent"])
        self.bands = score["program"]["bands"]
        self.multipliers = score["program"]["multipliers"]
        self.mismatch = float(score["program"]["mismatch_multiplier"])
        self.lunar_model = score["lunar_model"]

        columns = init["targets"]["columns"]
        col = {name: columns.index(name) for name in columns}
        rows = init["targets"]["rows"]
        self.ids = [row[col["target_id"]] for row in rows]
        self.index_of = {target_id: i for i, target_id in enumerate(self.ids)}
        self.ra = [float(row[col["ra_deg"]]) for row in rows]
        self.dec = [float(row[col["dec_deg"]]) for row in rows]
        self.flux = [float(row[col["feature_flux"]]) for row in rows]
        self.weight = [float(row[col["science_weight"]]) for row in rows]
        self.required = [bool(row[col["required"]]) for row in rows]
        self.sin_dec = [math.sin(math.radians(float(row[col["dec_deg"]]))) for row in rows]
        self.cos_dec = [math.cos(math.radians(float(row[col["dec_deg"]]))) for row in rows]
        self.sin_lat = math.sin(math.radians(self.lat))
        self.cos_lat = math.cos(math.radians(self.lat))
        self.hmax = [max_hour_angle_deg(d, self.lat, self.min_alt + ALT_MARGIN_DEG) for d in self.dec]
        self.factor = [0.0] * len(rows)          # best estimated exposure factor so far
        self.cur = [0.0] * len(rows)             # best realised score / weight so far
        self.rate_ema = 0.0                      # recent best gain rate (the time price follows it)
        self.e_hours: list = []                  # (hour, night, [E samples]): quality level / band level
        self.e_ratios: deque = deque(maxlen=400)   # (hours, quality ratio) of hits outside announced directional events
        self.misses = [0] * len(rows)            # assigned but not hit (e.g. too close to a fibre edge)
        self.vcache = None                       # value(i) per target; None = rebuild all
        self.vdirty: set = set()
        self.attempts = [0] * len(rows)          # required hits that still ended below factor 0.5
        self.active = [i for i in range(len(rows)) if self.hmax[i] > 0.0]
        self.alt_a = [self.sin_lat * s for s in self.sin_dec]
        self.alt_b = [self.cos_lat * c * math.cos(math.radians(r)) for c, r in zip(self.cos_dec, self.ra)]
        self.alt_c = [self.cos_lat * c * math.sin(math.radians(r)) for c, r in zip(self.cos_dec, self.ra)]
        self.sin_alt_limit = math.sin(math.radians(self.min_alt + ALT_MARGIN_DEG))
        self._build_index()
        self._build_windows()
        # best sky model a target can ever get: at transit, Moon down (used to time faint required targets)
        self.ideal_model = [1.0 / (self.q0 * normalized_airmass(max(1.0, 90.0 - abs(d - self.lat))) ** self.airmass_exponent)
                            for d in self.dec]
        self._build_required_calendar()
        # Time scarcity: fibre-seconds the catalogue needs (typical sky) / night seconds on offer. A time-rich
        # season should price telescope time lower than a tight one.
        night_seconds = sum((end - start).total_seconds() for start, end in self.nights)
        need = sum(min(self.max_exposure, self.f0t0 / (max(f, 1e-3) * TYPICAL_Q)) for f in self.flux) / self.grid.n
        self.scarcity = need / max(1.0, night_seconds)
        self.lambda_frac = LAMBDA_FRAC * min(1.0, max(0.2, self.scarcity / SCARCITY_REF) ** SCARCITY_POWER)
        log(f"planner: scarcity {self.scarcity:.2f}, time price fraction {self.lambda_frac:.2f}")

        self.scale = 1.0                         # learned sky quality relative to the clear-sky model
        self.prior_scale = 1.0                   # long-run median, used when recent samples are missing
        self.samples: deque = deque(maxlen=24)   # (hours, ratio) of recent unsaturated hits
        self.all_ratios: deque = deque(maxlen=400)
        self.pending_night = -1
        self.pending_cmd = None
        self.bad_forecast = False                # set by the agent from tonight's forecast
        self.req_calib: dict = {}                # required target -> achieved / predicted on its last failed try
        self.offset = (0.0, 0.0)                 # learned pointing offset (deg): actual = command + offset
        self.offset_evidence: deque = deque(maxlen=400)   # (command, [(alt, az, fiber, hit)]) per observe
        self.offset_scores = None
        self.offset_step = self.grid.pitch / 12.0
        self.offset_steps = OFFSET_STEPS
        self.offset_grid = self._make_offset_grid()
        self.offset_misses = 0
        self.offset_updates = 0
        self.band_level = None
        self.band_obs: deque = deque(maxlen=300)    # (hours, model, program, matched, direction-clean) of saturated hits
        self.pending: dict[str, dict] = {}       # target_id -> prediction for the observe in flight
        self.pending_program = "BACKUP"
        self.pending_duration = 0
        self.blocked: list[tuple[float, float]] = []          # (az, alt) where a hit scored zero
        self.notices: set[tuple[str, str]] = set()
        self.terrain: set[str] = set()
        self.extra_avoid: set[str] = set()       # directions an advisor asked to avoid tonight
        self.duration_scale = 1.0
        self.fast_level = 0
        self.request_bonus: dict[int, float] = {}
        self.request_threshold: dict[int, float] = {}
        # season plan: targets worth completing (value density w*flux above a cut that fills the capacity)
        self.density_order = sorted(range(len(rows)), key=lambda i: -self.weight[i] * self.flux[i])
        self.planned = [False] * len(rows)
        self.plan_night = -1

    # --- precomputation --------------------------------------------------------------------------

    def _build_index(self) -> None:
        self.cells: dict[int, list[tuple[float, int]]] = {}
        for i in self.active:
            self.cells.setdefault(int(math.floor(self.dec[i])), []).append((self.ra[i], i))
        for band in self.cells.values():
            band.sort()
        self.cell_ras = {key: [ra for ra, _ in band] for key, band in self.cells.items()}

    def _build_required_calendar(self) -> None:
        """Per night, the best public sky model (airmass + Moon, no weather) a required target can get.
        Uses only geometry and the lunar ephemeris: when the target is highest during that night's window."""
        self.night_best = {}
        if not REQ_CALENDAR:
            return
        lsts = [(start, local_sidereal_deg(start, self.lon), (end - start).total_seconds()) for start, end in self.nights]
        moons: dict = {}
        for i in range(len(self.ids)):
            if not self.required[i] or self.hmax[i] <= 0.0:
                continue
            row = []
            for k, (start, l0, span) in enumerate(lsts):
                # seconds after night start when hour angle is closest to 0
                ha0 = wrap180(l0 - self.ra[i])
                t = min(max(-ha0 / SIDEREAL_DEG_PER_SECOND, 0.0), span)
                ha = wrap180(ha0 + t * SIDEREAL_DEG_PER_SECOND)
                if abs(ha) > self.hmax[i]:
                    row.append(0.0)
                    continue
                key = (k, int(t // 1800))
                moon = moons.get(key)
                if moon is None:
                    moment = start + timedelta(seconds=(key[1] + 0.5) * 1800)
                    moon = moons[key] = Moon(moment, local_sidereal_deg(moment, self.lon), self.lat, self.lunar_model)
                alt, _ = radec_to_altaz(self.ra[i], self.dec[i], l0 + t * SIDEREAL_DEG_PER_SECOND, self.lat)
                row.append(moon.lunar_factor(self.ra[i], self.dec[i])
                           / (self.q0 * normalized_airmass(max(alt, 1.0)) ** self.airmass_exponent))
            self.night_best[i] = row

    def best_future_model(self, i: int, night_index: int) -> float:
        row = self.night_best.get(i)
        if not row or night_index + 1 >= len(row):
            return 0.0
        return max(row[night_index + 1:])

    def neighbours(self, ra: float, dec: float, radius: float) -> list[int]:
        found = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = radius / cos_dec
        for key in range(int(math.floor(dec - radius)), int(math.floor(dec + radius)) + 1):
            band = self.cells.get(key)
            if not band:
                continue
            ras = self.cell_ras[key]
            spans = [(ra - width, ra + width)]
            if spans[0][0] < 0:
                spans = [(0.0, spans[0][1]), (spans[0][0] + 360.0, 360.0)]
            elif spans[0][1] >= 360:
                spans = [(spans[0][0], 360.0), (0.0, spans[0][1] - 360.0)]
            for low, high in spans:
                for k in range(bisect.bisect_left(ras, low), bisect.bisect_right(ras, high)):
                    found.append(band[k][1])
        return found

    def _build_windows(self) -> None:
        """First and last night on which each target has at least 20 minutes above the limit."""
        need = 20 * 60 * SIDEREAL_DEG_PER_SECOND
        spans = [(local_sidereal_deg(start, self.lon), (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND)
                 for start, end in self.nights]
        self.first_night = [len(self.nights)] * len(self.ra)
        self.last_night = [-1] * len(self.ra)
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = (self.ra[i] - h - l0) % 360.0
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if self.first_night[i] > k:
                        self.first_night[i] = k
                    self.last_night[i] = k

    # --- messages and results --------------------------------------------------------------------

    def on_messages(self, messages: list, latest_bulletin) -> None:
        for message in messages:
            kind = message.get("record_type")
            if kind == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        self.terrain.add(notice.get("direction", ""))
            elif kind == "state_resync":
                self._resync(message)
        bulletin = latest_bulletin or {}
        self.notices = {(n.get("event_kind", ""), n.get("direction", "")) for n in bulletin.get("notices", [])
                        if n.get("event_kind") != "terrain_obstruction"}

    def on_requests(self, requests: list) -> None:
        """Turn the current all-or-nothing request rewards into per-target planning values."""
        old = self.request_bonus
        self.request_bonus = {}
        self.request_threshold = {}
        for request in requests:
            # A request that already met its minimum still appears until its deadline
            # with remaining_count 0; its reward is settled, so it adds no value.
            remaining = int(request.get("remaining_count", request["minimum_completed"]))
            if remaining <= 0:
                continue
            completed = set(request.get("completed_target_ids", []))
            unit = REQUEST_MULT * float(request["completion_reward"]) / remaining
            threshold = float(request["completion_factor_threshold"])
            for target_id in request["target_ids"]:
                if target_id in completed or target_id not in self.index_of:
                    continue
                i = self.index_of[target_id]
                # Overlapping requests: the marginal rewards add up; the combined gain is
                # only collectible at the highest threshold of the contributing requests.
                self.request_bonus[i] = self.request_bonus.get(i, 0.0) + unit
                self.request_threshold[i] = max(self.request_threshold.get(i, 0.0), threshold)
                if self.hmax[i] > 0.0 and i not in self.active:
                    self.active.append(i)
        if old != self.request_bonus:
            self.vdirty.update(old)
            self.vdirty.update(self.request_bonus)

    def _resync(self, message: dict) -> None:
        """Part of the recent data was lost: restart the factor estimates from the engine's best scores."""
        best = {row["target_id"]: float(row["best_score"]) for row in message.get("best_scores", [])}
        top = max(self.multipliers.values())
        for i, target_id in enumerate(self.ids):
            score = best.get(target_id, 0.0)
            self.factor[i] = min(1.0, score / (self.weight[i] * top)) if score > 0 else 0.0
            self.cur[i] = score / self.weight[i]
        self.active = [i for i in range(len(self.ids)) if self.hmax[i] > 0.0]
        self.vcache = None
        self.pending = {}
        self.log(f"state_resync: {len(best)} targets keep a score; plan rebuilt")

    def site_closed(self) -> bool:
        return any(kind in CLOSED_KINDS and direction == "ALL" for kind, direction in self.notices)

    def all_sky_weather(self) -> bool:
        return any(direction == "ALL" and kind in SKY_WEATHER_KINDS for kind, direction in self.notices)

    def all_sky_notice(self) -> bool:
        return any(direction == "ALL" and kind not in NEUTRAL_KINDS for kind, direction in self.notices)

    def on_result(self, result, now: datetime, hours: float) -> None:
        """Update factor estimates and the sky-quality estimate from the previous observe."""
        if not result or result.get("action") != "observe" or not self.pending:
            self.pending = {}
            return
        hits = {hit["target_id"]: float(hit["score"]) for hit in result.get("hits", [])}
        any_positive = any(score > 0 for score in hits.values())
        declared = self.multipliers[self.pending_program]
        for target_id, prediction in self.pending.items():
            i = self.index_of[target_id]
            self.vdirty.add(i)
            score = hits.get(target_id)
            if score is None:
                self.misses[i] += 1  # a miss: the target did not land on its fibre glass
                continue
            if score <= 0.0:
                if any_positive:
                    self.blocked.append((prediction["az"], prediction["alt"]))
                continue
            # score = weight * factor * multiplier; the multiplier is `declared` if the program matched.
            # A saturated hit (factor = 1) shows the multiplier exactly, so it tells whether the sky's
            # program band matched the declared program. Instrument efficiency does not enter the band.
            multiplier_seen = score / self.weight[i]
            self.cur[i] = max(self.cur[i], multiplier_seen)
            if abs(multiplier_seen - declared) < 2e-4:
                self.band_obs.append((hours, prediction["model"], self.pending_program, True, prediction["dir_clean"]))
            elif abs(multiplier_seen - self.mismatch) < 2e-4:
                self.band_obs.append((hours, prediction["model"], self.pending_program, False, prediction["dir_clean"]))
            factor_if_match = score / (self.weight[i] * declared)
            factor_if_miss = score / (self.weight[i] * self.mismatch)
            ratio_match = factor_if_match * self.f0t0 / (self.flux[i] * self.pending_duration * prediction["model"])
            matched = self._band(ratio_match * prediction["band_model"]) == self.pending_program
            factor = factor_if_match if matched else factor_if_miss
            self.factor[i] = max(self.factor[i], min(1.0, factor))
            if self.required[i] and self.factor[i] < 0.5:
                pred = prediction.get("pred", 0.0)
                if pred > 0 and factor < 0.97:
                    self.req_calib[i] = min(1.0, max(0.3, factor / pred)) ** REQ_CALIB_POWER
                self.attempts[i] += 1  # not enough yet: lower its priority a little for next time
            if factor < 0.97:
                ratio = factor * self.f0t0 / (self.flux[i] * self.pending_duration * prediction["model"])
                self.samples.append((hours, ratio))
                self.all_ratios.append(ratio)
                if prediction["dir_clean"]:
                    self.e_ratios.append((hours, ratio))
        self._offset_evidence(hits)
        self.pending = {}
        self.update_scale(hours)

    # --- pointing offset (Hard-mode cards) ----------------------------------------------------------------

    def _offset_evidence(self, hits: dict) -> None:
        """Hard-mode cards add a hidden fixed offset to every pointing (participant guide: actual = command +
        (d_alt, d_az)). Each assigned target's hit or miss is evidence; keep a score for each candidate offset
        on a grid and adopt the best one once it clearly explains the misses better than no offset."""
        if not self.pending or self.pending_cmd is None:
            return
        rows = [(p["alt"], p["az"], p["fiber"], target_id in hits) for target_id, p in self.pending.items() if "fiber" in p]
        if not rows:
            return
        missed = sum(1 for row in rows if not row[3])
        self.offset_misses += missed
        self.offset_evidence.append((self.pending_cmd, rows))
        if self.offset_misses < OFFSET_MIN_MISSES:
            return
        if self.offset_scores is None:
            self._rescore_offsets(list(self.offset_evidence))
        else:
            self._score_offsets(self.pending_cmd, rows)
        self.offset_updates += 1
        if self.offset_updates % OFFSET_REFINE_EVERY == 1 or OFFSET_REFINE_EVERY <= 1:
            self._refine_offset()

    def _consistent(self, cmd, rows, d_alt, d_az) -> int:
        c_alt, c_az = cmd[0] + d_alt, (cmd[1] + d_az) % 360.0
        ok = 0
        for t_alt, t_az, fiber, hit in rows:
            offsets = tangent_offsets(t_alt, t_az, c_alt, c_az)
            fib = self.grid.classify(*offsets)[0] if offsets is not None else None
            ok += (fib == fiber) == hit
        return ok

    def _make_offset_grid(self) -> list:
        n, step = self.offset_steps, self.offset_step
        return [(step * i, step * j) for i in range(-n, n + 1) for j in range(-n, n + 1)]

    def _rescore_offsets(self, evidence) -> None:
        self.offset_scores = [0] * len(self.offset_grid)
        for cmd, past in evidence:
            self._score_offsets(cmd, past)

    def _score_offsets(self, cmd, rows) -> None:
        for k, (d_alt, d_az) in enumerate(self.offset_grid):
            self.offset_scores[k] += self._consistent(cmd, rows, d_alt, d_az)

    def _refine_offset(self) -> None:
        k = max(range(len(self.offset_grid)), key=lambda n: self.offset_scores[n])
        base_alt, base_az = self.offset_grid[k]
        n = self.offset_steps
        edge = max(abs(base_alt), abs(base_az)) >= (n - 0.5) * self.offset_step
        if edge and (n + 1) * self.offset_step < self.grid.fov / 2.0:
            # the best candidate sits on the edge of the grid: the offset may be larger, widen the search
            self.offset_steps = int(n * 1.5) + 1
            self.offset_grid = self._make_offset_grid()
            self._rescore_offsets(list(self.offset_evidence)[-100:])
            self.log(f"pointing offset: search widened to +-{self.offset_steps * self.offset_step:.2f} deg")
            k = max(range(len(self.offset_grid)), key=lambda m: self.offset_scores[m])
            base_alt, base_az = self.offset_grid[k]
        evidence = list(self.offset_evidence)[-OFFSET_FINE_EVIDENCE:]
        zero = sum(self._consistent(cmd, rows, 0.0, 0.0) for cmd, rows in evidence)
        scored = []
        fine = self.offset_step / 5.0
        for i in range(-6, 7):
            for j in range(-6, 7):
                d_alt, d_az = base_alt + fine * i, base_az + fine * j
                scored.append((sum(self._consistent(cmd, rows, d_alt, d_az) for cmd, rows in evidence), d_alt, d_az))
        top = max(score for score, _, _ in scored)
        if top - zero < OFFSET_MARGIN:
            return
        # several offsets often explain the evidence equally well: take the centre of that set
        tied = [(d_alt, d_az) for score, d_alt, d_az in scored if score == top]
        centre = (round(sum(a for a, _ in tied) / len(tied), 3), round(sum(z for _, z in tied) / len(tied), 3))
        if abs(centre[0] - self.offset[0]) + abs(centre[1] - self.offset[1]) < 0.005:
            return
        total = sum(len(rows) for _, rows in evidence)
        self.log(f"pointing offset: alt {centre[0]:+.3f} az {centre[1]:+.3f} deg explains {top}/{total} "
                 f"fibre outcomes (no offset: {zero}; {len(tied)} equally good grid points)")
        if self.offset == (0.0, 0.0):
            self.misses = [0] * len(self.ids)   # the misses were the offset, not the targets
            self.vcache = None
        self.offset = centre

    def calibrated_band_scale(self, hours: float) -> float:
        """Sky level for the program band, fitted to recent saturated hits (they show the multiplier exactly).
        Falls back to the quality-based estimate; an unreported instrument fault lowers quality but not the band."""
        guess = self.scale / 0.95
        recent = [item for item in self.band_obs if item[0] >= hours - BAND_MEMORY_HOURS]
        if len(recent) < 4 and BAND_FALLBACK:
            # few saturated hits lately (poor quality, or an instrument fault): the band does not follow the
            # instrument, so keep fitting the latest saturated hits instead of the quality level
            recent = [item for item in self.band_obs if item[0] >= hours - BAND_FALLBACK_HOURS][-8:]
        if len(recent) < 4:
            return guess
        # among equally consistent levels prefer the one closest to the last fitted level (the sky band moves
        # with the weather, not with the instrument), else to the quality-based guess
        centre = self.band_level if (BAND_CONT and self.band_level) else guess
        best = None
        if BAND_HALF_LIFE > 0:
            weights = [0.5 ** ((hours - h) / BAND_HALF_LIFE) for h, _, _, _, _ in recent]
        else:
            weights = [1.0] * len(recent)
        for step in range(-25, 31):
            w = centre * (1.06 ** step)
            ok = sum(wt for wt, (_, model, program, matched, _) in zip(weights, recent) if (self._band(model * w) == program) == matched)
            key = (round(ok, 6), -abs(step))
            if best is None or key > best[0]:
                best = (key, w)
        self.band_level = best[1]
        return best[1]

    def clean_e(self, hours: float):
        """E = clean-sky quality level / clean-sky band level over the last BAND_MEMORY_HOURS (1 = consistent)."""
        ratios = sorted(r for h, r in self.e_ratios if h >= hours - BAND_MEMORY_HOURS)
        obs = [item for item in self.band_obs if item[0] >= hours - BAND_MEMORY_HOURS and item[4]]
        if len(ratios) < 4 or len(obs) < 4:
            return None
        guess = ratios[len(ratios) // 2] / 0.95
        best = None
        for step in range(-20, 31):
            w = guess * (1.06 ** step)
            ok = sum(1 for _, model, program, matched, _ in obs if (self._band(model * w) == program) == matched)
            key = (ok, -abs(step))
            if best is None or key > best[0]:
                best = (key, w)
        return min(1.0, guess / best[1])

    def update_scale(self, hours: float) -> None:
        """Sky quality now = median of recent samples; fall back to the long-run median when stale."""
        if len(self.all_ratios) >= 8:
            ordered = sorted(self.all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self.samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if len(recent) >= 4 else self.prior_scale

    def _band(self, q_band: float) -> str:
        if q_band >= float(self.bands["DARK"]):
            return "DARK"
        if q_band >= float(self.bands["BRIGHT"]):
            return "BRIGHT"
        return "BACKUP"

    # --- anomaly check ---------------------------------------------------------------------------

    def forget_quality_history(self) -> None:
        """After a correct report the instrument is repaired: start the quality estimates afresh."""
        self.e_hours = []
        self.e_ratios.clear()
        self.samples.clear()
        self.all_ratios.clear()
        self.prior_scale = 1.0

    # --- planning --------------------------------------------------------------------------------

    def current_night(self, now: datetime):
        for k, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return k, start, end
        return None

    def next_night_start(self, now: datetime):
        for start, _ in self.nights:
            if start > now:
                return start
        return None

    def _direction_factor(self, alt: float, az: float) -> float:
        factor = 1.0
        for direction in self.terrain:
            if direction in DIRECTION_AZ and alt < 50.0 and _az_distance(az, DIRECTION_AZ[direction]) <= 60.0:
                return 0.0
        for kind, direction in self.notices:
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, 0.35)
        for direction in self.extra_avoid:
            if direction in DIRECTION_AZ and _az_distance(az, DIRECTION_AZ[direction]) <= 67.5 and alt < 70.0:
                factor = min(factor, 0.35)
        for blocked_az, blocked_alt in self.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12.0 and alt <= blocked_alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def value(self, i: int) -> float:
        f = self.factor[i]
        damp = 0.6 ** self.misses[i]
        request = self.request_bonus.get(i, 0.0)
        if self.required[i]:
            if f >= REQUIRED_SAFE_FACTOR:
                return (self.weight[i] * max(0.0, 1.0 - f * f) + request) * damp
            return (self.weight[i] * (1.0 - f * f) + REQUIRED_BONUS * (1.0 if f < 0.5 else 0.35) + request) * damp
        return ((0.0 if f >= DONE_FACTOR else self.weight[i] * (1.0 - f * f)) + request) * damp

    def _season_plan(self, now: datetime, night_index: int) -> None:
        """Which targets will the season complete? Fill the remaining useful fibre-time with targets in
        order of value density (w * flux); the rest are fillers whose partial exposures are worth taking."""
        self.plan_night = night_index
        cap = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in self.nights if end > now)
        cap *= self.grid.n * PLAN_UTIL
        self.planned = [False] * len(self.ids)
        n = 0
        for i in self.density_order:
            if cap <= 0:
                break
            if self.hmax[i] <= 0.0 or self.factor[i] >= DONE_FACTOR:
                continue
            cap -= min(self.max_exposure, self.f0t0 / (max(self.flux[i], 1e-3) * PLAN_Q))
            self.planned[i] = True
            n += 1
        self.log(f"season plan: {n} targets to complete")

    def plan(self, now: datetime, night_end: datetime, night_index: int, hours: float):
        """Return an observe action dict, or None when nothing useful is up."""
        if PARTIAL_DISCOUNT < 1.0 and night_index != self.plan_night:
            self._season_plan(now, night_index)
        self.update_scale(hours)
        self.night_index = night_index
        lst = local_sidereal_deg(now, self.lon)
        horizon = min(night_end, self.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < self.min_exposure:
            return None
        lst_later = lst + 1800.0 * SIDEREAL_DEG_PER_SECOND
        moon = Moon(now + timedelta(seconds=600), lst, self.lat, self.lunar_model)
        scale = self.scale * PLAN_FACTOR_SAFETY
        band_scale = self.calibrated_band_scale(hours) * BAND_OPT
        e_now = self.clean_e(hours)
        if e_now is not None and not (WEATHER_GATE and self.all_sky_weather()):
            hour = int(hours)
            if not self.e_hours or self.e_hours[-1][0] != hour:
                self.e_hours.append((hour, night_index, []))
            self.e_hours[-1][2].append(e_now)
        min_up = min(MIN_VISIBLE_SECONDS, seconds_left)
        level = self.fast_level
        base = {}
        visible = {}
        still_active = []
        min_up_deg = min_up * SIDEREAL_DEG_PER_SECOND
        sl, cl = self.sin_lat, self.cos_lat
        sin_min = math.sin(math.radians(self.min_alt))
        pool_size = (POOL, POOL // 2, POOL // 4, POOL // 8, POOL // 8)[min(level, 4)]
        proxy = []
        bins: dict = {}
        bin_best: dict = {}
        sd, cd = self.sin_dec, self.cos_dec
        if self.vcache is None:
            self.vcache = [self.value(i) for i in range(len(self.ids))]
        else:
            for i in self.vdirty:
                self.vcache[i] = self.value(i)
        self.vdirty = set()
        vc = self.vcache
        hmax, ra = self.hmax, self.ra
        # sin(alt) = A + B cos(LST) + C sin(LST): no trigonometry per target
        pa, pb, pc = self.alt_a, self.alt_b, self.alt_c
        c1, s1 = math.cos(math.radians(lst)), math.sin(math.radians(lst))
        c2, s2 = math.cos(math.radians(lst + min_up_deg)), math.sin(math.radians(lst + min_up_deg))
        sin_lim = self.sin_alt_limit
        for i in self.active:
            a = pa[i]
            sin_alt = a + pb[i] * c1 + pc[i] * s1
            if sin_alt < sin_lim or a + pb[i] * c2 + pc[i] * s2 < sin_lim:
                still_active.append(i)   # not up now (or setting soon): keep it, check its value when it rises
                continue
            v = vc[i]
            if v <= 0.0:
                continue
            still_active.append(i)
            visible[i] = None
            # proxy priority: planning value x a rough airmass factor x urgency (no Moon, no direction)
            nights_left = self.last_night[i] - night_index + 1
            proxy.append((v * (sin_alt ** 0.6) * (1.0 + URGENCY / (nights_left if nights_left > 1 else 1)), i))
            if N_DENSE:
                # plain science still to gain here, binned on the sky at roughly one field size
                key = (int((self.ra[i] * cd[i]) // DENSE_BIN_DEG), int((self.dec[i] + 90.0) // DENSE_BIN_DEG))
                dense_val = self.weight[i] * max(0.0, 1.0 - self.cur[i] / 1.2) * (sin_alt ** 0.6)
                bins[key] = bins.get(key, 0.0) + dense_val
                if dense_val > bin_best.get(key, (0.0, -1))[0]:
                    bin_best[key] = (dense_val, i)
        self.active = still_active
        if not visible:
            return None

        def exact(i):
            item = base.get(i)
            if item is None and i not in base:
                ha = (lst - self.ra[i] + 180.0) % 360.0 - 180.0
                alt, az = radec_to_altaz(self.ra[i], self.dec[i], lst, self.lat)
                dirf = self._direction_factor(alt, az)
                if alt < self.min_alt or dirf <= 0.0:
                    base[i] = None
                    return None
                h = self.hmax[i]
                up = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180.0 else 1e9
                lunar = moon.lunar_factor(self.ra[i], self.dec[i])
                nights_left = max(1, self.last_night[i] - night_index + 1)
                damp = 0.6 ** self.misses[i] * 0.8 ** self.attempts[i]
                item = base[i] = (alt, az, lunar, up, (1.0 + URGENCY / nights_left) * damp * dirf)
            return item

        pool = [i for _, i in (heapq.nlargest(pool_size, proxy) if len(proxy) > pool_size else proxy)]
        for i in pool:
            exact(i)
        pool = [i for i in pool if base[i] is not None]
        if not pool:
            return None
        info = {}

        def full(i):
            item = info.get(i)
            if item is None:
                alt, az, lunar, up, mult = base[i]
                alt2, _ = radec_to_altaz(self.ra[i], self.dec[i], lst_later, self.lat)
                m0 = lunar / (self.q0 * normalized_airmass(max(alt, 1.0)) ** self.airmass_exponent)
                m1 = lunar / (self.q0 * normalized_airmass(max(alt2, 1.0)) ** self.airmass_exponent)
                item = info[i] = (alt, az, m0, m1, up, mult)
            return item

        top_mult = max(self.multipliers.values())

        def shaped(v):
            """Convex value of score/weight v: finishing a target beats half-doing it twice, because only
            its best exposure counts (a partial exposure is wasted if the target is redone later)."""
            return top_mult * (v / top_mult) ** KAPPA if KAPPA != 1.0 else v

        def gain(i, T):
            alt, az, m0, m1, up, mult = full(i)
            if up < T:
                return 0.0
            model = m0 + (m1 - m0) * min(1.0, T / 3600.0)
            reach = min(1.0, self.flux[i] * T * model * scale / self.f0t0)
            m = self.multipliers[self._band(model * band_scale)]
            g = self.weight[i] * max(0.0, shaped(reach * m) - shaped(self.cur[i]))
            if reach < PARTIAL_DONE and self.planned[i] and self.last_night[i] - night_index >= PARTIAL_NIGHTS:
                g *= PARTIAL_DISCOUNT   # it will be completed later: this partial exposure would be wasted
            if self.required[i] and self.factor[i] < 0.5:
                raw = self.flux[i] * T * model * self.scale / self.f0t0 / 0.5 * self.req_calib.get(i, 1.0)
                bonus = REQUIRED_BONUS * min(1.0, max(0.0, (raw - REQ_P_LO) / (REQ_P_HI - REQ_P_LO)))
                target_best = self.best_future_model(i, night_index) if REQ_CALENDAR else self.ideal_model[i]
                if REQ_TIMING and model < REQ_TIMING * target_best and self.last_night[i] - night_index >= REQ_TIMING_NIGHTS:
                    bonus *= REQ_TIMING_DISCOUNT   # a better moment for this target will come
                if self.bad_forecast and self.last_night[i] - night_index >= REQ_TIMING_NIGHTS:
                    bonus *= FORECAST_DISCOUNT     # tonight is forecast bad over the whole sky
                g += bonus
            if i in self.request_bonus:
                threshold = self.request_threshold.get(i, 0.5)
                raw = self.flux[i] * T * model * self.scale / self.f0t0 / max(1e-6, threshold)
                g += self.request_bonus[i] * min(1.0, max(0.0, (raw - REQ_P_LO) / (REQ_P_HI - REQ_P_LO)))
            return g * mult

        def quick(i, T):
            """gain() with the start-of-exposure sky model only (no second alt/az): for ranking."""
            alt, az, lunar, up, mult = base[i]
            T = min(T, up)
            if T < self.min_exposure:
                return 0.0, T
            model = lunar / (self.q0 * normalized_airmass(max(alt, 1.0)) ** self.airmass_exponent)
            reach = min(1.0, self.flux[i] * T * model * scale / self.f0t0)
            g = self.weight[i] * max(0.0, shaped(reach * self.multipliers[self._band(model * band_scale)]) - shaped(self.cur[i]))
            if reach < PARTIAL_DONE and self.planned[i] and self.last_night[i] - night_index >= PARTIAL_NIGHTS:
                g *= PARTIAL_DISCOUNT
            if self.required[i] and self.factor[i] < 0.5:
                raw = self.flux[i] * T * model * self.scale / self.f0t0 / 0.5
                g += REQUIRED_BONUS * min(1.0, max(0.0, (raw - REQ_P_LO) / (REQ_P_HI - REQ_P_LO)))
            if i in self.request_bonus:
                raw = self.flux[i] * T * model * self.scale / self.f0t0 / max(1e-6, self.request_threshold.get(i, 0.5))
                g += self.request_bonus[i] * min(1.0, max(0.0, (raw - REQ_P_LO) / (REQ_P_HI - REQ_P_LO)))
            return g * mult, T

        durations = [T for T in LEVEL_DURATIONS[min(level, 3)] if T <= seconds_left and T >= MIN_T]
        if not durations:
            durations = [int(seconds_left)]
        t_long = durations[-1]
        t_mid = durations[len(durations) // 2]
        lam = self.lambda_frac * self.rate_ema
        ranked = []
        for i in pool:
            g1, T1 = quick(i, t_long)
            g2, T2 = quick(i, t_mid)
            best_net = max(g1 - lam * T1 / 16.0, g2 - lam * T2 / 16.0)
            if best_net > 0:
                ranked.append((best_net, i))
        if not ranked:
            self.rate_ema *= 0.9
            lam = 0.0
            ranked = [(quick(i, t_long)[0], i) for i in pool]
            ranked = [item for item in ranked if item[0] > 0]
        if not ranked:
            return None
        ranked.sort(reverse=True)
        n_anchors = (N_ANCHORS, 3, 1, 1)[min(level, 3)]
        anchors = [i for _, i in ranked[:n_anchors]]
        if N_DENSE and level <= 1 and bins:
            # also try the densest patches of remaining science: fields with no single outstanding target
            for _, key in heapq.nlargest(N_DENSE if level == 0 else 2, ((v, k) for k, v in bins.items())):
                j = bin_best[key][1]
                if j not in anchors and exact(j) is not None:
                    anchors.append(j)
        fibers = (range(self.grid.n), range(self.grid.n), (5, 6, 9, 10), (5,))[min(level, 3)]
        best = None
        best_rate_here = 0.0
        best_rate = [0.0]

        def evaluate(c_alt, c_az, near, durations):
            """Best (net, T, pick, total) for one pointing, or None."""
            cells: dict = {}
            for j in near:
                offsets = tangent_offsets(base[j][0], base[j][1], c_alt, c_az)
                if offsets is None:
                    continue
                fib, margin = self.grid.classify(*offsets)
                if fib is None or margin < EDGE_MARGIN_DEG * (0.5 + 1.5 * self.misses[j]):
                    continue
                cells.setdefault(fib, []).append(j)
            if not cells:
                return None
            found = None
            for T in durations:
                total = 0.0
                pick = {}
                for fib, js in cells.items():
                    g, j = max((gain(j, T), j) for j in js)
                    if g > 0:
                        total += g
                        pick[fib] = j
                if not pick:
                    continue
                if total / T > best_rate[0]:
                    best_rate[0] = total / T
                net = total - lam * T
                if found is None or net > found[0]:
                    found = (net, T, pick, total)
            return found

        best_near = None
        n_value_anchors = min(len(anchors), n_anchors)
        for rank, anchor in enumerate(anchors):
            a_alt, a_az = base[anchor][0], base[anchor][1]
            near = [j for j in self.neighbours(self.ra[anchor], self.dec[anchor], NEIGHBOUR_RADIUS_DEG) if j in visible and exact(j) is not None]
            # density anchors mark a patch, not a target to centre: a few central placements, then refine
            for fiber in (fibers if rank < n_value_anchors else DENSE_FIBERS):
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not self.min_alt <= c_alt <= 89.0:
                    continue
                c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360.0
                found = evaluate(c_alt, c_az, near, durations)
                if found is not None and (best is None or found[0] > best[0]):
                    best = (found[0], c_alt, c_az, found[1], found[2], found[3])
                    best_near = near
        if best is not None and REFINE_STEPS and level == 0:
            # local search: nudge the winning pointing to catch targets near the cell edges
            for _ in range(REFINE_ROUNDS):
                improved = False
                b_alt, b_az = best[1], best[2]
                for d_north, d_east in REFINE_STEPS:
                    c_alt, c_az = shift_altaz(b_alt, b_az, d_north, d_east)
                    if not self.min_alt <= c_alt <= 89.0:
                        continue
                    c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360.0
                    found = evaluate(c_alt, c_az, best_near, (best[3],) if REFINE_FIXED_T else durations)
                    if found is not None and found[0] > best[0] + 1e-9:
                        best = (found[0], c_alt, c_az, found[1], found[2], found[3])
                        improved = True
                if not improved:
                    break
        best_rate_here = best_rate[0]
        self.rate_ema = (1 - LAMBDA_EMA) * self.rate_ema + LAMBDA_EMA * best_rate_here if self.rate_ema > 0 else best_rate_here
        if best is None or best[5] <= 0:
            if int(hours) != getattr(self, "_dbgnone", None):
                self._dbgnone = int(hours)
                self.log(f"plan none: info={len(info)} ranked={len(ranked)} scale={self.scale:.3f} lam={lam:.4f} best={best and best[:1]}")
            return None
        _, c_alt, c_az, T, pick, _ = best
        # program: maximise expected score over the assigned targets
        votes = {"DARK": 0.0, "BRIGHT": 0.0, "BACKUP": 0.0}
        for fib, j in pick.items():
            alt, az, m0, m1, up, _ = info[j]
            model = m0 + (m1 - m0) * min(1.0, T / 3600.0)
            reach = min(1.0, self.flux[j] * T * model * scale / self.f0t0)
            votes[self._band(model * band_scale)] += self.weight[j] * reach + (0.05 * REQUIRED_BONUS if self.required[j] else 0.0)
        total_votes = sum(votes.values())
        program = max(votes, key=lambda p: (votes[p] * self.multipliers[p] + (total_votes - votes[p]) * self.mismatch, p))
        clean = not self.all_sky_notice()
        self.pending = {}
        for fib, j in pick.items():
            alt, az, m0, m1, up, _ = info[j]
            model = m0 + (m1 - m0) * min(1.0, T / 3600.0)
            dir_clean = self._direction_factor(alt, az) >= 1.0
            self.pending[self.ids[j]] = {"model": model, "band_model": model / 0.95, "alt": alt, "az": az,
                                        "pred": self.flux[j] * T * model * self.scale / self.f0t0,
                                        "clean": clean and dir_clean, "dir_clean": dir_clean}
        self.pending_program = program
        self.pending_duration = T
        self.pending_night = night_index
        for fib, j in pick.items():
            self.pending[self.ids[j]]["fiber"] = fib
        # the mount lands at command + offset: command the desired centre minus the learned offset
        cmd_alt = round(min(90.0, max(0.0, c_alt - self.offset[0])), 4)
        cmd_az = round((c_az - self.offset[1]) % 360.0, 4)
        if cmd_az >= 360.0:
            cmd_az = 0.0
        self.pending_cmd = (cmd_alt, cmd_az)
        return {"action": "observe", "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
                "assignments": {str(fib): self.ids[j] for fib, j in sorted(pick.items())},
                "duration_seconds": int(T), "program": program}
