"""The model-driven stages of the pro agent. Two calls start at the beginning of every night:

1. night_plan   (natural-language understanding + plan adaptation): reads tonight's forecast and the current
   bulletin and decides whether tonight is a bad night for faint must-observe targets and which compass
   sectors to keep away from. The planner uses both answers for the whole night.
2. fault_review (data parsing + action decision): reads the agent's own hour-by-hour quality table of the last
   nights and judges how likely an unannounced instrument fault is. The answer sets how readily the agent
   reports (probes) a fault tonight.

A third, occasional call confirms a paid fault report before it is sent.

Calls run in the background (llm_client.Call); the agent waits for them only as long as the wall clock allows
and keeps planning otherwise. Every answer is validated; a missing or invalid answer leaves the rule-based
value in place for that night.
"""
from __future__ import annotations

import time

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}

NIGHT_PLAN_SYSTEM = (
    "You plan one night of a robotic spectroscopic survey. The input lists tonight's weather forecast notices and the "
    "current bulletin; each notice is an event kind and a compass sector (N, NE, E, SE, S, SW, W, NW) or ALL (the "
    "whole sky). Decide two things.\n"
    "bad_night: true when tonight's forecast or bulletin has rain, storm, overcast or haze over ALL of the sky. Faint "
    "must-observe targets need a one-hour exposure in a clear sky, so on a bad night they should wait for a better "
    "night.\n"
    "avoid_directions: the sectors with rain, storm, overcast, haze or cold_snap tonight. Ignore earthquake, "
    "rocket_launch and terrain_obstruction (the scheduler handles those itself). Never list a sector nothing names.\n"
    'Reply with one JSON object only: {"bad_night": true|false, "avoid_directions": ["SW", ...], "reason": "<12 words"}'
)

FAULT_REVIEW_SYSTEM = (
    "You watch the data quality of a robotic telescope. An instrument fault is never announced: it lowers the "
    "instrument efficiency, and so the quality of every exposure, until someone reports it; a correct report "
    "repairs it at once. An earthquake (it appears in the bulletin) also lowers instrument efficiency, and that loss "
    "fades night by night; a report does not repair it. Weather lowers quality too, but it also lowers the program "
    "band, which the instrument does not affect.\n"
    "Columns per hour: E = measured quality / quality the program bands allow (about 1 when healthy; low when the "
    "instrument is the cause; in a very clear sky the bands bound it only loosely, so it can stay near 1), scale = "
    "measured sky quality relative to the clear-sky model, ref = the usual scale since the last repair. "
    "notices_now lists the current bulletin.\n"
    "Signs of a fault: quality that drops and stays down without recovering, E low for many hours across nights, "
    "not explained by announced weather or by a recent earthquake whose effect is fading.\n"
    "Reporting: a correct report earns 100 and repairs the instrument; false reports are free while "
    "free_false_reports_left > 0, afterwards each costs 150.\n"
    'Reply with one JSON object only: {"fault_likely": <0..1>, "reason": "<15 words"}'
)

CONFIRM_SYSTEM = (
    "You check the evidence for an unannounced instrument fault on a robotic telescope before a paid report. A false "
    "report costs 150 points; a correct one earns 100 and repairs the instrument. E per hour = measured quality / "
    "quality the program bands allow: about 1 when healthy, low while the instrument is the cause. Weather lowers both "
    "quality and band; an earthquake lowers instrument efficiency in a way that fades night by night and that a "
    "report does not repair.\n"
    'Reply with one JSON object only: {"report": true|false, "reason": "<15 words"}'
)


class Advisor:
    def __init__(self, client, log=lambda text: None):
        self.client = client
        self.log = log
        self.plan_call = None
        self.fault_call = None
        self.plan_applied = True
        self.fault_applied = True
        self.night_date = None
        self.announced: set = set()

    # --- night start ---------------------------------------------------------------------------------

    def start_night(self, night_date: str, tonight: list, bulletin: list, fault_table: dict, wallclock_left: float,
                    wait_seconds: float):
        """Submit both calls; wait up to wait_seconds for them. Returns (plan, fault) answers that are ready."""
        self.night_date = night_date
        self.announced = {n.get("direction") for n in tonight + bulletin if n.get("event_kind") in WEATHER_KINDS}
        notices = {"night": night_date,
                   "forecast_tonight": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in tonight],
                   "bulletin_now": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in bulletin]}
        self.plan_call = self.client.submit("night_plan", NIGHT_PLAN_SYSTEM, notices, wallclock_left)
        self.fault_call = self.client.submit("fault_review", FAULT_REVIEW_SYSTEM, fault_table, wallclock_left)
        self.plan_applied = self.plan_call is None
        self.fault_applied = self.fault_call is None
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for call in (self.plan_call, self.fault_call):
            if call is not None:
                call.wait(deadline - time.monotonic())
        return self.poll()

    def poll(self):
        """(plan, fault) answers that arrived since the last poll; None for each one not (newly) available."""
        plan = fault = None
        if not self.plan_applied and self.plan_call.done():
            self.plan_applied = True
            plan = self._valid_plan(self.client.collect(self.plan_call))
        if not self.fault_applied and self.fault_call.done():
            self.fault_applied = True
            fault = self._valid_fault(self.client.collect(self.fault_call))
        return plan, fault

    def _valid_plan(self, answer):
        if not isinstance(answer, dict) or not isinstance(answer.get("bad_night"), bool):
            return None
        avoid = answer.get("avoid_directions", [])
        if not isinstance(avoid, list):
            return None
        # the model may rank announced weather; it may not close sky that nothing announced
        avoid = sorted({str(d).upper() for d in avoid if str(d).upper() in DIRECTIONS and str(d).upper() in self.announced})
        return {"bad_night": answer["bad_night"], "avoid_directions": avoid, "reason": str(answer.get("reason", ""))[:80]}

    @staticmethod
    def _valid_fault(answer):
        if not isinstance(answer, dict):
            return None
        try:
            p = float(answer.get("fault_likely"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= p <= 1.0:
            return None
        return {"fault_likely": p, "reason": str(answer.get("reason", ""))[:80]}

    # --- paid report confirmation ----------------------------------------------------------------------

    def confirm_report(self, evidence: dict, wallclock_left: float, wait_seconds: float):
        """True / False from the model, or None (no answer in time: the rule decides)."""
        call = self.client.submit("confirm_report", CONFIRM_SYSTEM, evidence, wallclock_left)
        if call is None:
            return None
        call.wait(wait_seconds)
        answer = self.client.collect(call)
        if isinstance(answer, dict) and isinstance(answer.get("report"), bool):
            return answer["report"]
        return None
