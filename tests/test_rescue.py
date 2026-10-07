import pytest
from pro.planner import Planner
from tests.test_pro_entry import _messages


def planner():
    init, _ = _messages()
    p = Planner(init['payload'], log=lambda _: None)
    p.rescue_mode = True
    return p


def test_repair_restores_quality_and_releases_failed_required_targets():
    p = planner()
    i = next(i for i, flag in enumerate(p.required) if flag)
    p.attempts[i] = 30
    p.scale = .05
    p.req_calib[i] = .3
    p.factor[i] = .2
    p.forget_quality_history()
    assert p.scale == 1
    assert p.attempts[i] == 0
    assert i not in p.req_calib
    assert p.factor[i] == .2  # actual achieved science is retained


def test_completion_duration_reaches_threshold_without_hour_rounding():
    p = planner()
    i = 0
    p.flux[i] = .5
    p.f0t0 = 450
    p.scale = 1
    t = p.rescue_duration(i, 1., 1., 3600)
    assert 600 < t < 750
    assert .5*t*.97/450 >= .675
    assert t % 30 == 0


def test_completion_duration_rejects_impossible_and_setting_targets():
    p = planner()
    p.flux[0] = .0001
    assert p.rescue_duration(0, 1, 1, 3600) is None
    p.flux[0] = .5
    assert p.rescue_duration(0, 1, 1, 300) is None


def test_target_duration_accounts_for_falling_quality():
    p = planner()
    p.flux[0] = .5
    t = p.rescue_duration(0, 1, .8, 3600)
    assert t >= p.rescue_duration(0, 1, 1, 3600)
    assert .5*t*(1-.2*t/3600)*.97/p.f0t0 >= .675


def test_low_quality_preserves_long_exposures_to_avoid_decision_explosion():
    from pro.agent import ObserverAgent
    from pro.skymath import parse_utc
    init, req = _messages()
    a = ObserverAgent(init['payload'], rules_only=True)
    a.planner.rescue_mode = True
    a.planner.scale = a.planner.prior_scale = .05
    seen = []
    def plan(now, end, *args):
        seen.append((end-now).total_seconds())
        return None
    a.planner.plan = plan
    a.respond(req['payload'])
    assert seen and seen[0] > 300


def test_plain_cards_keep_rescue_disabled():
    init, _ = _messages()
    p = Planner(init['payload'], log=lambda _: None)
    assert p.rescue_mode is False
    p.attempts[0] = 4
    p.forget_quality_history()
    assert p.attempts[0] == 4
