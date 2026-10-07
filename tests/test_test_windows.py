from datetime import timedelta
from pro.skymath import parse_utc, format_utc
from tests.test_duty_delivery import setup


def test_flat_test_waits_without_false_report():
    a, payload = setup()
    now = parse_utc(payload['now_utc'])
    end = now + timedelta(minutes=15)
    a._apply_duty({'report_utc': [], 'test_windows_utc': [
        {'start_utc': format_utc(now), 'end_utc': format_utc(end)}]})
    result = a.respond(payload)
    assert result['action'] == 'wait'
    assert result['until_utc'] == format_utc(end)
    assert a.reports == 0


def test_test_window_limits_plan_horizon():
    a, payload = setup()
    now = parse_utc(payload['now_utc'])
    start = now + timedelta(minutes=5)
    a._apply_duty({'report_utc': [], 'test_windows_utc': [
        {'start_utc': format_utc(start), 'end_utc': format_utc(start + timedelta(minutes=15))}]})
    seen = []
    def plan(now, end, *args):
        seen.append(end)
        return None
    a.planner.plan = plan
    a.respond(payload)
    assert seen == [start]


def test_bad_windows_and_out_of_survey_dates_are_rejected():
    a, payload = setup()
    now = parse_utc(payload['now_utc'])
    a._apply_duty({'report_utc': ['2040-01-01T00:00:00Z'], 'test_windows_utc': [
        {'start_utc': format_utc(now), 'end_utc': format_utc(now-timedelta(minutes=1))},
        {'start_utc': format_utc(now), 'end_utc': format_utc(now+timedelta(days=2))},
        {'start_utc': 'bad', 'end_utc': 'bad'}]})
    assert not a.duty_times
    assert not a.duty_test_windows
