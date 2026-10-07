from datetime import timedelta
from pro.agent import ObserverAgent
from pro.skymath import parse_utc, format_utc
from tests.test_pro_entry import _messages


class Call:
    def done(self):
        return True


class Client:
    def __init__(self, answer=None):
        self.answer = answer
        self.sent = []
        self.accept = True

    def collect(self, call):
        return self.answer

    def submit(self, tag, system, user, left):
        self.sent.append(user)
        return Call() if self.accept else None


def setup():
    init, req = _messages()
    a = ObserverAgent(init['payload'], rules_only=True)
    a.client = Client()
    return a, req['payload']


def test_completed_call_applies_without_new_message_and_counts_report():
    a, payload = setup()
    a.duty_call = Call()
    a.client.answer = {'report_utc': [payload['now_utc']]}
    assert a.respond(payload)['action'] == 'report'
    assert a.reports == 1
    assert a.last_report_hours == 0
    assert not a.client.sent  # no empty prompt every decision


def test_rejected_submission_keeps_every_queued_log():
    a, payload = setup()
    a.client.accept = False
    a.duty_chunks = [str(i) for i in range(6)]
    a._duty_advice(payload)
    assert a.duty_chunks == [str(i) for i in range(6)]
    a.client.accept = True
    a._duty_advice(payload)
    assert '0' in a.client.sent[-1]['duty_log']
    assert a.duty_chunks == ['1', '2', '3', '4', '5']
    for i in range(1, 6):
        a._duty_advice(payload)
        assert a.client.sent[-1]['duty_log'] == str(i)
    assert a.duty_chunks == []


def test_cancellation_removes_previously_scheduled_event():
    a, payload = setup()
    stamp = payload['now_utc']
    a._apply_duty({'report_utc': [stamp]})
    a._apply_duty({'report_utc': [], 'cancelled_utc': [stamp]})
    assert a._due_report(parse_utc(stamp)) is None


def test_wait_and_planner_horizon_stop_at_scheduled_event():
    a, payload = setup()
    now = parse_utc(payload['now_utc'])
    boundary = now + timedelta(seconds=120)
    a.duty_times = [boundary]
    seen = []
    def plan(now, end, *args):
        seen.append(end)
        return None
    a.planner.plan = plan
    result = a.respond(payload)
    assert seen == [boundary]
    assert result['until_utc'] == format_utc(boundary)
    assert 'duration_seconds' not in result


def test_one_repair_consumes_multiple_overdue_notices():
    a, payload = setup()
    now = parse_utc(payload['now_utc'])
    a.duty_times = [now - timedelta(minutes=30), now]
    assert a._due_report(now)['action'] == 'report'
    assert a._due_report(now) is None
