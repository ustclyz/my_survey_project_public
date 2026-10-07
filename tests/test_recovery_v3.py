from datetime import timedelta
import io
import json
import pytest
from pro.agent import ObserverAgent
from pro.llm_client import LLMClient
from pro.skymath import parse_utc, format_utc
from tests.test_pro_entry import _messages


def agent():
    init, req = _messages()
    return ObserverAgent(init['payload'], rules_only=True), req['payload']


def test_absolute_fault_does_not_require_saturated_band_samples():
    a, payload = agent()
    a.planner.scale = .05
    a.scale_hours = {0: [.05], 1: [.05], 2: [.05]}
    a.planner.e_hours.clear()
    assert a._fault_verdict(2, payload)


def test_absolute_fault_does_not_use_stale_samples():
    a, payload = agent()
    a.planner.scale = .05
    a.scale_hours = {0: [.05], 1: [.05], 2: [.05]}
    assert not a._scale_fault(100)


def test_explicit_public_notice_works_without_model_response():
    a, payload = agent()
    now = parse_utc(payload['now_utc'])
    a.utc_offset_hours = 0
    # Explicit UTC avoids any dependency on the task-card site timezone.
    payload['new_messages'] = [{'record_type': 'observation_request',
        'request_id': 'maintenance', 'issued_at_utc': payload['now_utc'],
        'reason': f'【祥子】工程组通知：{now.month}/{now.day} {now:%H:%M} UTC 动导星相机，到点请报修。' + '供记录。'*20}]
    assert a.respond(payload)['action'] == 'report'


def test_truncated_model_reply_is_not_executed(monkeypatch):
    c = LLMClient()
    response = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"report_utc": []}'}}]}
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError, match='truncated'):
        c._request('JSON', {}, 3)


def test_fenced_json_with_prose_braces_is_parsed(monkeypatch):
    c = LLMClient()
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Schema {not JSON}. Result:\n```json\n{"report_utc": []}\n```'}}]}
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    assert c._request('JSON', {}, 3) == {'report_utc': []}


def test_absolute_fault_respects_earthquake_hold():
    a, payload = agent()
    a.planner.scale = .05
    a.scale_hours = {0: [.05], 1: [.05], 2: [.05]}
    a.quake_onset_hours = a.quake_last_hours = 0
    assert a._maybe_report(2, payload) is None


def test_no_repeat_probe_from_pre_report_samples():
    a, payload = agent()
    a.planner.scale = .05
    a.scale_hours = {0: [.05], 1: [.05], 2: [.05]}
    a.last_report_hours = 1
    assert not a._scale_fault(2)


def test_ambiguous_model_json_is_rejected(monkeypatch):
    c = LLMClient()
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"report": true} {"report": false}'}}]}
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError, match='ambiguous'):
        c._request('JSON', {}, 3)
