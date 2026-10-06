#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""协议解析 / 动作构造单测.

验证 ``protocol.py``:
    * 能解析 initialize / decision_request 样例;
    * 能构造合法的 observe / wait / report / finish 动作;
    * 输出是合法 JSON 且字段裁剪正确 (不含未知字段);
    * decision_sequence 正确回填.

运行: ``py -m pytest tests/test_protocol.py -q``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import protocol  # noqa: E402
from models import Action  # noqa: E402

INIT_MESSAGE = {
    "protocol_version": "participant-agent-protocol-v4",
    "message_type": "initialize",
    "payload": {
        "schema_version": "v4-initialize-v1",
        "task_card": {"card_id": "alpha", "scenario_slug": "v4-practice-alpha", "phase": "local"},
        "site": {"latitude_deg": -24.6157, "longitude_deg": -70.3976,
                 "sun_altitude_limit_deg": -18.0, "minimum_altitude_deg": 30.0},
        "survey": {
            "start_utc": "2026-10-02T00:00:00Z",
            "end_utc": "2026-10-08T08:45:00Z",
            "slot_seconds": 900,
            "nights": [{"night_id": "N20261001", "night_date": "2026-10-01",
                        "observing_start_utc": "2026-10-02T00:00:00Z",
                        "observing_end_utc": "2026-10-02T09:00:00Z", "slot_count": 36}],
        },
        "instrument": {"n_fibers": 16, "grid_side": 4, "fiber_area_deg2": 0.4, "gap_deg": 0.0,
                       "glass_side_deg": 0.632456, "pitch_deg": 0.632456, "fov_side_deg": 2.529822,
                       "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}},
        "scoring": {"q0": 0.68, "flux_zero_point": 0.5, "exposure_zero_point_seconds": 900,
                    "reporting": {"correct_reward": 100, "false_penalty": -150,
                                  "false_report_free_allowance": 2, "max_consecutive_reports": 32}},
        "footprint": [{"component_id": "C00", "vertices": [[335.0, -5.2], [339.3, -6.3], [337.2, -3.8]]}],
        "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "target_class",
                                 "feature_flux", "science_weight", "required"],
                    "rows": [["V4T000001", 337.0, -5.0, "BGS", 1.48, 0.45, False]]},
        "limits": {"global_wallclock_seconds": 900, "max_consecutive_reports": 32,
                   "response_max_bytes": 524288},
    },
}

DECISION_MESSAGE = {
    "protocol_version": "participant-agent-protocol-v4",
    "message_type": "decision_request",
    "decision_sequence": 7,
    "payload": {
        "schema_version": "v4-decision-snapshot-v1",
        "now_utc": "2026-10-02T00:00:00Z",
        "survey_end_utc": "2026-10-08T08:45:00Z",
        "observe_action_index": 0,
        "running_total": 0.0,
        "wallclock": {"elapsed_seconds": 0.045, "remaining_seconds": 899.955,
                      "remaining_real_cpu_seconds": 899.955, "speed_factor": 1.0,
                      "wall_remaining_seconds": 1799.95, "clock_mode": "cpu"},
        "latest_bulletin": {"record_type": "bulletin", "slot_id": "N20261001-S001",
                            "night_id": "N20261001", "issued_at_utc": "2026-10-02T00:00:00Z",
                            "initial": True, "notices": []},
        "latest_forecast": None,
        "active_requests": [],
        "new_messages": [],
        "last_result": None,
    },
}


def test_parse_initialize():
    data = protocol.parse_initialize(INIT_MESSAGE)
    assert data.card_id == "alpha"
    assert data.n_fibers == 16
    assert data.grid_side == 4
    assert data.min_altitude_deg == 30.0
    assert len(data.nights) == 1
    assert data.target_columns[0] == "target_id"
    assert data.target_rows[0][0] == "V4T000001"
    assert data.max_exposure_seconds == 3600


def test_parse_decision_request():
    req = protocol.parse_decision_request(DECISION_MESSAGE)
    assert req.decision_sequence == 7
    assert req.now_utc is not None
    assert req.remaining_seconds() == 899.955
    assert req.wall_remaining_seconds() == 1799.95
    assert req.last_result is None


def test_observe_response_fields():
    action = Action(
        type="observe",
        pointing={"alt_deg": 55.0, "az_deg": 120.0},
        assignments={"0": "V4T000001", "5": "V4T000002"},
        exposure_seconds=900,
        program="DARK",
        reason="test",
        decision_source="static",
    )
    obj = protocol.build_response_object(7, action)
    assert obj["protocol_version"] == protocol.PROTOCOL_VERSION
    assert obj["message_type"] == "decision_response"
    assert obj["decision_sequence"] == 7
    assert obj["action"] == "observe"
    assert obj["duration_seconds"] == 900
    assert obj["program"] == "DARK"
    assert obj["assignments"] == {"0": "V4T000001", "5": "V4T000002"}
    # 不含未知字段 (如 ra_deg/dec_deg/type/exposure_seconds)
    for bad in ("type", "ra_deg", "dec_deg", "exposure_seconds"):
        assert bad not in obj
    json.dumps(obj)  # 必须可 JSON 序列化


def test_wait_response_mutual_exclusion():
    dur = Action(type="wait", duration_seconds=900)
    obj = protocol.build_response_object(1, dur)
    assert obj["action"] == "wait"
    assert obj["duration_seconds"] == 900
    assert "until_utc" not in obj

    until = Action(type="wait", until_utc="2026-10-03T00:00:00Z")
    obj2 = protocol.build_response_object(2, until)
    assert obj2["until_utc"] == "2026-10-03T00:00:00Z"
    assert "duration_seconds" not in obj2


def test_report_and_finish_no_extra_fields():
    rep = protocol.build_response_object(3, Action(type="report", reason="fault"))
    assert rep["action"] == "report"
    assert "duration_seconds" not in rep and "pointing" not in rep

    fin = protocol.build_response_object(4, Action(type="finish"))
    assert fin["action"] == "finish"
    assert "duration_seconds" not in fin and "pointing" not in fin


def test_read_messages_skips_bad_lines():
    lines = ["", "not json", json.dumps(INIT_MESSAGE)]
    messages = list(protocol.read_messages(lines))
    assert len(messages) == 1
    assert messages[0]["message_type"] == "initialize"


def test_send_response_writes_stdout_json(capsys):
    action = Action(type="wait", duration_seconds=900)
    protocol.send_response(42, action)
    captured = capsys.readouterr()
    line = captured.out.strip()
    obj = json.loads(line)
    assert obj["decision_sequence"] == 42
    assert obj["action"] == "wait"
    # 日志必须走 stderr, stdout 只有一行 JSON
    assert captured.out.count("\n") == 1


def test_stdout_is_pure_json(capsys):
    for seq, action in [
        (1, Action(type="observe", pointing={"alt_deg": 50.0, "az_deg": 10.0},
                   assignments={"0": "t1"}, exposure_seconds=300, program="BRIGHT")),
        (2, Action(type="wait", duration_seconds=600)),
        (3, Action(type="report")),
        (4, Action(type="finish")),
    ]:
        protocol.send_response(seq, action)
    captured = capsys.readouterr()
    out_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(out_lines) == 4
    for ln in out_lines:
        json.loads(ln)  # 每行都是合法 JSON


def test_send_response_respects_bound_protocol_stream():
    """绑定协议流后, 即使 sys.stdout 被重定向, 协议消息也只写到绑定流 (stdout)."""
    import io
    import sys as _sys
    import protocol
    # 用本次私有文件对象模拟"真实 stdout"
    class _Sink(io.StringIO):
        pass
    sink = _Sink()
    orig_stdout = _sys.stdout
    orig_binding = protocol._PROTOCOL_STREAM
    try:
        protocol.bind_protocol_stdout(sink)
        # 把当前 sys.stdout 重定向到别处 (模拟被 stderr 接管)
        redirect = io.StringIO()
        _sys.stdout = redirect
        protocol.send_response(99, Action(type="wait", duration_seconds=900))
        _sys.stdout = orig_stdout
        # 协议消息应落在 sink, 而不是 redirect
        assert "decision_response" in sink.getvalue()
        assert "decision_response" not in redirect.getvalue()
    finally:
        _sys.stdout = orig_stdout
        protocol._PROTOCOL_STREAM = orig_binding
