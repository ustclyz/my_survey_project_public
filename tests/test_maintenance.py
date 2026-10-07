from datetime import datetime, timezone

from pro.maintenance import MaintenanceSchedule


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def feed(schedule, text):
    schedule.ingest([{'record_type': 'observation_request', 'request_id': text,
                     'issued_at_utc': '2026-10-02T00:00:00Z', 'reason': text}])


def test_official_local_and_utc_duplicate_rumor_ignored():
    s = MaintenanceSchedule(-4)
    feed(s, '【立希】工程组通知：10/1（周四）21:30 前后会动导星相机。到点直接报修。\n'
         '【祥子】按 UTC 来写，同一件事是：导星相机 10/2（UTC）1:30。\n'
         '【立希】10/1 23:30 导星相机也要动，这是听说的？没确认，先别当真。')
    assert s.faults == {utc('2026-10-02T01:30:00')}
    assert not s.due(utc('2026-10-02T01:29:00'))
    assert s.due(utc('2026-10-02T01:30:00'))
    s.acknowledge(utc('2026-10-02T01:30:00'))
    assert not s.due(utc('2026-10-02T02:00:00'))


def test_tokyo_postponement_and_cancellation():
    s = MaintenanceSchedule(-4)
    feed(s, '【祥子】工程组给的是东京时间（UTC+9）：10/4 12:15 动导星相机。\n'
         '【祥子】导星相机原定 10/5 19:15，工程组刚通知推迟两小时。原来的时间点不要报。\n'
         '【祥子】工程组通知：10/6 1:30 前后，导星相机也会有动作。\n'
         '【立希】刚接到工程组的电话，10/6 1:30 那次导星相机取消了，不动，别報。')
    assert s.faults == {utc('2026-10-04T03:15:00'), utc('2026-10-06T01:15:00')}


def test_flat_test_overnight_and_relative_fault():
    s = MaintenanceSchedule(-4)
    feed(s, '【祥子】１０／６ 当晚的平场灯和镜盖测试有 ３ 段：２３：１５到２３：３０、３：００到３：３０、４：１５到４：３０。并非故障，请不要报修。\n'
         '【祥子】在 10/6 那晚，第一段平场灯测试结束以后，再等两小时四十五分，工程组就要去动导星相机；请在那一刻报修，测试本身不必报。')
    assert s.faults == {utc('2026-10-07T06:15:00')}
    assert s.test_end(utc('2026-10-07T03:20:00')) == utc('2026-10-07T03:30:00')
    assert s.test_end(utc('2026-10-07T07:15:00')) == utc('2026-10-07T07:30:00')
    assert s.next_boundary(utc('2026-10-07T03:00:00')) == utc('2026-10-07T03:15:00')


def test_chinese_time_correction_and_no_unconfirmed_guess():
    s = MaintenanceSchedule(-4)
    feed(s, '【灯】导星相机 10/5 二十一点整，对吗？\n'
         '【立希】灯搞错了，10/5 21:00 不对，导星相机是 10/5 22:30，按工程组通知来。')
    assert s.faults == {utc('2026-10-06T02:30:00')}


def test_more_than_two_faults_and_no_duplicate_replay():
    s = MaintenanceSchedule(-4)
    for day in range(2, 7):
        text = f'【立希】工程组通知：10/{day} 21:30 动导星相机。到点报修。'
        feed(s, text)
        when = utc(f'2026-10-{day+1:02}T01:30:00')
        assert s.due(when)
        s.acknowledge(when)
        feed(s, text)
        assert not s.due(when)


def test_caesar_engineering_notice_and_timezone_before_date():
    s = MaintenanceSchedule(-4)
    feed(s, '【立希】（シーザー暗号、ずらし数は 1）Uif hvjefs dbnfsb xpsl jt bu 12:15 po 10/4, Uplzp ujnf (VUD+9). Dpowfsu ju up pvs mpdbm ujnf boe sfqpsu bu uibu npnfou.')
    assert s.faults == {utc('2026-10-04T03:15:00')}


def test_unknown_timezone_does_not_guess_local_notice():
    s = MaintenanceSchedule()
    feed(s, '【立希】工程组通知：10/1 21:30 动导星相机，到点报修。')
    assert not s.faults


def test_emoticon_in_clock_and_malformed_window_do_not_shift_pairs():
    s = MaintenanceSchedule(-10)
    feed(s, '【祥子】12/1 当晚平场灯测试：21：(￣▽￣)30到22：00、乱码到23:00、0:00到0:15。不是故障。')
    assert s.tests == {(utc('2026-12-02T07:30:00'), utc('2026-12-02T08:00:00')),
                       (utc('2026-12-02T10:00:00'), utc('2026-12-02T10:15:00'))}


def test_conditional_schedule_not_treated_as_unconditional():
    s = MaintenanceSchedule(-4)
    feed(s, '【立希】工程组通知：导星相机 10/5 21:00。要是下雨，就顺延一天，到点报修。')
    assert not s.faults


def test_chinese_cancellation_stays_cancelled_after_duplicate_translation():
    s = MaintenanceSchedule(-4)
    feed(s, '【祥子】工程组通知：10/6 1:30 导星相机要动，到点报修。\n'
         '【立希】工程组电话：10/6 1:30 导星相机取消了。\n'
         '【祥子】Guider camera work is on 10/6 at 1:30 local time. Report at that moment.')
    assert not s.faults

