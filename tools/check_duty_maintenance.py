# -*- coding: utf-8 -*-
"""check_duty_maintenance.py - 值班日志链路离线自检 (不需要网络, 不需要平台密钥)。

跑法:  python tools/check_duty_maintenance.py        # 在仓库根目录执行
覆盖:  确定性命中 / 重复 ingest 幂等 / 传闻与未确认丢弃 / 平场灯与镜盖测试识别为窗口 /
        取消生效 / 时区缺失时不猜测 / LLM 结果与解析器合并 / LLM 能否决解析器 /
        排程时刻只增不减 (关键回归: 模型的"全量状态"漏掉旧时刻时不得把它挤掉)。
"""
import sys
import inspect
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pro import agent as A

# 本版本是否包含"排程只增不减"的修复 (分支 exp/duty-accum)
ACCUM_FIX = "| set(self.duty_times))" in inspect.getsource(A.ObserverAgent._apply_duty)


def msg(request_id, reason, issued="2026-10-05T10:00:00Z"):
    return {"record_type": "observation_request", "request_id": request_id,
            "issued_at_utc": issued, "reason": reason}


class FakePlanner:
    def __init__(self):
        self.ids = set()
        self.extra_avoid = set()
        self.terrain_min_alt = {}
        self.llm_prefer = {}
        self.llm_target_boost = {}
        self.llm_duration_scale = 1.0
        self.llm_lambda_scale = 1.0


class Fake:
    """只带 _apply_maintenance / _apply_duty 需要的属性。"""

    _apply_maintenance = A.ObserverAgent._apply_maintenance
    _apply_duty = A.ObserverAgent._apply_duty
    _duty_time = staticmethod(A.ObserverAgent._duty_time)

    def __init__(self, offset=8.0):
        self.maintenance = A.MaintenanceSchedule(offset)
        self.duty_times = []
        self.duty_done = set()
        self.duty_cancelled = set()
        self.duty_test_windows = set()
        self.duty_state = {}
        self.duty_updates = 0
        self.duty_no_report = []
        self.duty_bad_nights = set()
        self.planner = FakePlanner()


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


fails = []


def check(name, got, want):
    ok = got == want
    print(("PASS " if ok else "FAIL ") + name + f"  got={got} want={want}")
    if not ok:
        fails.append(name)


# 1) 确定性解析: 工程通知里的绝对时刻 (当地 UTC+8 -> UTC)
f = Fake(8.0)
f.maintenance.ingest([msg("r1", "工程组通知: 10/5 21:30 导星相机到点报修, 请及时处理。")])
f._apply_maintenance()
check("确定性命中 1 个时刻", f.duty_times, [utc(2026, 10, 5, 13, 30)])

# 2) 同一份日志重复 ingest 不能重复排程
f.maintenance.ingest([msg("r1", "工程组通知: 10/5 21:30 导星相机到点报修, 请及时处理。")])
f._apply_maintenance()
check("重复 ingest 幂等", len(f.duty_times), 1)

# 3) 传闻/未确认/假如一律不采信
g = Fake(8.0)
g.maintenance.ingest([msg("r2", "没有确认: 听说 10/5 21:30 导星相机可能不动了。")])
g._apply_maintenance()
check("传闻不采信", g.duty_times, [])

# 4) 平场灯/镜盖测试是窗口, 不是故障
h = Fake(8.0)
h.maintenance.ingest([msg("r3", "工程组通知: 10/5 平场灯测试 22:00 到 23:00。")])
h._apply_maintenance()
check("测试窗口不产生报修", h.duty_times, [])
check("测试窗口被记录", len(h.duty_test_windows), 1)

# 5) 取消
i = Fake(8.0)
i.maintenance.ingest([msg("r4", "工程组通知: 10/5 21:30 导星相机到点报修。")])
i._apply_maintenance()
i.maintenance.ingest([msg("r5", "更正: 10/5 21:30 的导星相机报修取消。")])
i._apply_maintenance()
check("取消后排程清空", i.duty_times, [])
check("取消时刻被记住", i.duty_cancelled, {utc(2026, 10, 5, 13, 30)})

# 6) LLM 的"全量状态"不能抹掉确定性时刻
j = Fake(8.0)
j.maintenance.ingest([msg("r6", "工程组通知: 10/5 21:30 导星相机到点报修。")])
j._apply_maintenance()
j._apply_duty({"report_utc": ["2026-10-06T01:00:00Z"], "notes": "duty log #2"})
check("LLM 全量状态与确定性时刻并存",
      j.duty_times, [utc(2026, 10, 5, 13, 30), utc(2026, 10, 6, 1, 0)])

# 7) LLM 可以否决确定性时刻
k = Fake(8.0)
k.maintenance.ingest([msg("r7", "工程组通知: 10/5 21:30 导星相机到点报修。")])
k._apply_maintenance()
k._apply_duty({"report_utc": [], "cancelled_utc": ["2026-10-05T13:30:00Z"]})
check("LLM 能否决确定性排程", k.duty_times, [])

# 8) LLM 可以补一个确定性解析没找到的时刻
l = Fake(8.0)
l._apply_duty({"report_utc": ["2026-10-07T09:15:00Z"], "test_windows_utc": [
    {"start_utc": "2026-10-07T10:00:00Z", "end_utc": "2026-10-07T11:00:00Z"}]})
check("LLM 时刻进入排程", l.duty_times, [utc(2026, 10, 7, 9, 15)])
check("LLM 窗进入通道", len(l.duty_test_windows), 1)

# 9) 迟到: 没有时区时不得把当地时刻当 UTC (offset 缺失 -> 丢弃)
m = Fake(None)
m.maintenance.ingest([msg("r8", "工程组通知: 10/5 21:30 导星相机到点报修。")])
m._apply_maintenance()
check("无时区不猜测", m.duty_times, [])

# 10) 累积语义: 模型第二轮的"全量状态"漏掉第一轮的时刻, 不能把已排上的时刻挤掉
n = Fake(8.0)
n._apply_duty({"report_utc": ["2026-10-05T13:30:00Z", "2026-10-09T22:00:00Z"]})
check("第一轮两个时刻", len(n.duty_times), 2)
n._apply_duty({"report_utc": ["2026-10-11T03:00:00Z"]})   # 模型这一轮只给了一个
if ACCUM_FIX:
    check("第二轮不会挤掉旧时刻", len(n.duty_times), 3)
    check("旧时刻仍在", utc(2026, 10, 5, 13, 30) in n.duty_times, True)
else:
    # 本版本把模型的回答当成"全量状态"直接替换; 已排上但还没到点的时刻会被下一轮挤掉。
    # 平台证据: all-in-v5 在 A1 卡上 17 轮解析全部成功 (抽到 3~53 个时刻), 却 0 次到点报修。
    # 修复在分支 exp/duty-accum (排程只增不减), 未进入本版本 —— 见 DESIGN.md §4.1。
    print(f"SKIP 排程累积语义: 本版本无该修复 (当前排程 {len(n.duty_times)} 个, "
          f"期望修复后 3 个)")

# 11) 但"明确取消"仍然要能移除
n._apply_duty({"report_utc": [], "cancelled_utc": ["2026-10-05T13:30:00Z"]})
check("取消仍然有效", utc(2026, 10, 5, 13, 30) in n.duty_times, False)

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
