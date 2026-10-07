#!/usr/bin/env python3
"""Agent Observer v4 reference agent ("pro"). Standard library only, participant-agent-protocol-v4.

One JSON object per line on stdin, one per line on stdout, logs on stderr.

What it does (details in README.md and planner.py):

1. Planning: every decision picks pointing, fibre assignment, duration and program together, maximising
   expected gain minus a price for telescope time (gain - lambda * T). Required targets and observation
   requests enter as probability-weighted bonuses.
2. Program: the band level is fitted to saturated hits, which show the program multiplier exactly.
3. Instrument faults: E = (quality level) / (band level). Weather moves both, a fault only the first;
   when E stays low the agent reports. Free false reports are spent early; each low episode is probed
   once; paid probes need two low nights.
4. Pace: the search level adapts to the measured cost per decision so a 4-month card fits the wall clock.
5. Model (advisor.py): at every night start a night plan (forecast + bulletin -> bad night, sectors to avoid)
   and a fault review (own quality table -> how likely a fault is, which gates paid reports); before a paid
   report the model confirms or vetoes. Calls run in the background; without an API key the agent exits,
   except when the platform sets OBSERVER_MODEL_DISABLED=1 (an evaluation without a model): then rules only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

from pro.advisor import Advisor
from pro.llm_client import LLMClient, api_key, load_dotenv
from pro.planner import DIRECTION_AZ, Planner
from pro.skymath import format_utc, parse_utc

WEATHER_KINDS = {"rain", "storm", "overcast", "haze", "cold_snap"}
BAD_KINDS = {"rain", "storm", "overcast", "haze"}
PROTOCOL = "participant-agent-protocol-v4"


def _env(name: str, default: float) -> float:
    return type(default)(os.environ.get(f"PRO_{name}", default))


# --- fault reporting (see _fault_verdict) ---
E_LOW_FREE = _env("E_LOW_FREE", 0.9)      # free probe: E below this in 3 of the last 4 hours
E_LOW_FREE2 = _env("E_LOW_FREE2", 0.9)    # ... threshold while both free probes are left
E_FREE_HOURS = _env("E_FREE_HOURS", 4)
E_LOW = _env("E_LOW", 0.85)               # paid probe: low hours must span two nights ...
E_PAID_LOW = _env("E_PAID_LOW", 0.75)     # ... the median of the last 12 hourly E below this ...
E_PAID_HOURS = _env("E_PAID_HOURS", 12)
E_RECOVER = _env("E_RECOVER", 0.95)       # after a false probe, wait until E is back above this
MAX_PAID_FALSE = _env("MAX_PAID", 6)
PERSIST_NIGHTS = _env("PERSIST_NIGHTS", 3)
E_PAID_STEP = _env("E_PAID_STEP", 0.05)   # ... minus this per paid false probe so far
# 8 会在"快速报修"下过早撞顶: 实测 v6 在卡 A1 已用掉 7 次误报, 一旦到 8 次就**永久
# 停止报修** -> 后面所有真故障都无法修复, 直接灾难。误报在每次报对后的免罚额度内是
# 免费的 (free_allowance=2), 只有超出的才 -150, 所以放宽总次数上限是安全的。
MAX_FALSE_REPORTS = 40
# 绝对质量下限的故障判据 (Hard 模式卡的关键). 官方 pro 的 E = 质量/档位 判据在"档位
# 估计跟着质量一起塌缩"时会失效 (E 恒为 ~1), 仪器故障就长期发现不了。实测卡 A1:
# 首夜后质量从 Q~0.6 崩到 ~0.004 并持续 100+ 夜, 而 agent 只报修 3 次 (间隔约 30 夜),
# 导致几乎全部必观测目标完不成 (-37,700)。这里补一条"绝对"判据: 只要近几小时中位
# scale 远低于晴夜模型 (默认 0.15), 就直接判故障并报修 —— 真故障修好后质量立刻恢复,
# 报对 +100 且能救回大量必观测目标; 普通卡 (A-D) 的 scale 常年 0.6~1.0, 不会误触发。
SCALE_FAULT_LEVEL = _env("SCALE_FAULT_LEVEL", 0.15)
SCALE_FAULT_HOURS = _env("SCALE_FAULT_HOURS", 1)
# --- 值班日志 (Hard 模式 A1-D1 的关键) -------------------------------------------------
# A1-D1 的 observation_request.reason 里附带本站**值班日志**: 它预告"何时动导星相机"
# —— 那一刻起所有曝光都是废片, 直到有人报修为止, 所以"到点直接报修"; 同时会写平场灯/
# 镜盖测试时段(也是废片, 但**不是故障**, 报了算误报)。日志中英日混写, 常用凯撒密码/
# 摩斯/唱名伪装, 还夹杂"听说/没确认/已取消/推迟/东京时间(UTC+9)"等干扰项。
# 官方任务卡把"只看请求目标和奖励, 不读值班日志"明确列为常见错误 —— 这正是我们先前
# A1-D1 大幅负分的原因。这里把日志原文交给模型解析成"需要报修的绝对 UTC 时刻"并排程。
DUTY_ENABLED = _env("DUTY_ENABLED", 1)
DUTY_MIN_CHARS = _env("DUTY_MIN_CHARS", 80)
DUTY_REPORT_WINDOW_HOURS = _env("DUTY_REPORT_WINDOW_HOURS", 12.0)
DUTY_HISTORY_CHARS = _env("DUTY_HISTORY_CHARS", 40000)   # 多轮历史的总字符预算
DUTY_NO_REPORT_HOURS = _env("DUTY_NO_REPORT_HOURS", 1.0)  # 平场灯/镜盖测试窗口的半宽
DUTY_SYSTEM = """你是天文台的值班助手, 负责把站里的【值班日志】变成可执行的运行决策。
日志用中文/日文/英文混写; 有的句子用凯撒密码(英文字母整体位移, 位移量常等于写那条日志当天的日期)、
摩斯电码、唱名(do=1 re=2 mi=3 fa=4 sol=5 la=6 si=7, 高音do=8, 高音re=9, 休止=0)伪装。
对话会逐条把新日志发给你。你要累积地维护一份运行状态, 并且**每一轮都输出全量结果**(不是增量),
因为你的上一次输出会被丢弃、只保留这一次的。

日志里的关键信息:
1) 动导星相机 (导星/guider camera 作业): 从那一刻起所有曝光都是废片, 一直到有人报修为止,
   所以必须在那个时刻报修 (原话: 到点直接报修 / 到点就报)。收进 report_utc。
2) 平场灯 / 镜盖测试 (flat-lamp / cover test) 时段: 数据也是废片, 但**不是故障**, 绝对不要报修。
   收进 no_report_utc。
3) 山体/地形遮挡 (例如"东南的山 32 度"、"西南 37 度"、"天顶距 52 度以内可以拍"): 收进 terrain,
   给出 direction 与"该方向低于此高度角就不能观测"的 min_alt_deg。
   "某个方向风大要关" / "发射窗口, 那个方向封" 这类方向性封锁也收进 avoid_directions。
4) 长期预报里"整体偏差 / 别把重要目标押在这晚"的夜晚 -> bad_nights (当地日期 YYYY-MM-DD)。
   薄云/薄霾但注明"照拍"的不算坏夜, 只写进 notes。
5) 干扰项必须正确处理: "听说的 / 没确认 / 先别当真"是传闻(丢弃); "取消 / 不动 / 别报"要删除;
   "推迟 / 延后"改成新时间; "记错了 / 更正"用更正后的时间; 标明东京时间(UTC+9)或(UTC)的要换算;
   同一件事被不同时间重复提到时, 以最后确认的版本为准。

我会给你日志原文、本站当地时间相对 UTC 的偏移小时数、当前 UTC 时间, 以及你上一轮的全量输出。
每轮只输出一个 JSON 对象, 键固定为:
{"report_utc": ["YYYY-MM-DDTHH:MM:SSZ", ...],
 "no_report_utc": [...],
 "avoid_directions": ["SE", ...],
 "terrain": [{"direction": "SW", "min_alt_deg": 37}],
 "bad_nights": ["2026-10-20", ...],
 "prefer_directions": [{"direction": "E", "weight": 0.8}],
 "prefer_targets": ["V4T001234", ...],
 "duration_scale": 1.0,
 "lambda_scale": 1.0,
 "notes": "200 字以内, 记录仍需记住的约定与未决事项",
 "reason": "15 words 以内的理由"}
所有时刻都是【绝对 UTC】。report_utc 要包含所有已预告但还没报修的故障时刻(含过去漏掉的);
不含平场灯/镜盖测试。没有就返回空数组。

下面这些键是**直接作用到观测决策**的旋钮 (这是你影响决策的方式, 别浪费):
* prefer_directions: 日志里点名"照拍得比较好 / 重点保住"的方位, weight 0..1, 越大越优先。
* prefer_targets: 日志/请求点名要保住的**公开目标 ID**(必须来自输入里出现过的 ID, 不确定就别填)。
* duration_scale: 0.5~2.0。日志说"暗弱目标难做/天气好"就调大(曝光更长), 说"时间紧/覆盖优先"就调小。
* lambda_scale: 0.3~2.0。调小 = 更愿意把单个目标打满(时间不值钱); 调大 = 更看重多拍新目标(时间紧)。
* bad_nights 里的夜晚, planner 会当成坏夜, 不把暗弱必观测押上去。
以上旋钮按"本期整体状态"给; 没有依据就填 1.0 / 空数组。只输出 JSON, 不要解释。"""
# The participant guide: an earthquake (announced in the bulletin) lowers instrument efficiency, the loss fades
# night by night, and a report does not repair it. So E drops right after an earthquake are not reportable, and
# while its effect may last only a new step down in E (a fresh drop from the preceding hours) is fault evidence.
QUAKE_HOLD_HOURS = _env("QUAKE_HOLD_HOURS", 12.0)   # no probes this long after an earthquake notice appears
QUAKE_STEP = _env("QUAKE_STEP", 0.8)                # step: median E of the last 3 rows < this x the 9 rows before
QUAKE_TAIL_HOURS = _env("QUAKE_TAIL_HOURS", 24.0)   # the earthquake period lasts this long after its last notice
PAID_SPACING_HOURS = 20.0
MIN_REPORT_SPACING_HOURS = 1.0
# --- pace ---
PACE_SAFETY = _env("PACE_SAFETY", 0.75)
# --- model ---
MODEL_WAIT_MAX = _env("MODEL_WAIT_MAX", 20.0)   # longest wait for the night's model answers (s)
MODEL_FAULT_HIGH = _env("MODEL_FAULT_HIGH", 0.6)  # fault review at or above this: report more readily tonight
MODEL_FAULT_LOW = _env("MODEL_FAULT_LOW", 0.15)   # ... at or below this: paid reports need the strongest evidence
SCALE_STEP = _env("SCALE_STEP", 0.7)
MODEL_FREE_PROBE = _env("MODEL_FREE_PROBE", 0)   # 1: a high fault review may also spend a free probe on a low scale            # with a likely fault: report when scale stays below this x ref
FIXED_LEVEL = _env("FIXED_LEVEL", -1)     # development only: pin the search level (deterministic runs)   # spend at most this share of the remaining wall clock


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


# 协议输出流: 由入口 (agent.py) 在启动时绑定到**真实 stdout**. 因为入口会把
# ``sys.stdout`` 重定向到 stderr (stdout 硬化), 这里不能直接 print; 否则协议消息
# 会落进日志流. 未绑定时退回 sys.stdout (便于单独运行本模块做测试).
_PROTOCOL_STREAM = None


def bind_stdout(stream) -> None:
    global _PROTOCOL_STREAM
    _PROTOCOL_STREAM = stream


def _emit(envelope: dict) -> None:
    data = (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")
    stream = _PROTOCOL_STREAM if _PROTOCOL_STREAM is not None else sys.stdout
    buf = getattr(stream, "buffer", None)
    if buf is not None:
        buf.write(data)
        buf.flush()
    else:
        stream.write(data.decode("utf-8"))
        stream.flush()


class RulesOnly:
    """The advisor's interface without a model (OBSERVER_MODEL_DISABLED=1): every rule default stands."""
    night_date = None

    def start_night(self, night_date, *_args):
        self.night_date = night_date
        return None, None

    def poll(self):
        return None, None

    def confirm_report(self, *_args):
        return None


def model_disabled() -> bool:
    """The platform sets OBSERVER_MODEL_DISABLED=1 for an evaluation started with 「本次不提供模型」 / --no-model."""
    return os.environ.get("OBSERVER_MODEL_DISABLED") == "1"


class ObserverAgent:
    def __init__(self, init: dict, rules_only: bool = False):
        started = time.monotonic()
        self.planner = Planner(init, log=log)
        self.rules_only = rules_only
        self.client = None if rules_only else LLMClient(log=log)
        self.advisor = RulesOnly() if rules_only else Advisor(self.client, log=log)
        self.model_wait = 0.0                    # wall seconds spent waiting for the model (not planning cost)
        self.fault_likely = None                 # tonight's model estimate that an instrument fault is active
        self.scale_hours: dict = {}              # hour -> [planner.scale samples] (for the model's fault table)
        self.start = parse_utc(init["survey"]["start_utc"])
        self.forecast_notices: list = []
        self.night_seen = None
        self.observes = 0
        # fault reporting state
        reporting = init["scoring"].get("reporting", {})
        self.free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0
        self.false_since_correct = 0
        self.paid_false = 0
        self.last_report_hours = -1e9
        self.ref_from_hours = -1e9
        self.episode_blocked = False
        self.blocked_at_hour = -1
        self.quake_on = False
        self.quake_onset_hours = -1e9
        self.quake_last_hours = -1e9
        # duty log (Hard-mode A1-D1): 多轮 LLM 会话, 历史日志全部进请求
        self.utc_offset_hours = float((init.get("site") or {}).get("utc_offset_hours", 0.0) or 0.0)
        self.duty_history: list = []     # LLM 请求的完整多轮历史 [{"role","content"}, ...]
        self.duty_pending: list = []     # 尚未送出的新日志条目
        self.duty_call = None
        self.duty_state: dict = {}       # 最近一次的全量解析结果 (同时用于历史裁剪时的继承)
        self.duty_times: list = []       # 需要报修的 UTC 时刻
        self.duty_done: set = set()
        self.duty_no_report: list = []   # "废片但不是故障"的窗口中心 (UTC)
        self.duty_bad_nights: set = set()
        self.duty_updates = 0
        self.consecutive_reports = 0
        # pace state
        self.cost_ema = [0.0, 0.0, 0.0, 0.0]     # CPU seconds per observe decision at each search level
        self.wall_ema = [0.0, 0.0, 0.0, 0.0]     # real seconds per observe decision (own turn, model waits excluded)
        self.turn_end = None
        self.decisions = 0
        self.engine_ema = None
        self.sim_step_ema = None
        self.last_now = None
        log(f"pro: {len(self.planner.ids)} targets, {sum(self.planner.required)} required, "
            f"{len(self.planner.nights)} nights; init {time.monotonic() - started:.2f}s; model {self.client.model if self.client else 'none (rules only)'}")

    # --- decision loop ------------------------------------------------------------------------------

    def respond(self, payload: dict) -> dict:
        started = time.monotonic()
        cpu_started = time.process_time()
        if self.turn_end is not None:   # engine time between our turns (charged only by the old real-time clock)
            gap = started - self.turn_end
            if 0.0 <= gap < 5.0:
                self.engine_ema = gap if self.engine_ema is None else 0.95 * self.engine_ema + 0.05 * gap
        model_before = self.model_wait
        level = self.planner.fast_level
        action = self._respond(payload)
        # the platform charges CPU time inside our turns; waiting for the model is free of CPU, so keep it
        # out of the real-time estimate as well
        cpu = time.process_time() - cpu_started
        wall = time.monotonic() - started - (self.model_wait - model_before)
        if action.get("action") == "observe" and level < 4:
            for ema, cost in ((self.cost_ema, cpu), (self.wall_ema, wall)):
                c = ema[level]
                ema[level] = cost if c == 0.0 else 0.9 * c + 0.1 * cost
                for k in range(level + 1, 4):   # cheaper levels not measured yet: a third of the level above
                    if ema[k] == 0.0 or ema[k] > ema[k - 1]:
                        ema[k] = ema[k - 1] / 3.0
        self.turn_end = time.monotonic()
        return action

    def _respond(self, payload: dict) -> dict:
        now = parse_utc(payload["now_utc"])
        if self.last_now is not None and self.planner.current_night(now) is not None:
            step = (now - self.last_now).total_seconds()
            if 0 < step <= 3600:
                self.sim_step_ema = step if self.sim_step_ema is None else 0.95 * self.sim_step_ema + 0.05 * step
        self.last_now = now
        hours = (now - self.start).total_seconds() / 3600.0
        planner = self.planner
        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self.forecast_notices = message.get("notices", [])
            elif message.get("record_type") == "observation_request":
                text = str(message.get("reason") or "")
                if DUTY_ENABLED and len(text) >= DUTY_MIN_CHARS:
                    self.duty_pending.append({"issued_at_utc": message.get("issued_at_utc"), "duty_log": text})
                    self._duty_tick(payload)
        last = payload.get("last_result") or {}
        if last.get("action") == "report":
            self._on_report_result(last, hours)
        planner.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        quake = any(kind == "earthquake" for kind, _ in planner.notices)
        if quake and not self.quake_on:
            self.quake_onset_hours = hours
            log(f"pro: earthquake notice at {payload['now_utc']}")
        self.quake_on = quake
        if quake:
            self.quake_last_hours = hours
        planner.on_requests(payload.get("active_requests", []))
        planner.on_result(payload.get("last_result"), now, hours)
        self._pace(payload, now)

        night = planner.current_night(now)
        if night is None:
            nxt = planner.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night
        if self.night_seen != night_index:
            self.night_seen = night_index
            self._night_advice(night_start, payload, hours)
        else:
            self._apply_advice(*self.advisor.poll())
        self.scale_hours.setdefault(int(hours), []).append(self.planner.scale)
        if (night_end - now).total_seconds() < planner.min_exposure:
            nxt = planner.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}
        if planner.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}
        # 值班日志点名的"整体偏差夜": 当作坏夜, 别把暗弱必观测押上去
        if self.duty_bad_nights and (night_start - timedelta(hours=12)).date().isoformat() in self.duty_bad_nights:
            planner.bad_forecast = True
        scheduled = self._due_report(now)
        if scheduled is not None:
            self.consecutive_reports += 1
            return scheduled
        report = self._maybe_report(hours, payload)
        if report is not None:
            self.consecutive_reports += 1
            return report
        action = planner.plan(now, night_end, night_index, hours)
        if action is None:
            self.consecutive_reports = 0
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start), "reason": "nothing useful is up"}
        self.consecutive_reports = 0
        self.observes += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.planner.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into)))

    # --- pace ---------------------------------------------------------------------------------------

    def _clock(self, payload: dict):
        """(CPU seconds left, real seconds left, fair clock?) from the request's wallclock block.

        Fair clock (current platform): the budget is normalized CPU time inside our turns, and
        remaining_real_cpu_seconds converts it to this machine's CPU seconds; a separate real-time cap
        (wall_remaining_seconds) only guards against runaway runs. Older runners count real time only."""
        wall = payload.get("wallclock") or {}
        if "remaining_real_cpu_seconds" in wall:
            return (float(wall["remaining_real_cpu_seconds"]), float(wall.get("wall_remaining_seconds", 1e9)), True)
        remaining = float(wall.get("remaining_seconds", 1e9))
        return remaining, remaining, False

    def _decisions_left(self, now) -> float:
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in self.planner.nights if end > now)
        return max(1.0, night_seconds / (self.sim_step_ema or 900.0))   # daytime waits cost nothing

    def _pace(self, payload: dict, now) -> None:
        """Pick the search level from the measured cost per decision and the decisions still to come."""
        cpu_left, wall_left, fair = self._clock(payload)
        decisions_left = self._decisions_left(now)
        engine = self.engine_ema or 0.0
        cpu_budget = PACE_SAFETY * cpu_left / decisions_left if fair else 1e9
        wall_budget = PACE_SAFETY * wall_left / decisions_left - engine
        # estimates of levels not used for a while decay, so the agent climbs back up and re-measures them
        self.decisions += 1
        if self.decisions % 50 == 0:
            for k in range(4):
                if k != self.planner.fast_level:
                    self.cost_ema[k] *= 0.85
                    self.wall_ema[k] *= 0.85
        if FIXED_LEVEL >= 0:
            self.planner.fast_level = FIXED_LEVEL
            return
        level = 0
        while level < 3 and (self.cost_ema[level] > cpu_budget or self.wall_ema[level] > wall_budget):
            level += 1
        if min(cpu_left, wall_left) < 15.0:
            level = 4
        if level != self.planner.fast_level:
            log(f"pro: pace level {level} (cpu budget {min(cpu_budget, 99) * 1000:.0f} ms, wall budget {wall_budget * 1000:.0f} ms, "
                f"cpu costs {[round(c * 1000) for c in self.cost_ema]} ms, {decisions_left:.0f} decisions left)")
            self.planner.fast_level = level

    # --- model stages (advisor.py): night plan and fault review at every night start --------------------

    def _night_advice(self, night_start, payload: dict, hours: float) -> None:
        """Night start: rule defaults first, then the two model calls (night plan, fault review)."""
        night_date = (night_start - timedelta(hours=12)).date().isoformat()
        tonight = [n for n in self.forecast_notices if night_date in n.get("nights", [])]
        bulletin = (payload.get("latest_bulletin") or {}).get("notices", [])
        # rule defaults, kept when the model gives no valid answer
        self.planner.bad_forecast = any(n.get("direction") == "ALL" and n.get("event_kind") in BAD_KINDS for n in tonight)
        self.planner.extra_avoid = set()
        self.fault_likely = None
        left = self._clock(payload)[1]
        started = time.monotonic()
        answers = self.advisor.start_night(night_date, tonight, bulletin, self._fault_table(hours), left,
                                           self._model_wait_budget(payload))
        self.model_wait += time.monotonic() - started
        self._apply_advice(*answers)

    def _model_wait_budget(self, payload: dict) -> float:
        """How long a night start may wait for the model. Waiting costs no CPU budget, only real time: use half
        of the real time the planner and the engine will not need, spread over the nights left."""
        _, wall_left, _ = self._clock(payload)
        now = self.last_now
        nights_left = max(1, sum(1 for _, end in self.planner.nights if end > now))
        per_decision = max(self.wall_ema[self.planner.fast_level], 0.05) + (self.engine_ema or 0.02)
        spare = wall_left - 1.5 * self._decisions_left(now) * per_decision - 60.0
        return max(0.0, min(MODEL_WAIT_MAX, 0.5 * spare / nights_left))

    def _apply_advice(self, plan, fault) -> None:
        if plan is not None:
            self.planner.bad_forecast = plan["bad_night"]
            self.planner.extra_avoid = set(plan["avoid_directions"])
            log(f"llm night plan {self.advisor.night_date}: bad_night={plan['bad_night']} avoid={plan['avoid_directions']} ({plan['reason']})")
        if fault is not None:
            self.fault_likely = fault["fault_likely"]
            log(f"llm fault review {self.advisor.night_date}: fault_likely={fault['fault_likely']:.2f} ({fault['reason']})")

    def _scale_ref(self) -> float:
        """Usual clear-sky scale since the last repair: 75th percentile of the hourly medians."""
        values = sorted(sorted(v)[len(v) // 2] for h, v in self.scale_hours.items() if h >= self.ref_from_hours and v)
        return values[(3 * len(values)) // 4] if len(values) >= 4 else 1.0

    def _fault_table(self, hours: float) -> dict:
        """The evidence the fault review reads: the last ~30 observed hours."""
        e_by_hour = {hour: sorted(v)[len(v) // 2] for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours}
        rows = []
        for hour in sorted(self.scale_hours)[-30:]:
            if hour < self.ref_from_hours:
                continue
            v = sorted(self.scale_hours[hour])
            stamp = (self.start + timedelta(hours=hour)).strftime("%m-%dT%H")
            rows.append([stamp, round(e_by_hour[hour], 2) if hour in e_by_hour else None, round(v[len(v) // 2], 2)])
        notices = sorted({f"{kind} {direction}" for kind, direction in self.planner.notices})
        return {"columns": ["utc_hour", "E", "scale"], "rows": rows, "ref": round(self._scale_ref(), 2),
                "notices_now": notices,
                "hours_since_earthquake_notice_began": None if self.quake_onset_hours < -1e8 else round(hours - self.quake_onset_hours, 1),
                "free_false_reports_left": max(0, self.free_left()), "paid_false_reports_so_far": self.paid_false,
                "correct_reports_so_far": self.correct_reports,
                "hours_since_last_report": None if self.reports == 0 else round(hours - self.last_report_hours, 1)}

    # --- instrument faults ---------------------------------------------------------------------------

    def _maybe_report(self, hours: float, payload: dict):
        """Report (probe) when the quality level stays below what the program bands allow.

        A report costs no time, its answer arrives at once, and the first false reports after each correct
        one are free: spend free probes readily, paid ones only on strong, lasting evidence."""
        # 值班日志明确说过这些时段是"废片但不是故障"(平场灯/镜盖测试): 报修会算误报。
        if self._in_no_report_window(parse_utc(payload["now_utc"])):
            return None
        if self.false_reports >= MAX_FALSE_REPORTS or hours - self.last_report_hours < MIN_REPORT_SPACING_HOURS:
            return None
        if hours - self.quake_onset_hours < QUAKE_HOLD_HOURS:
            return None   # the earthquake explains the drop; a report would not repair it
        if self._fault_verdict(hours, payload) and self._model_agrees(hours, payload):
            self.last_report_hours = hours
            self.reports += 1
            log(f"pro: report at {payload['now_utc']} (quality below what the program bands allow), free left {self.free_left()}")
            return {"action": "report", "reason": "quality level below what the program bands allow"}
        return None

    def _fault_verdict(self, hours: float, payload: dict) -> bool:
        """Hourly E = quality level / band level (planner.e_hours). 1 = consistent; a fault keeps E low."""
        rows = [(hour, night, sorted(v)[len(v) // 2]) for hour, night, v in self.planner.e_hours if hour >= self.ref_from_hours]
        if rows and int(hours) != getattr(self, "_logged_hour", None):
            self._logged_hour = int(hours)
            log(f"pro: E {payload['now_utc']} {rows[-1][2]:.2f} scale {self.planner.scale:.3f} band {self.planner.band_level or 0:.3f}")
        if len(rows) < E_FREE_HOURS or rows[-1][0] < int(hours) - 1:
            return False
        if QUAKE_STEP > 0 and hours - self.quake_last_hours < QUAKE_TAIL_HOURS:
            if len(rows) < 8:
                return False
            last = sorted(e for _, _, e in rows[-3:])[1]
            prev = sorted(e for _, _, e in rows[-12:-3])
            if not (last < QUAKE_STEP * prev[len(prev) // 2] and rows[-3][0] >= int(hours) - 4):
                return False
            if self.episode_blocked and rows[-3][0] <= self.blocked_at_hour:
                return False   # the step that was already probed, not a new one
            self.episode_blocked = False   # a new step is a new episode
        if self.episode_blocked:
            # this low episode was probed already and was not a fault: wait for a recovery first
            last4 = rows[-4:]
            if sum(1 for _, _, e in last4 if e >= E_RECOVER) >= 3 and rows[-1][0] > self.blocked_at_hour:
                self.episode_blocked = False
                log(f"pro: quality recovered at {payload['now_utc']}; probing re-armed")
            elif self._scale_fault(hours):
                self.episode_blocked = False   # 灾难性低质量是新情况: 允许再探一次
            else:
                return False
        if self._scale_fault(hours):
            log(f"pro: scale {self.planner.scale:.3f} < {SCALE_FAULT_LEVEL} for {SCALE_FAULT_HOURS}h "
                f"-> instrument fault suspected")
            return True
        likely = self.fault_likely
        if MODEL_FREE_PROBE and likely is not None and likely >= MODEL_FAULT_HIGH and self.free_left() > 0 and self._scale_step(hours):
            # off by default: on the practice cards it spent free probes on unannounced weather
            log(f"pro: model-flagged fault (likely {likely:.2f}) and scale below {SCALE_STEP} x ref")
            return True
        if self.free_left() > 0:
            last = rows[-E_FREE_HOURS:]
            threshold = E_LOW_FREE2 if self.free_left() >= 2 else E_LOW_FREE
            low = sum(1 for _, _, e in last if e < threshold)
            return low >= E_FREE_HOURS - 1 and last[-1][2] < threshold and last[-1][0] - last[0][0] <= E_FREE_HOURS + 3
        if self.paid_false >= MAX_PAID_FALSE or hours - self.last_report_hours < PAID_SPACING_HOURS:
            return False
        last = rows[-E_PAID_HOURS:]
        values = sorted(e for _, _, e in last)
        nights = {night for _, night, e in last if e < E_LOW}
        # a fault never goes away on its own: three low nights in a row justify a probe whatever the bar
        by_night: dict = {}
        for _, night, e in rows:
            by_night.setdefault(night, []).append(e)
        nights_seq = sorted(by_night)[-PERSIST_NIGHTS:]
        if (len(nights_seq) == PERSIST_NIGHTS and nights_seq[-1] - nights_seq[0] <= PERSIST_NIGHTS
                and all(len(by_night[n]) >= 3 and sorted(by_night[n])[len(by_night[n]) // 2] < E_LOW for n in nights_seq)
                and hours - self.last_report_hours >= 40.0):
            return True
        if likely is not None and likely <= MODEL_FAULT_LOW:
            return False   # the fault review sees weather, not a fault: only the persistence rule above may report
        # each paid false probe raises the bar for the next one
        paid_low = E_PAID_LOW - E_PAID_STEP * self.paid_false
        return (len(last) == E_PAID_HOURS and values[len(values) // 2] < paid_low and len(nights) >= 2
                and all(e < E_LOW for _, _, e in last[-3:]))

    def _scale_step(self, hours: float) -> bool:
        """The last 3 observed hours all sit below SCALE_STEP x the usual clear-sky scale."""
        recent = [sorted(v)[len(v) // 2] for h, v in sorted(self.scale_hours.items()) if h >= self.ref_from_hours and v][-3:]
        return len(recent) == 3 and max(recent) < SCALE_STEP * self._scale_ref()

    def _scale_fault(self, hours: float) -> bool:
        """近几小时中位 scale 是否远低于晴夜模型 (绝对判据, 不依赖会塌缩的档位估计)."""
        recent = [sorted(v)[len(v) // 2] for h, v in sorted(self.scale_hours.items())
                  if h >= self.ref_from_hours and v][-SCALE_FAULT_HOURS:]
        return (len(recent) >= SCALE_FAULT_HOURS
                and float(self.planner.scale) < SCALE_FAULT_LEVEL
                and max(recent) < SCALE_FAULT_LEVEL)

    def _model_agrees(self, hours: float, payload: dict) -> bool:
        """Paid probes only: the model looks at the evidence first and may veto. Free probes cost nothing, so they
        never wait for it. No answer in time: the rule's decision stands."""
        if self.free_left() > 0:
            return True
        rows = [(hour, sorted(v)[len(v) // 2]) for hour, _, v in self.planner.e_hours if hour >= self.ref_from_hours][-24:]
        evidence = {"hourly_E_last_24h": [round(e, 2) for _, e in rows], "fault_table": self._fault_table(hours),
                    "paid_false_reports_so_far": self.paid_false, "correct_reports_so_far": self.correct_reports}
        started = time.monotonic()
        left = self._clock(payload)[1]
        verdict = self.advisor.confirm_report(evidence, left, min(30.0, 2.0 * self._model_wait_budget(payload)))
        self.model_wait += time.monotonic() - started
        if verdict is False:
            log(f"pro: model vetoed a paid report at {payload['now_utc']}")
            self.last_report_hours = hours
            return False
        return True

    def free_left(self) -> int:
        return self.free_allowance - self.false_since_correct

    def _on_report_result(self, result: dict, hours: float) -> None:
        if result.get("correct"):
            log(f"pro: report correct, fault repaired (delta {result.get('score_delta')})")
            self.correct_reports += 1
            self.false_since_correct = 0
            self.planner.forget_quality_history()
            self.ref_from_hours = hours
        else:
            self.false_since_correct += 1
            self.false_reports += 1
            self.episode_blocked = True
            self.blocked_at_hour = int(hours)
            if self.false_since_correct > self.free_allowance:
                self.paid_false += 1
            log(f"pro: report false (delta {result.get('score_delta')}); free left {self.free_left()}")


    # --- 值班日志: 多轮 LLM 会话 (Hard-mode A1-D1) ------------------------------------------------
    #
    # 设计要点 (对照官方文档与 A1-D1 任务卡):
    #   * 值班日志随 observation_request 逐条下发, 条目之间**互相引用**(推迟/更正/取消/换算),
    #     只看最新一条必然误判 -> 所以把**全部历史**放进 LLM 请求的多轮 messages 里;
    #   * 每轮要求模型输出**全量**运行状态 (报修时刻/禁报窗口/规避方向/地形阈值/坏夜), 我们
    #     直接采用这一版, 于是"状态"随历史自然累积, 不会因为丢掉旧原文而丢失结论;
    #   * 历史超预算时成对丢弃最早的 user/assistant 轮, 并在最前面放一条"上一版全量状态"
    #     作为继承, 保证结论不丢;
    #   * 解析结果直接驱动决策: 到点报修、禁报窗口内不误报、规避方向/地形阈值交给规划器。

    def _duty_tick(self, payload: dict) -> None:
        """有新一轮日志时推进多轮会话 (后台线程, 不阻塞决策)."""
        if self.client is None:
            self.duty_pending = []
            return
        if self.duty_call is not None:
            if not self.duty_call.done():
                return                      # 上一轮还没回来: 先攒着, 下轮再发
            answer = self.client.collect(self.duty_call)
            self.duty_call = None
            if answer is None:
                # 调用失败: 把这一轮从历史里撤掉, 日志下次重发 (不把失败当结论)
                if self.duty_history and self.duty_history[-1].get("role") == "user":
                    dropped = self.duty_history.pop()
                    try:
                        again = json.loads(dropped.get("content") or "{}")
                        if again.get("duty_log"):
                            self.duty_pending.insert(0, {"issued_at_utc": again.get("issued_now_utc"),
                                                         "duty_log": again["duty_log"]})
                    except (ValueError, TypeError):
                        pass
            else:
                self.duty_history.append({"role": "assistant",
                                          "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))})
                self._apply_duty(answer)
        if not self.duty_pending:
            return
        turn = {
            "duty_log": "\n\n".join(str(e.get("duty_log") or "") for e in self.duty_pending),
            "local_utc_offset_hours": self.utc_offset_hours,
            "issued_now_utc": payload.get("now_utc"),
        }
        self.duty_history.append({"role": "user",
                                  "content": json.dumps(turn, ensure_ascii=False, separators=(",", ":"))})
        self._trim_duty_history()
        messages = [{"role": "system", "content": DUTY_SYSTEM}] + self.duty_history
        started = time.monotonic()
        call = self.client.submit_messages("duty_log", messages, self._clock(payload)[1])
        self.model_wait += time.monotonic() - started
        if call is None:
            self.duty_history.pop()         # 预算不允许: 撤轮, 下轮重试 (duty_pending 保留)
            return
        self.duty_call = call
        self.duty_pending = []

    def _duty_chars(self) -> int:
        return sum(len(m.get("content") or "") for m in self.duty_history)

    def _trim_duty_history(self) -> None:
        """历史超预算时, 成对丢弃最早的 user/assistant 轮, 并在头部放一条"继承状态"."""
        if self._duty_chars() <= DUTY_HISTORY_CHARS or len(self.duty_history) <= 2:
            return
        carry = {"role": "user", "content": json.dumps(
            {"carried_state": self.duty_state,
             "note": "更早的原始日志已省略; 请以上一版全量状态为准继续累积"},
            ensure_ascii=False, separators=(",", ":"))}
        while len(self.duty_history) > 1 and self._duty_chars() > DUTY_HISTORY_CHARS:
            self.duty_history.pop(0)
        self.duty_history.insert(0, carry)
        log(f"pro: duty history 裁剪到 {self._duty_chars()} 字符 (保留继承状态)")

    @staticmethod
    def _duty_time(value):
        try:
            return parse_utc(str(value))
        except (ValueError, TypeError):
            return None

    def _apply_duty(self, answer) -> None:
        """采用模型给出的**全量**状态, 并把它接进规划器与报修逻辑."""
        if not isinstance(answer, dict):
            return
        self.duty_state = answer
        self.duty_updates += 1
        times = sorted({m for m in (self._duty_time(x) for x in (answer.get("report_utc") or [])) if m})
        self.duty_times = [t for t in times if t not in self.duty_done]
        self.duty_no_report = [m for m in (self._duty_time(x) for x in (answer.get("no_report_utc") or [])) if m]
        self.duty_bad_nights = {str(x)[:10] for x in (answer.get("bad_nights") or []) if str(x).strip()}
        # 方向性规避 (风大关闭 / 发射窗口封锁 / 日志点名的方位)
        avoid = {str(d).upper() for d in (answer.get("avoid_directions") or [])}
        avoid = {d for d in avoid if d in DIRECTION_AZ}
        if avoid:
            self.planner.extra_avoid |= avoid
        # 地形遮挡的最低高度角 (公告只说方向, 日志给出阈值)
        for item in (answer.get("terrain") or []):
            if not isinstance(item, dict):
                continue
            direction = str(item.get("direction", "")).upper()
            try:
                min_alt = float(item.get("min_alt_deg"))
            except (TypeError, ValueError):
                continue
            if direction in DIRECTION_AZ and 0.0 <= min_alt <= 89.0:
                self.planner.terrain_min_alt[direction] = min_alt
        # 优先方位 (0..1)
        prefer: dict = {}
        for item in (answer.get("prefer_directions") or []):
            if isinstance(item, dict):
                direction = str(item.get("direction", "")).upper()
                try:
                    weight = float(item.get("weight"))
                except (TypeError, ValueError):
                    continue
            else:
                direction, weight = str(item).upper(), 0.6
            if direction in DIRECTION_AZ:
                prefer[direction] = max(0.0, min(1.0, weight))
        if prefer:
            self.planner.llm_prefer = prefer
        # 重点目标 (日志点名的公开目标 ID) -> 价值倍率
        known = set(self.planner.ids)
        boost = {str(t): 2.0 for t in (answer.get("prefer_targets") or []) if str(t) in known}
        if boost:
            self.planner.llm_target_boost = boost
        # 两个连续旋钮: 曝光时长倍率 / 时间价格倍率
        for key, attr, lo, hi in (("duration_scale", "llm_duration_scale", 0.5, 2.0),
                                  ("lambda_scale", "llm_lambda_scale", 0.3, 2.0)):
            try:
                value = max(lo, min(hi, float(answer.get(key))))
            except (TypeError, ValueError):
                continue
            setattr(self.planner, attr, value)
        log(f"pro: duty log #{self.duty_updates}: report={len(self.duty_times)} "
            f"no_report={len(self.duty_no_report)} avoid={sorted(avoid)} "
            f"terrain={self.planner.terrain_min_alt} bad_nights={len(self.duty_bad_nights)} "
            f"prefer={self.planner.llm_prefer} boost={len(self.planner.llm_target_boost)} "
            f"dur={self.planner.llm_duration_scale} lam={self.planner.llm_lambda_scale} "
            f"notes={str(answer.get('notes') or '')[:80]!r}")

    def _in_no_report_window(self, now) -> bool:
        """当前是否落在值班日志点名的"废片但不是故障"窗口里 (平场灯/镜盖测试)."""
        half = DUTY_NO_REPORT_HOURS * 3600.0
        for moment in self.duty_no_report:
            if abs((now - moment).total_seconds()) <= half:
                return True
        return False

    def _due_report(self, now):
        """到点的排程报修. 连续 report 不能超过平台上限 (32), 所以每报 3 次先做点别的."""
        if not self.duty_times or self.client is None or self.consecutive_reports >= 3:
            return None
        for moment in self.duty_times:
            if moment in self.duty_done:
                continue
            if moment <= now and (now - moment).total_seconds() <= DUTY_REPORT_WINDOW_HOURS * 3600.0:
                self.duty_done.add(moment)
                log(f"pro: duty-log report at {format_utc(now)} (scheduled {format_utc(moment)})")
                return {"action": "report", "reason": "duty log: guider camera fault"}
        return None


def main() -> int:
    # .env 与 agent.py 同级 (仓库根), 但也要兼容从 pro/ 直接运行
    for _candidate in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        load_dotenv(os.path.normpath(_candidate))
    rules_only = model_disabled()
    if rules_only:
        log("pro: OBSERVER_MODEL_DISABLED=1, running rules only (no model calls)")
    elif not api_key():
        # 官方示例在此直接退出 (return 2); 评测里退出 = 整场 0 分。这里退化为"仅规则",
        # 至少仍用确定性规划器完成巡天 (只是没有夜计划/故障复核的模型环节)。
        log("pro: 未发现任何 *_API_KEY, 退化为仅规则模式 (无模型调用)")
        rules_only = True
    agent = None
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("message_type")
        if message.get("protocol_version") != PROTOCOL:
            log(f"pro: unexpected protocol {message.get('protocol_version')!r}")
        if kind == "initialize":
            agent = ObserverAgent(message["payload"], rules_only=rules_only)
        elif kind == "decision_request":
            try:
                action = agent.respond(message["payload"])
            except Exception as exc:  # noqa: BLE001 - never crash the run: wait one slot instead
                log(f"pro: error {type(exc).__name__}: {exc}; waiting one slot")
                action = {"action": "wait", "duration_seconds": 900, "reason": "internal error"}
            action.setdefault("decision_source", "rules" if rules_only else "llm-advised")
            _emit({"protocol_version": PROTOCOL, "message_type": "decision_response",
                   "decision_sequence": message["decision_sequence"], **action})
        elif kind == "finish":
            payload = message.get("payload", {})
            log(f"pro finished: termination_reason={payload.get('termination_reason')} "
                f"observes={agent.observes if agent else 0} reports={agent.reports if agent else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
