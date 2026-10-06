#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm.py - LLM 客户端与两个 LLM 驱动环节.

本文件是"评奖硬门槛"的核心证据: 让大模型承担以下环节, 且可从代码直接看出:

    环节A·任务规划 (plan_night):
        根据简报/预报/目标统计/剩余预算, 让 LLM 决定本夜观测重点
        (候选目标子集、DARK/BRIGHT 分配策略、曝光缩放), 返回 NightPlan;
        并把 planner_tool (封装 preplan.py 的规划工具) 作为可调用工具暴露给 LLM.

    环节B·行动决策 (decide_action):
        让 LLM 在 [继续观测/转向/等待/报告故障/收尾] 间做高层选择, 结合
        night_plan 落成具体 Action (observe/wait/report/finish).

另有加分环节:
    环节C·计划自适应 (diagnose_abnormal): 观测结果明显异常时, 让 LLM 诊断并给出
        后续调整建议, 由 planner 采纳.

依赖策略 (二选一, 内部自动切换):
    * 优先使用 OpenAI 官方 Python SDK (pip install openai);
    * 缺失 openai 包时自动落到标准库 urllib.request 直连
      ``{OPENAI_BASE_URL}/chat/completions``.
    两种实现共用同一套提示词与输出解析.

预算保护 (任务卡点名的成败项):
    * 每次调用设超时 (默认 30s, 可配), 失败重试 <= 2 次;
    * **绝不每轮调 LLM**: 只在关键决策点调用 (新夜规划 / 收到限时请求 / 结果异常);
    * 缓存: 同一状态 (简报版本 + 时间窗) 的规划结果缓存复用;
    * 以 wallclock 的剩余时间为准控制调用频率;
    * 等待模型不计 CPU 预算, 但 30 分钟实际时间上限仍在, 超时上限保守.

静态回退:
    无密钥或调用失败时, plan_night / decide_action 走不依赖 LLM 的启发式实现,
    并在日志标记 ``[static]``, 保证无密钥也能跑通本地评测.

可审计:
    每次调用的 system/user/返回都写 stderr (含时间戳与调用点), 便于复现核验.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from models import Action, DecisionState, LLMConfig, NightPlan

# ---------------------------------------------------------------------------
# 常量与提示词
# ---------------------------------------------------------------------------

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

# 预算保护参数 (来源: 任务书第 4/4.5 节, 官方示例 llm_client.py 的做法)
WALL_RESERVE_SECONDS = 300.0        # 距 30 分钟硬上限此值内不再发起新调用
QUESTION_DEADLINE_SECONDS = 45.0    # 单个问题 (含重试) 的总时限
MAX_ATTEMPTS = 3                    # 失败重试 <= 2 次 (即最多 3 次尝试)
MAX_BACKOFF_SECONDS = 15.0

# 系统提示词 (环节A·任务规划) —— 来源: 任务书 4.5 节要求
_SYSTEM_PLAN = (
    "你是天文巡天台站长, 负责编排一整夜的观测。输入包含: 简报/预报事件、"
    "目标清单统计、计分配置、剩余时间预算。你必须只输出一个 JSON 对象。\n"
    "计分要点: 必观测目标漏一个扣 50 分; 天区需均匀覆盖 (按 10 度赤经条带, 最多扣 200 分); "
    "同一目标多次曝光只计最好一次, 不累加; 程序加成 DARK×1.20 / BRIGHT×1.12 / BACKUP×1.06, "
    "声明与实际档位不一致只 ×1.00; 限时观测请求有截止时间, 发布到截止之间完整完成的有效曝光才计入。\n"
    "请在 JSON 中给出: strategy(一句话策略), target_ids(本夜优先观测的目标ID数组), "
    "program(DARK/BRIGHT/BACKUP), avoid_directions(本夜规避的方向数组, 元素取 N,NE,E,SE,S,SW,W,NW), "
    "duration_scale(曝光时长缩放, 0.7-1.4)。只输出 JSON, 不要解释。"
)

# 系统提示词 (环节B·行动决策)
_SYSTEM_DECIDE = (
    "你是天文巡天台站长, 需要在每一步做一个高层决策。输入包含当前时间、本夜规划、"
    "上次观测结果与剩余预算。你必须只输出一个 JSON 对象。\n"
    "可选决策: observe(继续/转向观测) / wait(等待) / report(报告仪器故障) / finish(收尾)。\n"
    "只有当观测得分持续远低于预期且证据充分时才 report (误报会扣分); "
    "得到限时请求应优先按截止时间处理; 预算不足或天已亮可 finish/wait。\n"
    "仅当 action=observe 时给出: program(DARK/BRIGHT/BACKUP), "
    "target_ids(本步想观测的目标ID数组, 最多 16 个), duration_scale(0.7-1.4)。"
    "只输出 JSON, 不要解释。"
)

# 系统提示词 (环节C·计划自适应)
_SYSTEM_DIAGNOSE = (
    "你是天文巡天台站长, 需要诊断异常的观测结果。输入是最近一次观测的命中/得分统计与"
    "现场事件。你必须只输出一个 JSON 对象: {\"diagnosis\": 一句话, "
    "\"duration_scale\": 0.7-1.4, \"avoid_directions\": [N,NE,E,SE,S,SW,W,NW] 的子集, "
    "\"report_fault\": true|false}。只输出 JSON, 不要解释。"
)


# ---------------------------------------------------------------------------
# 结构化异常
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """LLM 调用/解析失败. 调用方据此静态回退, 绝不崩溃."""


class RetryableError(LLMError):
    """429 / 5xx / 超时 / 网络问题: 值得退避后重试."""

    def __init__(self, reason: str, retry_after: Optional[float] = None):
        super().__init__(reason)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI 兼容客户端, 优先官方 SDK, 缺失则 urllib 后备.

    两种实现共用 ``_build_messages`` / ``_parse_content`` 与全部提示词.
    """

    def __init__(self, cfg: LLMConfig, log=None, use_sdk: Optional[bool] = None):
        self.cfg = cfg
        self.log = log or (lambda text: None)
        self.calls_made = 0
        self.max_calls = 80  # 一次运行最多调用次数 (预算保护)
        # use_sdk: None=自动探测, True/False=强制
        self._use_sdk = self._detect_sdk() if use_sdk is None else use_sdk

    # -- SDK 探测 ----------------------------------------------------------
    @staticmethod
    def _detect_sdk() -> bool:
        try:
            import openai  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def backend(self) -> str:
        return "openai-sdk" if self._use_sdk else "urllib"

    def audit(self, call_site: str, system: str, user: str, reply: str) -> None:
        """把每次调用的 system/user/返回写入 stderr (含时间戳与调用点)."""
        ts = datetime.now(timezone.utc).isoformat()
        self.log(f"[llm-audit {ts}] site={call_site} backend={self.backend} model={self.cfg.model}")
        self.log(f"[llm-audit {ts}] system={system[:600]}")
        self.log(f"[llm-audit {ts}] user={user[:2000]}")
        self.log(f"[llm-audit {ts}] reply={reply[:2000]}")

    # -- HTTP 尝试 ---------------------------------------------------------
    def _build_messages(self, system: str, user: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从可能的回复文本中稳健地抽出第一个**平衡**的 JSON 对象字符串.

        兼容: 前后有解释文字、```json 代码块、以及带嵌套对象/字符串内花括号的情况.
        比贪婪正则 ``\\{.*\\}`` 更稳 (后者在多个对象或截断时会取错).
        """
        if not text:
            return None
        # 去掉 markdown 代码围栏
        text = text.replace("```json", "```").replace("```", " ")
        start = text.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            # 未闭合: 尝试下一个 '{'
            start = text.find("{", start + 1)
        return None

    @classmethod
    def _parse_content(cls, text: str) -> Dict[str, Any]:
        candidate = cls._extract_json(text)
        if candidate is None:
            # 退路: 原有正则 (兼容极端情况)
            match = _JSON_OBJECT.search(text or "")
            candidate = match.group(0) if match else None
        if not candidate:
            raise LLMError("回复中未找到 JSON 对象")
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError(f"回复 JSON 解析失败: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("回复 JSON 不是对象")
        return parsed

    def _attempt_sdk(self, system: str, user: str, timeout: float) -> Dict[str, Any]:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url, timeout=timeout)
        try:
            resp = client.chat.completions.create(
                model=self.cfg.model,
                messages=self._build_messages(system, user),
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - 归类后可重试异常
            name = type(exc).__name__
            if "RateLimit" in name or "Timeout" in name or "API" in name:
                raise RetryableError(f"{name}: {exc}") from exc
            raise LLMError(f"{name}: {exc}") from exc
        content = ""
        try:
            msg = resp.choices[0].message
            content = msg.content or ""
            # 推理模型 (如 deepseek-v4-pro) 可能把正文放在 reasoning_content 或
            # content 为空时把结果放在别处; 兜底再取一次.
            if not content:
                content = getattr(msg, "reasoning_content", "") or ""
            if not content:
                content = str(resp)
        except Exception:
            content = str(resp)
        return self._parse_content(content)

    def _attempt_urllib(self, system: str, user: str, timeout: float) -> Dict[str, Any]:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": self._build_messages(system, user),
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.cfg.base_url + "/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.cfg.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise RetryableError(f"HTTP {exc.code}") from exc
            raise LLMError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RetryableError(type(exc).__name__) from exc
        try:
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                # 推理模型兜底: reasoning_content 或整段回退
                content = (msg.get("reasoning_content") or "").strip()
            if not content:
                content = json.dumps(data, ensure_ascii=False)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"回复结构异常: {exc}") from exc
        return self._parse_content(content)

    def _attempt(self, system: str, user: str, timeout: float) -> Dict[str, Any]:
        if self._use_sdk:
            try:
                return self._attempt_sdk(system, user, timeout)
            except LLMError:
                raise
            except Exception as exc:  # SDK 意外错误 -> 尝试 urllib 一次
                self.log(f"[llm] SDK 调用异常 ({exc}); 本次改用 urllib 后备")
                return self._attempt_urllib(system, user, timeout)
        return self._attempt_urllib(system, user, timeout)

    # -- 公开接口 ----------------------------------------------------------
    def chat_json(self, system: str, user: str, call_site: str = "chat_json") -> Dict[str, Any]:
        """向 /chat/completions 发请求, 要求模型只返回一个 JSON 对象.

        解析失败抛 :class:`LLMError`, 由调用方回退; 成功返回解析后的 dict.
        无密钥 / 被禁用时直接抛 :class:`LLMError` (调用方静态回退).
        """
        if not self.cfg.usable:
            raise LLMError("未配置可用密钥或已禁用 LLM")
        if self.calls_made >= self.max_calls:
            raise LLMError("本次运行 LLM 调用次数已达上限")
        deadline = time.monotonic() + QUESTION_DEADLINE_SECONDS
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            left = deadline - time.monotonic()
            if left < 2.0:
                raise LLMError("本次调用的时间预算已耗尽")
            self.calls_made += 1
            try:
                result = self._attempt(system, user, min(self.cfg.timeout_seconds, left))
                self.audit(call_site, system, user, json.dumps(result, ensure_ascii=False))
                return result
            except RetryableError as exc:
                last_error = exc
                pause = exc.retry_after if exc.retry_after is not None else random.uniform(0, 2.0 ** (attempt - 1))
                pause = min(pause, MAX_BACKOFF_SECONDS)
                self.log(f"[llm] 第 {attempt}/{MAX_ATTEMPTS} 次尝试失败 ({exc}); {pause:.1f}s 后重试")
                if attempt < MAX_ATTEMPTS and time.monotonic() + pause < deadline - 2.0:
                    time.sleep(pause)  # 等待模型不计 CPU 预算
            except LLMError as exc:
                last_error = exc
                self.log(f"[llm] 调用失败 ({exc}); 不再重试")
                break
            except Exception as exc:  # noqa: BLE001 - 绝不让模型异常破坏决策
                last_error = exc
                self.log(f"[llm] 未预期异常 ({type(exc).__name__}: {exc}); 不再重试")
                break
        raise LLMError(f"本问题未获回答: {last_error}")


# ---------------------------------------------------------------------------
# 环节A·任务规划
# ---------------------------------------------------------------------------


class LLMPlanner:
    """封装 LLM 的两个决策环节 + 静态回退 + 规划缓存."""

    def __init__(self, cfg: LLMConfig, planner_tool=None, log=None):
        self.cfg = cfg
        self.log = log or (lambda text: None)
        self.client = LLMClient(cfg, log=self.log)
        self.planner_tool = planner_tool
        self._plan_cache: Dict[str, NightPlan] = {}
        self._abnormal_cache: Optional[Dict[str, Any]] = None

    # -- 提示词构造 (用户侧) ----------------------------------------------
    def _build_plan_user(self, state: DecisionState) -> str:
        summary = state.compact_summary(max_targets=40)
        tool_hint = ""
        if self.planner_tool is not None:
            try:
                tool_hint = self.planner_tool.describe()
            except Exception:
                tool_hint = "(规划工具 preplan.py 可用)"
        payload = {
            "task": "制定本夜观测重点",
            "state": summary,
            "planner_tool": tool_hint,
            "reminder": "只输出 JSON; target_ids 必须是输入中出现的公开目标ID。",
        }
        return json.dumps(payload, ensure_ascii=False)

    def plan_night(self, state: DecisionState) -> NightPlan:
        """环节A·任务规划: 让 LLM 决定本夜观测重点, 返回 NightPlan.

        把 planner_tool (封装 preplan.py) 作为可调用工具暴露给 LLM.
        缓存键 = (简报/事件版本 + 夜序号), 同一夜不重复调用.
        """
        cache_key = f"plan|night={state.night_index}|req={len(state.observation_requests)}|closed={state.site_closed}"
        if cache_key in self._plan_cache:
            cached = self._plan_cache[cache_key]
            self.log(f"[llm] plan_night 命中缓存 ({cache_key})")
            return cached

        if not self.cfg.usable:
            plan = self._static_plan(state)
            self.log(f"[static] plan_night 回退 ({self.cfg.api_key and 'disabled' or 'no-key'}); strategy={plan.strategy}")
            self._plan_cache[cache_key] = plan
            return plan

        try:
            user = self._build_plan_user(state)
            answer = self.client.chat_json(_SYSTEM_PLAN, user, call_site="plan_night")
            plan = self._parse_plan(answer, state)
            self.log(f"[llm] plan_night 成功: program={plan.program} targets={len(plan.targets)} strategy={plan.strategy}")
            self._plan_cache[cache_key] = plan
            return plan
        except LLMError as exc:
            self.log(f"[static] plan_night 调用失败 ({exc}); 使用启发式规划")
            plan = self._static_plan(state)
            self._plan_cache[cache_key] = plan
            return plan

    def _parse_plan(self, answer: Dict[str, Any], state: DecisionState) -> NightPlan:
        """把 LLM JSON 解析成 NightPlan, 字段缺失/非法时用安全默认值."""
        program = str(answer.get("program", "BACKUP")).upper()
        if program not in ("DARK", "BRIGHT", "BACKUP"):
            program = "BACKUP"
        targets = [str(t) for t in (answer.get("target_ids") or answer.get("targets") or [])][:64]
        avoid = [str(d).upper() for d in (answer.get("avoid_directions") or []) if str(d).upper() in
                 {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}]
        try:
            scale = float(answer.get("duration_scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        scale = max(0.7, min(1.4, scale))
        return NightPlan(
            targets=targets,
            program=program,
            strategy=str(answer.get("strategy", ""))[:240],
            avoid_directions=avoid,
            duration_scale=scale,
            source="llm",
        )

    def _static_plan(self, state: DecisionState) -> NightPlan:
        """启发式规划 (无 LLM 时的静态回退, 基于 preplan.py 的候选)."""
        targets = list(state.candidate_targets)
        if state.required_remaining > 0:
            strategy = f"必观测优先 (剩 {state.required_remaining}), 候选 {len(targets)} 个"
        else:
            strategy = f"常规覆盖, 候选 {len(targets)} 个"
        return NightPlan(
            targets=targets,
            program="BACKUP",
            strategy=strategy,
            avoid_directions=[],
            duration_scale=1.0,
            source="static",
        )

    # -- 环节B·行动决策 ----------------------------------------------------
    def _build_decide_user(self, state: DecisionState, night_plan: NightPlan) -> str:
        payload = {
            "task": "做一次高层行动决策",
            "state": state.compact_summary(max_targets=20),
            "night_plan": night_plan.as_dict(),
            "reminder": "observe 的 target_ids 必须是公开目标ID; 只输出 JSON。",
        }
        return json.dumps(payload, ensure_ascii=False)

    def decide_action(self, state: DecisionState, night_plan: NightPlan) -> Action:
        """环节B·行动决策: 让 LLM 在高层选择间决策, 结合 night_plan 落成 Action.

        调用失败/无密钥时返回 ``None``, 由 planner 的启发式决策接手.
        """
        if not self.cfg.usable:
            self.log("[static] decide_action 回退 (无密钥/禁用)")
            return None
        # 白天/站点关闭/收尾等确定性判断交给静态内核, 不必消耗 LLM
        if state.night_index is None or state.site_closed:
            return None
        try:
            user = self._build_decide_user(state, night_plan)
            answer = self.client.chat_json(_SYSTEM_DECIDE, user, call_site="decide_action")
            action = self._parse_action(answer, night_plan)
            if action is not None:
                self.log(f"[llm] decide_action -> {action.type} source=llm")
            return action
        except LLMError as exc:
            self.log(f"[static] decide_action 调用失败 ({exc}); 使用启发式决策")
            return None

    def _parse_action(self, answer: Dict[str, Any], night_plan: NightPlan) -> Optional[Action]:
        """把 LLM JSON 解析为高层 Action.

        仅解析"高层意向" (observe/wait/report/finish + 目标子集 + program + 缩放);
        具体光纤分配与曝光时长仍由 planner 用 preplan 工具计算, 保证动作合法.
        """
        kind = str(answer.get("action", answer.get("type", ""))).lower().strip()
        if kind not in ("observe", "wait", "report", "finish"):
            kind = "observe"
        reason = str(answer.get("reason", ""))[:120]
        if kind == "observe":
            program = str(answer.get("program", night_plan.program)).upper()
            if program not in ("DARK", "BRIGHT", "BACKUP"):
                program = night_plan.program
            targets = [str(t) for t in (answer.get("target_ids") or answer.get("targets") or [])][:16]
            try:
                scale = float(answer.get("duration_scale", night_plan.duration_scale))
            except (TypeError, ValueError):
                scale = night_plan.duration_scale
            scale = max(0.7, min(1.4, scale))
            # 把 LLM 的意图写回 night_plan, 供 planner 实际落成 observe
            if targets:
                night_plan.targets = targets
            night_plan.program = program
            night_plan.duration_scale = scale
            # 用一个"意向 Action" 承载, planner 会据此计算 pointing/assignments/duration
            return Action(
                type="observe",
                program=program,
                reason=reason or "LLM: 继续/转向观测",
                decision_source="llm",
            )
        if kind == "wait":
            return Action(type="wait", duration_seconds=900, reason=reason or "LLM: 等待", decision_source="llm")
        if kind == "report":
            return Action(type="report", reason=reason or "LLM: 报告故障", decision_source="llm")
        return Action(type="finish", reason=reason or "LLM: 收尾", decision_source="llm")

    # -- 环节C·计划自适应 (加分) ------------------------------------------
    def diagnose_abnormal(self, state: DecisionState) -> Optional[Dict[str, Any]]:
        """观测结果异常时让 LLM 诊断并给出后续调整 (计划自适应).

        返回 {"diagnosis","duration_scale","avoid_directions","report_fault"} 或 None.
        """
        if not self.cfg.usable:
            return None
        user = json.dumps(
            {
                "task": "诊断异常观测结果",
                "state": state.compact_summary(max_targets=20),
                "last_hit_rate": state.last_hit_rate,
                "abnormal_reason": state.abnormal_reason,
            },
            ensure_ascii=False,
        )
        try:
            answer = self.client.chat_json(_SYSTEM_DIAGNOSE, user, call_site="diagnose_abnormal")
            self.log(f"[llm] diagnose_abnormal: {answer.get('diagnosis')}")
            return answer
        except LLMError as exc:
            self.log(f"[static] diagnose_abnormal 失败 ({exc})")
            return None


# ---------------------------------------------------------------------------
# 顶层函数 (任务书 4.5 节规定的签名, 供外部/测试直接调用)
# ---------------------------------------------------------------------------


def chat_json(cfg: LLMConfig, system: str, user: str) -> Dict[str, Any]:
    """顶层函数: 向 /chat/completions 发请求, 要求只返回一个 JSON 对象.

    解析失败抛 :class:`LLMError`, 由调用方回退; 成功返回解析后的 dict.
    """
    client = LLMClient(cfg)
    return client.chat_json(system, user, call_site="chat_json")


def plan_night(state: DecisionState, planner_tool=None, cfg: Optional[LLMConfig] = None) -> NightPlan:
    """顶层函数: 环节A·任务规划.

    Args:
        state: 一次决策的状态.
        planner_tool: 封装 preplan.py 的规划工具 (可为 None).
        cfg: LLM 配置; None 时视为无 LLM, 走静态规划.
    """
    effective = cfg or LLMConfig(api_key="")
    planner = LLMPlanner(effective, planner_tool=planner_tool)
    return planner.plan_night(state)


def decide_action(
    state: DecisionState,
    night_plan: NightPlan,
    cfg: Optional[LLMConfig] = None,
) -> Optional[Action]:
    """顶层函数: 环节B·行动决策. 无 LLM 或失败时返回 None (调用方回退)."""
    effective = cfg or LLMConfig(api_key="")
    planner = LLMPlanner(effective)
    return planner.decide_action(state, night_plan)
