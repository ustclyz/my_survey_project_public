"""OpenAI-compatible chat client for the pro agent (standard library only).

Configuration (environment, or a local .env next to agent.py that is never packed):
    OPENAI_API_KEY    required (KIMI_API_KEY is accepted as an alternate name)
    OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (Kimi Coding Plan; outside mainland China
                      use https://api.kimi.ai/coding/v1). On the platform this is injected automatically.
    OPENAI_MODEL      default k3

Calls never block the decision loop: `submit()` starts the request on a background thread and returns a
handle; the agent picks the answer up with `result()` on a later decision. The wall clock keeps running
while the model thinks, but the planner keeps working, so a slow reasoning model costs almost nothing.
k3 only accepts the default temperature, so none is sent.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
# 各类调用期望出现的"答案字段": 用来在模型回复里挑出**真正的答案对象**, 而不是推理文本里的
# 草稿/空 {} / 提示词模板回显。(v12 实测: 31 次解析"成功"但全是空的, 就是抓错了对象。)
_ANSWER_KEYS = ("report_utc", "no_report_utc", "avoid_directions", "terrain", "bad_nights",
                "prefer_directions", "prefer_targets", "duration_scale", "lambda_scale",
                "report_now", "fault_likely", "bad_night", "avoid", "report", "action", "reason")
_ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|\+00:00)?")
_NEXT_KEY = re.compile(r'"(?:no_report_utc|avoid_directions|terrain|bad_nights|prefer_directions|'
                       r'prefer_targets|duration_scale|lambda_scale|report_now|fault_likely|notes|reason)"\s*:')


def _salvage_report_times(text: str) -> list:
    """从**被截断**的回复里抢救 report_utc 里的时刻.

    实测 (relay 真机回放): 模型把答案放在开头, 但长推理会把输出预算吃光 -> JSON 断在半路,
    只有嵌套的小对象能解析出来, 结果"解析成功但全空"。这里直接按 `report_utc` 段落抓 ISO 时刻。
    """
    for marker in ('"report_utc"', "report_utc"):
        index = text.find(marker)
        if index >= 0:
            break
    else:
        return []
    tail = text[index + len(marker):]
    stop = _NEXT_KEY.search(tail)
    segment = tail[: stop.start()] if stop else tail[:20000]
    return _ISO_UTC.findall(segment)


def _normalise_stamp(stamp: str) -> str:
    value = stamp.strip().replace(" ", "T")
    if value.endswith("+00:00"):
        value = value[:-6]
    if not value.endswith("Z"):
        value += "Z"
    if len(value) == len("2026-10-02T01:30Z"):
        value = f"{value[:-1]}:00Z"
    return value


def _balanced_objects(text: str) -> list:
    """text 里所有**括号平衡**的 {..} 片段 (按出现顺序)."""
    out = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:index + 1])
                start = None
    return out


def _answer_score(obj: dict) -> int:
    """越大越像"真正的答案": 有非空的报修时刻/坏夜/动作/把握度最重。"""
    score = 0
    if isinstance(obj.get("report_utc"), list) and obj["report_utc"]:
        score += 3
    for key in ("bad_night", "fault_likely", "action", "report"):
        if key in obj:
            score += 2
    for key in ("avoid_directions", "terrain", "bad_nights", "prefer_directions", "avoid"):
        if isinstance(obj.get(key), list) and obj[key]:
            score += 1
    return score


def _extract_json(text: str):
    """从模型回复里取一个 JSON 对象.

    候选 = 贪心匹配 + 所有括号平衡片段; 其中"含答案字段且内容非空"的得分最高, 同分取最后
    出现的(最终答案)。这样既不会抓到推理里的空 {}, 也不会抓到提示词模板回显。
    """
    if not text:
        return None
    candidates = []
    match = _JSON_OBJECT.search(text)
    if match:
        candidates.append(match.group(0))
    candidates.extend(_balanced_objects(text))
    parsed: list = []
    for candidate in candidates:
        # 常见坏法: 字符串里带裸换行/制表符, 或对象尾部多一个逗号 -> 修一下再试
        repaired = re.sub(r"[\x00-\x1f]", " ", candidate)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        for text_try in (candidate, repaired):
            try:
                obj = json.loads(text_try)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                parsed.append(obj)
                break
    if not parsed:
        return None
    best_index = max(range(len(parsed)), key=lambda i: (_answer_score(parsed[i]), i))
    return parsed[best_index]


def _prefix_order() -> list:
    """候选密钥前缀, 按优先级排列.

    官方示例只认 ``OPENAI_*`` / ``KIMI_*``, 但**平台生效的模型服务前缀是 ``SOAD_*``**:
    被标记为 model off 的 ``OPENAI_*`` 不会下发给评测。因此这里既覆盖官方前缀, 也自动
    发现环境里任意其它 ``<PREFIX>_API_KEY`` (与仓库 config.py 的 provider_candidates
    同源), 避免密钥存成 SOAD 时整场退化为无模型。
    """
    known = ("OPENAI", "KIMI", "MOONSHOT", "DEEPSEEK")
    suffix = "_API_KEY"
    extra = sorted({
        name[: -len(suffix)]
        for name in os.environ
        if name.endswith(suffix) and len(name) > len(suffix)
        and name[: -len(suffix)] not in known
        and name[: -len(suffix)] != "ANTHROPIC"
    })
    return [*known, *extra]


_DEFAULT_BASE_URL_BY_PREFIX = {
    "OPENAI": DEFAULT_BASE_URL,
    "KIMI": DEFAULT_BASE_URL,
    "MOONSHOT": "https://api.moonshot.cn/v1",
    "DEEPSEEK": "https://api.deepseek.com",
}


def _credentials() -> tuple:
    """返回 ``(key, base_url, model)``; 第一组提供了 ``<PREFIX>_API_KEY`` 的前缀胜出."""
    for prefix in _prefix_order():
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        if key:
            base = (os.environ.get(f"{prefix}_BASE_URL", "").strip()
                    or os.environ.get("OPENAI_BASE_URL", "").strip()
                    or _DEFAULT_BASE_URL_BY_PREFIX.get(prefix, DEFAULT_BASE_URL))
            model = (os.environ.get(f"{prefix}_MODEL", "").strip()
                     or os.environ.get("OPENAI_MODEL", "").strip()
                     or DEFAULT_MODEL)
            return key, base.rstrip("/"), model
    return "", DEFAULT_BASE_URL, DEFAULT_MODEL


def api_key() -> str:
    return _credentials()[0]


def load_dotenv(path: str) -> None:
    """Fill missing environment variables from a local .env (for local runs only)."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip('"').strip("'")
                if name and value and not os.environ.get(name):
                    os.environ[name] = value
    except OSError:
        pass


class Call:
    """One background chat completion."""

    def __init__(self, client: "LLMClient", tag: str, messages: list, timeout: float, max_tokens: int):
        self.tag = tag
        self.answer = None
        self.error = None
        self.seconds = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(client, messages, timeout, max_tokens), daemon=True)
        self._thread.start()

    def _run(self, client, messages, timeout, max_tokens) -> None:
        started = time.monotonic()
        for attempt in range(client.max_retries):
            try:
                self.answer = client._request(messages, timeout, max_tokens)
                self.error = None
                break
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                self.error = type(exc).__name__
                if time.monotonic() - started > timeout:
                    break
                time.sleep(1.0 + attempt)
        self.seconds = time.monotonic() - started
        self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, seconds: float) -> bool:
        return self._done.wait(max(0.0, seconds))


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout: float = 90.0, max_calls: int = 1500,
                 max_retries: int = 3, max_in_flight: int = 4, max_tokens: int = 2000):
        self.log = log
        self.key, self.base_url, self.model = _credentials()
        self.call_timeout = call_timeout
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.max_in_flight = max_in_flight
        self.max_tokens = max_tokens
        self.calls: list[Call] = []
        self.ok = 0
        self.failed = 0

    def _request(self, messages: list, timeout: float, max_tokens: int = None) -> dict:
        """把完整的 messages 数组 (含 system / 历史 user+assistant 轮) 发给 /chat/completions."""
        body = json.dumps({"model": self.model, "messages": messages,
                           "max_tokens": int(max_tokens or self.max_tokens)}).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        raw = (message.get("content") or "") + "\n" + (message.get("reasoning_content") or "")
        parsed = _extract_json(message.get("content") or "")
        if parsed is None:
            # 推理型模型可能把答案放在 reasoning_content, content 为空
            parsed = _extract_json(message.get("reasoning_content") or "")
        salvaged = [_normalise_stamp(s) for s in _salvage_report_times(raw)]
        if parsed is None:
            if salvaged:
                # JSON 被截断到连一个完整对象都没有: 只抢救报修时刻 (这是最值钱的信息)
                self.log(f"llm: JSON 不完整, 从原文抢救出 {len(salvaged)} 个报修时刻")
                parsed = {"report_utc": list(dict.fromkeys(salvaged))}
            else:
                # 失败时把原始回复(截断)打到 stderr: 排查"模型到底回了什么"的唯一线索
                self.log(f"llm: 回复里没有 JSON, 原文前 400 字: {raw[:400]!r}")
                raise ValueError("no JSON object in the reply")
        if not any(key in parsed for key in _ANSWER_KEYS):
            self.log(f"llm: 解析到的对象没有已知字段 {sorted(parsed)[:6]}, 原文前 300 字: {raw[:300]!r}")
        # 截断抢救: 只要报修时刻缺失, 就从原文里按 report_utc 段落直接抓 ISO 时刻
        if not isinstance(parsed.get("report_utc"), list) or not parsed.get("report_utc"):
            if salvaged:
                merged = list(dict.fromkeys([*salvaged, *(parsed.get("report_utc") or [])]))
                parsed = {**parsed, "report_utc": merged}
                self.log(f"llm: 回复被截断, 从原文抢救出 {len(salvaged)} 个报修时刻")
            elif "report_utc" in parsed:
                # 模型明确给了空列表: 打出原文前 300 字, 判断是"真没有"还是"没读懂日志"
                self.log(f"llm: report_utc 为空, 原文前 300 字: {raw[:300]!r}")
        return parsed

    def in_flight(self) -> int:
        return sum(1 for call in self.calls if not call.done())

    def submit_messages(self, tag: str, messages: list, wallclock_left: float, max_tokens: int = None):
        """Start a background call with a **full message array** (system + 多轮历史). None if limits say no."""
        timeout = min(self.call_timeout, wallclock_left - 30.0)
        if len(self.calls) >= self.max_calls or timeout < 5.0 or self.in_flight() >= self.max_in_flight:
            return None
        call = Call(self, tag, messages, timeout, int(max_tokens or self.max_tokens))
        self.calls.append(call)
        return call

    def submit(self, tag: str, system: str, user: dict, wallclock_left: float):
        """单轮调用 (system + 一个 JSON user 消息). 保持与 advisor.py 的旧接口兼容."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, separators=(",", ":"))}]
        return self.submit_messages(tag, messages, wallclock_left)

    def collect(self, call):
        """The parsed answer of a finished call (None while running or after a failure). Logs once."""
        if call is None or not call.done():
            return None
        if not getattr(call, "_logged", False):
            call._logged = True
            if call.answer is not None:
                self.ok += 1
            else:
                self.failed += 1
                self.log(f"llm: {call.tag} failed ({call.error}); rules decide")   # never log the key
        return call.answer
