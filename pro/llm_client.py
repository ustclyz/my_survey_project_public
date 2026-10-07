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

    def __init__(self, client: "LLMClient", tag: str, messages: list, timeout: float):
        self.tag = tag
        self.answer = None
        self.error = None
        self.seconds = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(client, messages, timeout), daemon=True)
        self._thread.start()

    def _run(self, client, messages, timeout) -> None:
        started = time.monotonic()
        for attempt in range(client.max_retries):
            try:
                self.answer = client._request(messages, timeout)
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
                 max_retries: int = 3, max_in_flight: int = 4):
        self.log = log
        self.key, self.base_url, self.model = _credentials()
        self.call_timeout = call_timeout
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.max_in_flight = max_in_flight
        self.calls: list[Call] = []
        self.ok = 0
        self.failed = 0

    def _request(self, messages: list, timeout: float) -> dict:
        """把完整的 messages 数组 (含 system / 历史 user+assistant 轮) 发给 /chat/completions."""
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": 2000}).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError("no JSON object in the reply")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("reply is not a JSON object")
        return parsed

    def in_flight(self) -> int:
        return sum(1 for call in self.calls if not call.done())

    def submit_messages(self, tag: str, messages: list, wallclock_left: float):
        """Start a background call with a **full message array** (system + 多轮历史). None if limits say no."""
        timeout = min(self.call_timeout, wallclock_left - 30.0)
        if len(self.calls) >= self.max_calls or timeout < 5.0 or self.in_flight() >= self.max_in_flight:
            return None
        call = Call(self, tag, messages, timeout)
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
