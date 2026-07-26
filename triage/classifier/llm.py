"""
triage.classifier.llm
~~~~~~~~~~~~~~~~~~~~~
Semantic failure classifier using an LLM backend.

Two backends are supported:

  Anthropic (default)::

      clf = LLMClassifier()
      clf = LLMClassifier(api_key="sk-ant-...", model="claude-haiku-4-5-20251001")

  OpenAI-compatible (Ollama, Groq, OpenAI, any base_url)::

      clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
      clf = LLMClassifier(base_url="https://api.groq.com/openai/v1",
                          api_key="gsk_...", model="llama-3.1-8b-instant")

Both paths use a synchronous client so they work inside a running async event
loop without calling asyncio.run(). Called only on failure — not in the per-step
hot path — so the ~100-400ms blocking latency is acceptable.

Install:
    pip install triage-agent[anthropic]          # Anthropic backend
    pip install triage-agent[openai]             # OpenAI-compatible backend
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import anyio

from triage.taxonomy import FailureType
from triage.trajectory import Trajectory
from triage.usage import Usage

# Exception "shapes" worth retrying — checked by attribute/name rather than by
# importing anthropic/openai's exception classes directly, so this works for
# whichever backend (or neither, if the SDK isn't installed) is in play.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
_RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
}


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    return type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


# Lazy module-level imports — None when the package is not installed.
# Keeping them at module scope (rather than inside _get_client) lets tests
# patch triage.classifier.llm._anthropic / triage.classifier.llm._openai.
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]

_FAILURE_TYPE_VALUES = [ft.value for ft in FailureType]

_SYSTEM_PROMPT = (
    "You are a failure classifier for AI agents. "
    "Given a trajectory of steps and a task description, classify the failure "
    "into exactly one of these categories: "
    + ", ".join(_FAILURE_TYPE_VALUES)
    + '. Respond with only the category name (e.g. "wrong_tool_called"), nothing else.'
)


class LLMClassifier:
    """Semantic failure classifier backed by an LLM.

    Satisfies the ``Classifier`` protocol (synchronous ``classify`` method).

    When ``base_url`` is ``None`` (default), uses ``anthropic.Anthropic``
    (requires ``pip install triage-agent[anthropic]``).

    When ``base_url`` is set, uses ``openai.OpenAI`` pointed at that base URL —
    compatible with Ollama, Groq, OpenAI, and any OpenAI-compatible provider
    (requires ``pip install triage-agent[openai]`` or ``pip install openai``).

    Falls back to ``FailureType.UNKNOWN`` on any error (network, parse, rate limit).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_trajectory_steps: int = 10,
        base_url: str | None = None,
        max_retries: int = 1,
        retry_backoff_base: float = 0.5,
    ) -> None:
        # Explicit args take precedence; env vars are the fallback.
        self._base_url = base_url or os.environ.get("TRIAGE_LLM_BASE_URL") or None
        self._api_key = api_key or os.environ.get("TRIAGE_LLM_API_KEY") or None
        self._model = (
            model
            or os.environ.get("TRIAGE_LLM_MODEL")
            or ("claude-haiku-4-5-20251001" if self._base_url is None else "llama3.2")
        )
        self._max_trajectory_steps = max_trajectory_steps
        # Retries only kick in for transient errors (429/5xx/timeout/connection) —
        # classification runs on the failure path, so this budget stays small by
        # default (1 retry, 0.5s base backoff) to avoid compounding latency on
        # top of an agent that's already failing.
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._client: Any = None
        self._async_client: Any = None
        self._lock = threading.Lock()
        self._async_lock = anyio.Lock()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                self._client = self._build_client()
        return self._client

    async def _get_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        async with self._async_lock:
            if self._async_client is None:
                self._async_client = self._build_async_client()
        return self._async_client

    def _build_client(self) -> Any:
        if self._base_url is not None:
            try:
                import openai as _oi
            except ImportError as exc:
                raise ImportError(
                    "LLMClassifier with base_url requires 'openai'. "
                    "Install it with: pip install openai"
                ) from exc
            return _oi.OpenAI(
                api_key=self._api_key or "no-key",
                base_url=self._base_url,
            )
        if _anthropic is None:
            raise ImportError(
                "LLMClassifier requires 'anthropic'. "
                "Install it with: pip install triage-agent[anthropic]"
            )
        return _anthropic.Anthropic(api_key=self._api_key)

    def _build_async_client(self) -> Any:
        if self._base_url is not None:
            try:
                import openai as _oi
            except ImportError as exc:
                raise ImportError(
                    "LLMClassifier with base_url requires 'openai'. "
                    "Install it with: pip install openai"
                ) from exc
            return _oi.AsyncOpenAI(
                api_key=self._api_key or "no-key",
                base_url=self._base_url,
            )
        if _anthropic is None:
            raise ImportError(
                "LLMClassifier requires 'anthropic'. "
                "Install it with: pip install triage-agent[anthropic]"
            )
        return _anthropic.AsyncAnthropic(api_key=self._api_key)

    def _build_prompt(self, trajectory: Trajectory, task: str) -> str:
        steps = trajectory.last_n_steps(self._max_trajectory_steps)
        lines = [f"Task: {task}", "", "Recent steps:"]
        for step in steps:
            lines.append(f"[{step.index}] {step.action}")
            if step.tool_called:
                lines.append(f"  tool: {step.tool_called}")
            if step.error:
                lines.append(f"  error: {step.error}")
            if step.llm_output:
                lines.append(f"  llm_output: {step.llm_output[:200]}")
        lines.append("")
        lines.append("Classify the failure type:")
        return "\n".join(lines)

    def _report_usage(self, response: object) -> None:
        """Push token/cost usage from a sync or async response to the run meter.

        Duck-typed so backends that don't expose `.usage` are silently skipped.
        The usage recorder is read from the contextvar set by Agent.run() — if
        called outside a triage run (e.g. standalone benchmark), this is a no-op.
        """
        try:
            from triage.agent import _record_usage_var  # lazy to avoid circular import

            record_fn = _record_usage_var.get()
            if record_fn is None:
                return
            u = getattr(response, "usage", None)
            if u is None:
                return
            # Anthropic: .input_tokens / .output_tokens
            # OpenAI:    .prompt_tokens / .completion_tokens
            input_t = getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", 0) or 0
            output_t = getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", 0) or 0
            record_fn(Usage(input_tokens=int(input_t), output_tokens=int(output_t)))
        except Exception:
            pass  # usage reporting is best-effort; never break classification

    def _parse_response(self, raw: str) -> FailureType:
        raw = raw.strip().lower()
        for ft in FailureType:
            if ft.value == raw:
                return ft
        return FailureType.UNKNOWN

    def _call_sync(self, prompt: str) -> str:
        client = self._get_client()
        if self._base_url is not None:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=32,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            self._report_usage(response)
            return str(response.choices[0].message.content or "")
        message = client.messages.create(
            model=self._model,
            max_tokens=32,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        self._report_usage(message)
        return str(message.content[0].text)

    async def _call_async(self, prompt: str) -> str:
        client = await self._get_async_client()
        if self._base_url is not None:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=32,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            self._report_usage(response)
            return str(response.choices[0].message.content or "")
        message = await client.messages.create(
            model=self._model,
            max_tokens=32,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        self._report_usage(message)
        return str(message.content[0].text)

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        prompt = self._build_prompt(trajectory, task)
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_sync(prompt)
                return self._parse_response(raw)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    return FailureType.UNKNOWN
                time.sleep(self._retry_backoff_base * (2**attempt))
        return FailureType.UNKNOWN

    async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType:
        """Async counterpart to ``classify()`` using the native async SDK client.

        Prefer this over ``classify()`` when calling from async code — it awaits
        the HTTP call directly instead of running the sync client in a thread.
        Same fallback-to-UNKNOWN behavior on any error, and the same retry
        budget for transient errors (429/5xx/timeout/connection).
        """
        prompt = self._build_prompt(trajectory, task)
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._call_async(prompt)
                return self._parse_response(raw)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    return FailureType.UNKNOWN
                await anyio.sleep(self._retry_backoff_base * (2**attempt))
        return FailureType.UNKNOWN
