"""
triage.classifier.llm
~~~~~~~~~~~~~~~~~~~~~
Semantic failure classifier using an LLM backend.

Two backends are supported. Neither is "the default" in the sense of being
assumed for you — ``LLMClassifier`` always requires an explicit ``model``
(constructor arg or ``TRIAGE_LLM_MODEL`` env var) and raises ``ValueError``
otherwise, precisely so it never silently guesses which vendor you meant.
Ollama is listed first because it needs no account or key — pass ``base_url``
to use it (or any other OpenAI-compatible endpoint); omit ``base_url``
entirely for Anthropic. See ``docs/concepts/classifiers.md``'s "Open by
default" note for why this ordering matters: ``RulesClassifier`` (triage's
actual default classifier) already makes zero API calls to any vendor, and
this class's own scripts (``scripts/*_accuracy.py``) default to local Ollama
for the same reason — a paid key should never be the only way to try this
out.

  OpenAI-compatible (Ollama, Groq, OpenAI, any base_url) — no account needed
  for Ollama::

      clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
      clf = LLMClassifier(base_url="https://api.groq.com/openai/v1",
                          api_key="gsk_...", model="llama-3.1-8b-instant")

  Anthropic — omit base_url::

      clf = LLMClassifier(model="claude-haiku-4-5-20251001")  # reads ANTHROPIC_API_KEY
      clf = LLMClassifier(api_key="sk-ant-...", model="claude-haiku-4-5-20251001")

Both paths use a synchronous client so they work inside a running async event
loop without calling asyncio.run(). Called only on failure — not in the per-step
hot path — so the ~100-400ms blocking latency is acceptable.

The classification call defaults to ``max_tokens=32`` — enough for a plain
instruct model to emit one category word, but too small for a *reasoning*
model (gpt-oss, o1/o3-style, DeepSeek-R1, Qwen3 "thinking" mode, ...), which
can spend the entire budget on hidden reasoning tokens before ever emitting
the answer. That failure mode is silent: the API call succeeds, the response
just has empty content, so ``classify()`` returns ``UNKNOWN`` with no error at
all. If you're pointing this at a reasoning model, pass a larger
``max_tokens`` explicitly (a few hundred is usually enough) or set
``TRIAGE_LLM_MAX_TOKENS``::

    clf = LLMClassifier(base_url="https://ollama.com/v1",
                        model="gpt-oss:120b-cloud", max_tokens=500)

Install:
    pip install triage-agent[openai]             # OpenAI-compatible backend (Ollama, Groq, ...)
    pip install triage-agent[anthropic]          # Anthropic backend
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import anyio

from triage.classifier.base import ClassificationResult
from triage.pricing import lookup_cost
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

# The explicit "unknown is correct, not a fallback to avoid" guidance below was
# added in response to a measured failure — scripts/hybrid_ambiguity_accuracy.py
# scored a 100% override rate on genuinely out-of-taxonomy entries beforehand —
# but re-measuring AFTER this change found it made no difference: still 12/12
# = 100% override rate. One entry's wrong guess moved from "external_fault" to
# "constraint_ignored"; nothing moved to "unknown". Kept anyway (harmless, no
# regression on corpus D's routing-sensitive recall either), but do NOT treat
# this as the fix — it isn't one. Prompt wording alone does not appear to move
# this model off its bias toward a specific-sounding guess. See
# docs/known-limitations.md's "LLMClassifier/HybridClassifier close the recall
# gap, but not the precision gap" section for both measurements (before and
# after), and re-run that script against any future prompt change here — this
# is a real production prompt, not a one-off tuning target. The real fix this
# points toward is a confidence signal the caller can gate on — see
# classify_with_confidence() below, not further prompt iteration on this one.
_CATEGORY_GUIDANCE = (
    "You are a failure classifier for AI agents. "
    "Given a trajectory of steps and a task description, classify the failure "
    "into exactly one of these categories: "
    + ", ".join(_FAILURE_TYPE_VALUES)
    + ". If the trajectory does not clearly support one of the other categories, "
    'respond "unknown" — this is the correct answer when the cause is genuinely '
    "unclear or not covered by the other categories, not a fallback to avoid. "
    'Do not guess a specific category just to avoid answering "unknown". '
)

_SYSTEM_PROMPT = (
    _CATEGORY_GUIDANCE
    + 'Respond with only the category name (e.g. "wrong_tool_called"), nothing else.'
)

# classify_with_confidence()'s prompt: same category guidance as _SYSTEM_PROMPT
# (factored into _CATEGORY_GUIDANCE so the two can't silently drift apart), plus
# a request for a confidence score. This exists specifically because the prompt
# guidance above, tried alone, measurably failed to reduce HybridClassifier's
# override rate on genuinely out-of-taxonomy trajectories (see the comment
# above _SYSTEM_PROMPT) — a categorical answer gives the caller nothing to act
# on when the model is guessing. A confidence score does: HybridClassifier
# with confidence_threshold= set can decline to trust a low-confidence guess
# instead of returning it as-is. Calibrate confidence_threshold against your
# own labeled data (see ROADMAP.md's SystemOneClassifier section) — this
# model's self-reported confidence has not been independently validated as
# calibrated, only as present and parseable.
_CONFIDENCE_SYSTEM_PROMPT = (
    _CATEGORY_GUIDANCE + "Respond with exactly two lines and nothing else: the category name on "
    "the first line, then a confidence score from 0.0 (pure guess) to 1.0 "
    "(certain) on the second line, indicating how confident you are that the "
    "category on the first line is correct. Example response:\n"
    "external_fault\n"
    "0.85"
)

# Leading "-?" matters: without it, "-0.3" matched as "0.3" (the sign simply
# dropped, not rejected), so an out-of-range negative silently became a valid
# in-range positive instead of being clamped to 0.0 downstream. Caught by
# tests/test_classifier_llm.py::test_confidence_parsing_is_conservative_on_malformed_input.
_CONFIDENCE_NUMBER_RE = re.compile(r"(-?\d*\.?\d+)")


class LLMClassifier:
    """Semantic failure classifier backed by an LLM.

    Satisfies the ``Classifier`` protocol (synchronous ``classify`` method).
    Also defines the optional, duck-typed ``classify_with_confidence()`` /
    ``aclassify_with_confidence()`` methods, returning a ``ClassificationResult``
    (failure type + a 0.0-1.0 self-reported confidence) instead of a bare
    ``FailureType`` — see that method's docstring, and
    ``HybridClassifier(confidence_threshold=...)`` for the primary consumer.

    When ``base_url`` is ``None`` (default), uses ``anthropic.Anthropic``
    (requires ``pip install triage-agent[anthropic]``).

    When ``base_url`` is set, uses ``openai.OpenAI`` pointed at that base URL —
    compatible with Ollama, Groq, OpenAI, and any OpenAI-compatible provider
    (requires ``pip install triage-agent[openai]`` or ``pip install openai``).

    ``max_tokens`` (default 32, or ``TRIAGE_LLM_MAX_TOKENS``) bounds the
    classification call's output. 32 suffices for a plain instruct model's
    one-word answer; a reasoning model (gpt-oss, o1/o3-style, DeepSeek-R1,
    Qwen3 "thinking" mode, ...) needs far more or the response is truncated to
    empty content before the answer is ever emitted — silently, since that's
    not an error. Falls back to ``FailureType.UNKNOWN`` on any error (network,
    parse, rate limit) — an empty response from a starved reasoning model
    fails the same way, indistinguishably, unless you've set ``max_tokens``
    high enough for that model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_trajectory_steps: int = 10,
        base_url: str | None = None,
        max_retries: int = 1,
        retry_backoff_base: float = 0.5,
        max_tokens: int | None = None,
    ) -> None:
        # Explicit args take precedence; env vars are the fallback.
        self._base_url = base_url or os.environ.get("TRIAGE_LLM_BASE_URL") or None
        self._api_key = api_key or os.environ.get("TRIAGE_LLM_API_KEY") or None
        resolved_model = model or os.environ.get("TRIAGE_LLM_MODEL")
        if not resolved_model:
            raise ValueError(
                "LLMClassifier requires a model. Pass model= explicitly or set "
                "the TRIAGE_LLM_MODEL environment variable.\n"
                "  Ollama:    LLMClassifier(base_url='http://localhost:11434/v1',\n"
                "                           model='llama3.2')  # no account or key needed\n"
                "  OpenAI:    LLMClassifier(base_url='https://api.openai.com/v1',\n"
                "                           model='gpt-4o-mini')\n"
                "  Anthropic: LLMClassifier(model='claude-haiku-4-5-20251001')"
            )
        self._model = resolved_model
        self._max_trajectory_steps = max_trajectory_steps
        # Retries only kick in for transient errors (429/5xx/timeout/connection) —
        # classification runs on the failure path, so this budget stays small by
        # default (1 retry, 0.5s base backoff) to avoid compounding latency on
        # top of an agent that's already failing.
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        # Output token budget for the classification call itself. 32 is enough
        # for a plain instruct model's one-word answer; a reasoning model needs
        # far more, or the answer gets truncated to empty content before it's
        # ever emitted — see the module docstring. TRIAGE_LLM_MAX_TOKENS lets
        # this be set without a code change, matching TRIAGE_LLM_MODEL etc.
        env_max_tokens = os.environ.get("TRIAGE_LLM_MAX_TOKENS")
        self._max_tokens = max_tokens or (int(env_max_tokens) if env_max_tokens else None) or 32
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
        """Serialize the trajectory into the user-turn prompt.

        Includes ``Step.agent_id`` per step, when set, so a multi-agent
        trajectory doesn't look identical to a single-agent one — a
        prerequisite for any future MAST-mode prompt work (see
        docs/concepts/multi-agent-failures.md's phase 3 scoping), not itself
        a change to what ``classify()`` can return: ``_SYSTEM_PROMPT`` and
        ``_parse_response()`` are untouched, so this still only ever yields
        one of the 9 stable ``FailureType`` members. ``None`` (the default)
        omits the line entirely — zero prompt change for single-agent callers.
        """
        steps = trajectory.last_n_steps(self._max_trajectory_steps)
        lines = [f"Task: {task}", "", "Recent steps:"]
        for step in steps:
            lines.append(f"[{step.index}] {step.action}")
            if step.agent_id:
                lines.append(f"  agent: {step.agent_id}")
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
            in_i, out_i = int(input_t), int(output_t)
            cost = lookup_cost(self._model, in_i, out_i)
            record_fn(Usage(input_tokens=in_i, output_tokens=out_i, cost_usd=cost))
        except Exception:
            pass  # usage reporting is best-effort; never break classification

    def _parse_response(self, raw: str) -> FailureType:
        raw = raw.strip().lower()
        for ft in FailureType:
            if ft.value == raw:
                return ft
        return FailureType.UNKNOWN

    def _parse_confidence_response(self, raw: str) -> ClassificationResult:
        """Parse a classify_with_confidence() response: category on the first
        non-empty line (exact match, same strictness as _parse_response()),
        a confidence number somewhere in the remaining lines. Conservative on
        anything unparseable — an unmatched category or a missing/malformed
        confidence number both resolve to UNKNOWN / 0.0 rather than raising,
        so a caller gating on confidence_threshold safely declines rather than
        crashes or silently trusts a response it couldn't actually read."""
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        failure_type = FailureType.UNKNOWN
        if lines:
            first = lines[0].lower()
            for ft in FailureType:
                if ft.value == first:
                    failure_type = ft
                    break
        confidence = 0.0
        for ln in lines[1:] or lines:
            match = _CONFIDENCE_NUMBER_RE.search(ln)
            if match:
                try:
                    confidence = float(match.group(1))
                except ValueError:
                    confidence = 0.0
                break
        confidence = max(0.0, min(1.0, confidence))
        return ClassificationResult(failure_type=failure_type, confidence=confidence)

    def _call_sync(self, prompt: str, system_prompt: str = _SYSTEM_PROMPT) -> str:
        client = self._get_client()
        if self._base_url is not None:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            self._report_usage(response)
            return str(response.choices[0].message.content or "")
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        self._report_usage(message)
        return str(message.content[0].text)

    async def _call_async(self, prompt: str, system_prompt: str = _SYSTEM_PROMPT) -> str:
        client = await self._get_async_client()
        if self._base_url is not None:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            self._report_usage(response)
            return str(response.choices[0].message.content or "")
        message = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
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

    def classify_with_confidence(self, trajectory: Trajectory, task: str) -> ClassificationResult:
        """Like ``classify()``, but also asks the model to self-report a
        confidence score (0.0-1.0) and returns both as a ``ClassificationResult``.

        Not part of the ``Classifier`` protocol — duck-typed, same pattern as
        ``aclassify()``. Exists specifically so a caller (``HybridClassifier``
        with ``confidence_threshold=`` set, most directly) can decline to trust
        a low-confidence guess instead of returning it as fact. See the comment
        above ``_CONFIDENCE_SYSTEM_PROMPT`` for why this exists instead of more
        prompt tuning on ``classify()`` itself, and note this model's confidence
        has not been independently validated as calibrated — calibrate any
        threshold against your own labeled data before trusting it in production.

        Falls back to ``ClassificationResult(FailureType.UNKNOWN, 0.0)`` on any
        error, same as ``classify()`` falls back to ``FailureType.UNKNOWN`` —
        zero confidence signals "don't trust this" just as clearly as UNKNOWN
        does, so a threshold-gated caller declines correctly either way.
        """
        prompt = self._build_prompt(trajectory, task)
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._call_sync(prompt, system_prompt=_CONFIDENCE_SYSTEM_PROMPT)
                return self._parse_confidence_response(raw)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    return ClassificationResult(FailureType.UNKNOWN, 0.0)
                time.sleep(self._retry_backoff_base * (2**attempt))
        return ClassificationResult(FailureType.UNKNOWN, 0.0)

    async def aclassify_with_confidence(
        self, trajectory: Trajectory, task: str
    ) -> ClassificationResult:
        """Async counterpart to ``classify_with_confidence()`` — see that
        method's docstring. Same native-async-client preference as
        ``aclassify()`` over ``classify()``."""
        prompt = self._build_prompt(trajectory, task)
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._call_async(prompt, system_prompt=_CONFIDENCE_SYSTEM_PROMPT)
                return self._parse_confidence_response(raw)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    return ClassificationResult(FailureType.UNKNOWN, 0.0)
                await anyio.sleep(self._retry_backoff_base * (2**attempt))
        return ClassificationResult(FailureType.UNKNOWN, 0.0)
