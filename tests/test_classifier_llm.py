"""
tests/test_classifier_llm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for LLMClassifier — Anthropic backend and OpenAI-compatible backend.
No real API calls are made; both clients are patched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from triage.classifier.base import ClassificationResult
from triage.classifier.llm import LLMClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

_MODEL = "claude-haiku-4-5-20251001"  # test fixture model — avoids hitting the no-default guard


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
) -> Step:
    return Step(
        index=index,
        action="test step",
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
    )


def traj(*steps: Step) -> Trajectory:
    t = Trajectory()
    for s in steps:
        t.append(s)
    return t


# ── Anthropic mock helpers ────────────────────────────────────────────────────


def _anthropic_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _anthropic_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(response_text)
    return client


def _anthropic_async_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_anthropic_response(response_text))
    return client


# ── OpenAI-compatible mock helpers ───────────────────────────────────────────


def _openai_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _openai_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(response_text)
    return client


def _openai_async_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_openai_response(response_text))
    return client


# ── Anthropic backend ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ft", list(FailureType))
def test_anthropic_classifies_each_failure_type(ft):
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client(ft.value)
        result = clf.classify(traj(make_step(0)), "task")
    assert result == ft


def test_anthropic_returns_unknown_on_unrecognized_response():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client("not_a_valid_type")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_returns_unknown_on_api_exception():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = Exception("network error")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_returns_unknown_on_empty_content():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        msg = MagicMock()
        msg.content = []
        client = MagicMock()
        client.messages.create.return_value = msg
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_client_created_once_and_reused():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client("unknown")
        clf.classify(traj(make_step(0)), "task")
        clf.classify(traj(make_step(0)), "task")
    MockAnthropic.assert_called_once()


def test_anthropic_custom_model_passed_to_client():
    clf = LLMClassifier(model="claude-opus-4-7")
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        clf.classify(traj(make_step(0)), "task")
    assert client.messages.create.call_args[1]["model"] == "claude-opus-4-7"


def test_anthropic_classify_is_case_insensitive():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client("  LOOP_DETECTED  ")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.LOOP_DETECTED


# ── Anthropic backend: aclassify() (native async client) ──────────────────────


@pytest.mark.parametrize("ft", list(FailureType))
async def test_anthropic_aclassify_classifies_each_failure_type(ft):
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        MockAsyncAnthropic.return_value = _anthropic_async_client(ft.value)
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == ft


async def test_anthropic_aclassify_returns_unknown_on_api_exception():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=Exception("network error"))
        MockAsyncAnthropic.return_value = client
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


async def test_anthropic_aclassify_uses_async_client_not_sync():
    """aclassify() must build/use AsyncAnthropic, never the sync Anthropic client."""
    clf = LLMClassifier(model=_MODEL)
    with (
        patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic,
        patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic,
    ):
        MockAsyncAnthropic.return_value = _anthropic_async_client("unknown")
        await clf.aclassify(traj(make_step(0)), "task")
    MockAsyncAnthropic.assert_called_once()
    MockAnthropic.assert_not_called()


async def test_anthropic_aclassify_async_client_created_once_and_reused():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        MockAsyncAnthropic.return_value = _anthropic_async_client("unknown")
        await clf.aclassify(traj(make_step(0)), "task")
        await clf.aclassify(traj(make_step(0)), "task")
    MockAsyncAnthropic.assert_called_once()


async def test_sync_and_async_clients_are_independent():
    """Calling both classify() and aclassify() builds separate sync/async clients."""
    clf = LLMClassifier(model=_MODEL)
    with (
        patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic,
        patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic,
    ):
        MockAnthropic.return_value = _anthropic_client("unknown")
        MockAsyncAnthropic.return_value = _anthropic_async_client("unknown")
        clf.classify(traj(make_step(0)), "task")
        await clf.aclassify(traj(make_step(0)), "task")
    MockAnthropic.assert_called_once()
    MockAsyncAnthropic.assert_called_once()


# ── OpenAI-compatible backend ─────────────────────────────────────────────────

_openai_mod = pytest.importorskip("openai", reason="openai not installed")

# Patch target: "openai.OpenAI" — works regardless of when triage.classifier.llm
# was first imported, because we patch the canonical source, not the module alias.
_OPENAI_PATCH = "openai.OpenAI"


@pytest.mark.parametrize("ft", list(FailureType))
def test_openai_compat_classifies_each_failure_type(ft):
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client(ft.value)
        result = clf.classify(traj(make_step(0)), "task")
    assert result == ft


def test_openai_compat_returns_unknown_on_unrecognized_response():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("not_a_valid_type")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_openai_compat_returns_unknown_on_exception():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("connection refused")
        MockOpenAI.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_openai_compat_client_created_once_and_reused():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("unknown")
        clf.classify(traj(make_step(0)), "task")
        clf.classify(traj(make_step(0)), "task")
    MockOpenAI.assert_called_once()


def test_openai_compat_base_url_and_api_key_passed_to_client():
    clf = LLMClassifier(
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test",
        model="llama-3.1-8b-instant",
    )
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("unknown")
        clf.classify(traj(make_step(0)), "task")
    call_kwargs = MockOpenAI.call_args[1]
    assert call_kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert call_kwargs["api_key"] == "gsk_test"


def test_openai_compat_custom_model_passed_to_completions():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="mistral")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = _openai_client("unknown")
        MockOpenAI.return_value = client
        clf.classify(traj(make_step(0)), "task")
    assert client.chat.completions.create.call_args[1]["model"] == "mistral"


def test_openai_compat_system_prompt_in_messages():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = _openai_client("unknown")
        MockOpenAI.return_value = client
        clf.classify(traj(make_step(0)), "task")
    messages = client.chat.completions.create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_openai_compat_no_api_key_defaults_to_placeholder():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("unknown")
        clf.classify(traj(make_step(0)), "task")
    assert MockOpenAI.call_args[1]["api_key"] == "no-key"


# ── OpenAI-compatible backend: aclassify() (native async client) ──────────────

_ASYNC_OPENAI_PATCH = "openai.AsyncOpenAI"


@pytest.mark.parametrize("ft", list(FailureType))
async def test_openai_compat_aclassify_classifies_each_failure_type(ft):
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI:
        MockAsyncOpenAI.return_value = _openai_async_client(ft.value)
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == ft


async def test_openai_compat_aclassify_returns_unknown_on_exception():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("connection refused"))
        MockAsyncOpenAI.return_value = client
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


async def test_openai_compat_aclassify_uses_async_client_not_sync():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI, patch(_OPENAI_PATCH) as MockOpenAI:
        MockAsyncOpenAI.return_value = _openai_async_client("unknown")
        await clf.aclassify(traj(make_step(0)), "task")
    MockAsyncOpenAI.assert_called_once()
    MockOpenAI.assert_not_called()


# ── Retry on transient errors ──────────────────────────────────────────────────
# max_retries defaults to 1. Retryable errors are identified by status_code in
# {429, 500, 502, 503, 529} or by exception class name (RateLimitError,
# APITimeoutError, APIConnectionError, InternalServerError) — checked structurally
# so this works without importing the real anthropic/openai exception classes.


class RateLimitError(Exception):
    """Mimics the real anthropic/openai RateLimitError by class name only —
    _is_retryable() checks type(exc).__name__, not isinstance."""


class _FakeStatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_classify_retries_once_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = [
            RateLimitError("rate limited"),
            _anthropic_response("timeout"),
        ]
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.TIMEOUT
    assert client.messages.create.call_count == 2


def test_classify_gives_up_after_max_retries_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    clf = LLMClassifier(model=_MODEL, max_retries=1)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = RateLimitError("still rate limited")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN
    assert client.messages.create.call_count == 2  # initial + 1 retry


def test_classify_retries_on_retryable_status_code():
    clf = LLMClassifier(model=_MODEL)
    with (
        patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic,
        patch("time.sleep") as mock_sleep,
    ):
        client = MagicMock()
        client.messages.create.side_effect = [
            _FakeStatusError("service unavailable", status_code=503),
            _anthropic_response("external_fault"),
        ]
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.EXTERNAL_FAULT
    mock_sleep.assert_called_once()


def test_classify_does_not_retry_non_retryable_error():
    """A non-transient error (e.g. malformed request) must not consume a retry."""
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = ValueError("bad request")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN
    assert client.messages.create.call_count == 1  # no retry attempted


def test_classify_max_retries_zero_disables_retry():
    clf = LLMClassifier(model=_MODEL, max_retries=0)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = RateLimitError("rate limited")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN
    assert client.messages.create.call_count == 1


async def test_aclassify_retries_once_on_rate_limit_then_succeeds(monkeypatch):
    import anyio as _anyio

    async def _no_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr(_anyio, "sleep", _no_sleep)

    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        client = MagicMock()
        client.messages.create = AsyncMock(
            side_effect=[RateLimitError("rate limited"), _anthropic_response("loop_detected")]
        )
        MockAsyncAnthropic.return_value = client
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == FailureType.LOOP_DETECTED
    assert client.messages.create.call_count == 2


async def test_aclassify_gives_up_after_max_retries_exhausted(monkeypatch):
    import anyio as _anyio

    async def _no_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr(_anyio, "sleep", _no_sleep)

    clf = LLMClassifier(model=_MODEL, max_retries=1)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RateLimitError("still rate limited"))
        MockAsyncAnthropic.return_value = client
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN
    assert client.messages.create.call_count == 2


async def test_aclassify_does_not_retry_non_retryable_error():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=ValueError("bad request"))
        MockAsyncAnthropic.return_value = client
        result = await clf.aclassify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN
    assert client.messages.create.call_count == 1


# ── Shared: prompt construction ───────────────────────────────────────────────


def test_prompt_includes_task_and_step_info():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        step = make_step(0, tool_called="search", error="404")
        clf.classify(traj(step), "find the answer")
    user_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "find the answer" in user_content
    assert "search" in user_content
    assert "404" in user_content


def test_prompt_includes_agent_id_when_set():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        step = Step(index=0, action="call tool", tool_called="search", agent_id="planner")
        clf.classify(traj(step), "find the answer")
    user_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "agent: planner" in user_content


def test_prompt_omits_agent_line_when_agent_id_unset():
    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        clf.classify(traj(make_step(0, tool_called="search")), "find the answer")
    user_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "agent:" not in user_content


def test_max_trajectory_steps_limits_prompt():
    clf = LLMClassifier(model=_MODEL, max_trajectory_steps=3)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        t = traj(*[make_step(i) for i in range(10)])
        clf.classify(t, "task")
    user_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[9]" in user_content
    assert "[0]" not in user_content


# ── BYOK env vars ─────────────────────────────────────────────────────────────


def test_env_var_base_url_used_when_no_arg(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "llama3.2")
    clf = LLMClassifier()
    assert clf._base_url == "http://localhost:11434/v1"
    assert clf._model == "llama3.2"


def test_env_var_api_key_used_when_no_arg(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_API_KEY", "test-key")
    monkeypatch.delenv("TRIAGE_LLM_BASE_URL", raising=False)
    clf = LLMClassifier(model=_MODEL)
    assert clf._api_key == "test-key"


def test_explicit_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://env-url/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "env-model")
    clf = LLMClassifier(base_url="http://explicit/v1", model="explicit-model")
    assert clf._base_url == "http://explicit/v1"
    assert clf._model == "explicit-model"


def test_no_model_and_no_env_raises_value_error(monkeypatch):
    monkeypatch.delenv("TRIAGE_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLMClassifier requires a model"):
        LLMClassifier()


def test_model_from_env_var_is_accepted(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    clf = LLMClassifier()
    assert clf._model == "claude-haiku-4-5-20251001"


def test_env_base_url_routes_to_openai_backend(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "llama3.2")
    clf = LLMClassifier(model=_MODEL)
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("unknown")
        clf.classify(traj(make_step(0)), "task")
    MockOpenAI.assert_called_once()
    assert MockOpenAI.call_args[1]["base_url"] == "http://localhost:11434/v1"


# ── max_tokens: configurable output budget (reasoning-model fix) ──────────────
# LLMClassifier hardcoded max_tokens=32 for the classification call. Fine for
# a plain instruct model's one-word answer; silently wrong for a reasoning
# model, which can spend the whole budget on hidden reasoning tokens and
# return empty content — classify() then returns UNKNOWN with no error at
# all, indistinguishable from a real auth/network failure. Found via a real
# gpt-oss:120b-cloud run: finish_reason="length", content="" at 32 tokens;
# finish_reason="stop", content="ok" at 500. These tests pin the fix, not
# just the default.


def test_max_tokens_default_is_32():
    clf = LLMClassifier(model=_MODEL)
    assert clf._max_tokens == 32


def test_max_tokens_explicit_arg_overrides_default():
    clf = LLMClassifier(model=_MODEL, max_tokens=500)
    assert clf._max_tokens == 500


def test_max_tokens_env_var_used_when_no_arg(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_MAX_TOKENS", "500")
    clf = LLMClassifier(model=_MODEL)
    assert clf._max_tokens == 500


def test_max_tokens_explicit_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_MAX_TOKENS", "500")
    clf = LLMClassifier(model=_MODEL, max_tokens=999)
    assert clf._max_tokens == 999


def test_max_tokens_passed_to_anthropic_client():
    clf = LLMClassifier(model=_MODEL, max_tokens=500)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        clf.classify(traj(make_step(0)), "task")
    assert client.messages.create.call_args[1]["max_tokens"] == 500


async def test_max_tokens_passed_to_anthropic_async_client():
    clf = LLMClassifier(model=_MODEL, max_tokens=500)
    with patch("triage.classifier.llm._anthropic.AsyncAnthropic") as MockAsyncAnthropic:
        client = _anthropic_async_client("unknown")
        MockAsyncAnthropic.return_value = client
        await clf.aclassify(traj(make_step(0)), "task")
    assert client.messages.create.call_args[1]["max_tokens"] == 500


def test_max_tokens_passed_to_openai_compat_client():
    clf = LLMClassifier(
        base_url="http://localhost:11434/v1", model="gpt-oss:120b-cloud", max_tokens=500
    )
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = _openai_client("unknown")
        MockOpenAI.return_value = client
        clf.classify(traj(make_step(0)), "task")
    assert client.chat.completions.create.call_args[1]["max_tokens"] == 500


async def test_max_tokens_passed_to_openai_compat_async_client():
    clf = LLMClassifier(
        base_url="http://localhost:11434/v1", model="gpt-oss:120b-cloud", max_tokens=500
    )
    with patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI:
        client = _openai_async_client("unknown")
        MockAsyncOpenAI.return_value = client
        await clf.aclassify(traj(make_step(0)), "task")
    assert client.chat.completions.create.call_args[1]["max_tokens"] == 500


def test_max_tokens_truncated_empty_content_returns_unknown():
    """Reproduces the exact failure mode found against gpt-oss:120b-cloud:
    the API call succeeds, content is truncated to empty by finish_reason
    'length' — classify() must return UNKNOWN, not raise."""
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="gpt-oss:120b-cloud")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = _openai_client("")
        MockOpenAI.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


# ── classify_with_confidence() / aclassify_with_confidence() ─────────────────
# Built in response to a measured negative result: telling LLMClassifier's
# plain classify() prompt that "unknown" is a valid answer did not reduce
# HybridClassifier's override rate on genuinely out-of-taxonomy trajectories
# (see docs/known-limitations.md). These tests cover the confidence-scoring
# path itself — HybridClassifier's gating on it is tested separately in
# tests/test_classifier_hybrid.py.


def test_confidence_parses_category_and_score():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("external_fault\n0.85")
        result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result == ClassificationResult(FailureType.EXTERNAL_FAULT, 0.85)


def test_confidence_uses_confidence_prompt_not_plain_system_prompt():
    from triage.classifier.llm import _CONFIDENCE_SYSTEM_PROMPT, _SYSTEM_PROMPT

    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = _openai_client("external_fault\n0.85")
        MockOpenAI.return_value = client
        clf.classify_with_confidence(traj(make_step(0)), "task")
    sent = client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert sent == _CONFIDENCE_SYSTEM_PROMPT
    assert sent != _SYSTEM_PROMPT


@pytest.mark.parametrize(
    "raw,expected_confidence",
    [
        ("external_fault\n1.5", 1.0),  # above range clamps to 1.0
        ("external_fault\n-0.3", 0.0),  # below range clamps to 0.0
        ("external_fault\nnot a number", 0.0),  # unparseable -> conservative 0.0
        ("external_fault", 0.0),  # no second line at all -> conservative 0.0
        ("external_fault\n\n0.6", 0.6),  # blank line between category and score
    ],
)
def test_confidence_parsing_is_conservative_on_malformed_input(raw, expected_confidence):
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client(raw)
        result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result.failure_type == FailureType.EXTERNAL_FAULT
    assert result.confidence == expected_confidence


def test_confidence_unmatched_category_returns_unknown():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("not_a_real_category\n0.9")
        result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result.failure_type == FailureType.UNKNOWN


def test_confidence_returns_unknown_zero_on_exception():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("network error")
        MockOpenAI.return_value = client
        result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result == ClassificationResult(FailureType.UNKNOWN, 0.0)


def test_confidence_retries_once_on_rate_limit_then_succeeds():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI:
        client = MagicMock()
        rate_limit_exc = Exception("rate limited")
        rate_limit_exc.status_code = 429  # type: ignore[attr-defined]
        client.chat.completions.create.side_effect = [
            rate_limit_exc,
            _openai_response("timeout\n0.7"),
        ]
        MockOpenAI.return_value = client
        with patch("triage.classifier.llm.time.sleep"):
            result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result == ClassificationResult(FailureType.TIMEOUT, 0.7)


async def test_aclassify_confidence_parses_category_and_score():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI:
        MockAsyncOpenAI.return_value = _openai_async_client("schema_mismatch\n0.42")
        result = await clf.aclassify_with_confidence(traj(make_step(0)), "task")
    assert result == ClassificationResult(FailureType.SCHEMA_MISMATCH, 0.42)


async def test_aclassify_confidence_uses_async_client_not_sync():
    clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    with patch(_OPENAI_PATCH) as MockOpenAI, patch(_ASYNC_OPENAI_PATCH) as MockAsyncOpenAI:
        MockAsyncOpenAI.return_value = _openai_async_client("unknown\n0.1")
        await clf.aclassify_with_confidence(traj(make_step(0)), "task")
    MockOpenAI.assert_not_called()
    MockAsyncOpenAI.assert_called_once()


def test_confidence_anthropic_backend_uses_confidence_prompt():
    from triage.classifier.llm import _CONFIDENCE_SYSTEM_PROMPT

    clf = LLMClassifier(model=_MODEL)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("wrong_tool_called\n0.9")
        MockAnthropic.return_value = client
        result = clf.classify_with_confidence(traj(make_step(0)), "task")
    assert result == ClassificationResult(FailureType.WRONG_TOOL_CALLED, 0.9)
    assert client.messages.create.call_args[1]["system"] == _CONFIDENCE_SYSTEM_PROMPT
