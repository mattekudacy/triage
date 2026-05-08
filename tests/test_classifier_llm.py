"""
tests/test_classifier_llm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for LLMClassifier — Anthropic backend and OpenAI-compatible backend.
No real API calls are made; both clients are patched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from triage.classifier.llm import LLMClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


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


# ── Anthropic backend ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("ft", list(FailureType))
def test_anthropic_classifies_each_failure_type(ft):
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client(ft.value)
        result = clf.classify(traj(make_step(0)), "task")
    assert result == ft


def test_anthropic_returns_unknown_on_unrecognized_response():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client("not_a_valid_type")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_returns_unknown_on_api_exception():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = Exception("network error")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_returns_unknown_on_empty_content():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        msg = MagicMock()
        msg.content = []
        client = MagicMock()
        client.messages.create.return_value = msg
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_anthropic_client_created_once_and_reused():
    clf = LLMClassifier()
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
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client("  LOOP_DETECTED  ")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.LOOP_DETECTED


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


# ── Shared: prompt construction ───────────────────────────────────────────────

def test_prompt_includes_task_and_step_info():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _anthropic_client("unknown")
        MockAnthropic.return_value = client
        step = make_step(0, tool_called="search", error="404")
        clf.classify(traj(step), "find the answer")
    user_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "find the answer" in user_content
    assert "search" in user_content
    assert "404" in user_content


def test_max_trajectory_steps_limits_prompt():
    clf = LLMClassifier(max_trajectory_steps=3)
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
    clf = LLMClassifier()
    assert clf._api_key == "test-key"


def test_explicit_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://env-url/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "env-model")
    clf = LLMClassifier(base_url="http://explicit/v1", model="explicit-model")
    assert clf._base_url == "http://explicit/v1"
    assert clf._model == "explicit-model"


def test_default_model_anthropic_when_no_base_url(monkeypatch):
    monkeypatch.delenv("TRIAGE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRIAGE_LLM_MODEL", raising=False)
    clf = LLMClassifier()
    assert clf._model == "claude-haiku-4-5-20251001"


def test_default_model_llama_when_base_url_set_via_env(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("TRIAGE_LLM_MODEL", raising=False)
    clf = LLMClassifier()
    assert clf._model == "llama3.2"


def test_env_base_url_routes_to_openai_backend(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "llama3.2")
    clf = LLMClassifier()
    with patch(_OPENAI_PATCH) as MockOpenAI:
        MockOpenAI.return_value = _openai_client("unknown")
        clf.classify(traj(make_step(0)), "task")
    MockOpenAI.assert_called_once()
    assert MockOpenAI.call_args[1]["base_url"] == "http://localhost:11434/v1"
