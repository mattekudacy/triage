"""
tests/test_classifier_llm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for LLMClassifier. Uses unittest.mock to patch the Anthropic client —
no real API calls are made.
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


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_client_mock(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(response_text)
    return client


# ---------------------------------------------------------------------------
# Correct classification for each FailureType
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ft", list(FailureType))
def test_classifies_each_failure_type(ft):
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _make_client_mock(ft.value)
        result = clf.classify(traj(make_step(0)), "task")
    assert result == ft


# ---------------------------------------------------------------------------
# Fallback to UNKNOWN
# ---------------------------------------------------------------------------

def test_returns_unknown_on_unrecognized_response():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _make_client_mock("not_a_valid_type")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_returns_unknown_on_api_exception():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        client.messages.create.side_effect = Exception("network error")
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


def test_returns_unknown_on_empty_content():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        msg = MagicMock()
        msg.content = []
        client = MagicMock()
        client.messages.create.return_value = msg
        MockAnthropic.return_value = client
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# Client reuse
# ---------------------------------------------------------------------------

def test_client_created_once_and_reused():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _make_client_mock("unknown")
        clf.classify(traj(make_step(0)), "task")
        clf.classify(traj(make_step(0)), "task")
    MockAnthropic.assert_called_once()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_includes_task_and_step_info():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _make_client_mock("unknown")
        MockAnthropic.return_value = client
        step = make_step(0, tool_called="search", error="404")
        clf.classify(traj(step), "find the answer")
    call_kwargs = client.messages.create.call_args
    user_content = call_kwargs[1]["messages"][0]["content"]
    assert "find the answer" in user_content
    assert "search" in user_content
    assert "404" in user_content


def test_max_trajectory_steps_limits_prompt():
    clf = LLMClassifier(max_trajectory_steps=3)
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _make_client_mock("unknown")
        MockAnthropic.return_value = client
        t = traj(*[make_step(i) for i in range(10)])
        clf.classify(t, "task")
    call_kwargs = client.messages.create.call_args
    user_content = call_kwargs[1]["messages"][0]["content"]
    # Only last 3 steps — step indices 7, 8, 9
    assert "[9]" in user_content
    assert "[0]" not in user_content


# ---------------------------------------------------------------------------
# Model parameter
# ---------------------------------------------------------------------------

def test_custom_model_passed_to_client():
    clf = LLMClassifier(model="claude-opus-4-7")
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        client = _make_client_mock("unknown")
        MockAnthropic.return_value = client
        clf.classify(traj(make_step(0)), "task")
    call_kwargs = client.messages.create.call_args
    assert call_kwargs[1]["model"] == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Case-insensitive response parsing
# ---------------------------------------------------------------------------

def test_classify_is_case_insensitive():
    clf = LLMClassifier()
    with patch("triage.classifier.llm._anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _make_client_mock("  LOOP_DETECTED  ")
        result = clf.classify(traj(make_step(0)), "task")
    assert result == FailureType.LOOP_DETECTED
