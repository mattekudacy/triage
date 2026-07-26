"""
tests/test_testing.py
~~~~~~~~~~~~~~~~~~~~~
Tests for triage.testing — make_step, RecordingAgent, assert_classifies_as.
"""

from __future__ import annotations

import pytest

from triage.agent import Agent
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureType, Step
from triage.testing import RecordingAgent, assert_classifies_as, make_step

# ── make_step ────────────────────────────────────────────────────────────────

def test_make_step_defaults():
    s = make_step()
    assert s.index == 0
    assert s.action == "test step"
    assert s.tool_called is None
    assert s.tool_input is None
    assert s.error is None
    assert s.llm_output is None


def test_make_step_populates_fields():
    s = make_step(index=3, tool_called="search", tool_input={"q": "x"},
                  error="oops", llm_output="hello")
    assert s.index == 3
    assert s.tool_called == "search"
    assert s.tool_input == {"q": "x"}
    assert s.error == "oops"
    assert s.llm_output == "hello"


def test_make_step_custom_action():
    s = make_step(action="fetch_data")
    assert s.action == "fetch_data"


def test_make_step_returns_step_instance():
    assert isinstance(make_step(), Step)


# ── RecordingAgent ───────────────────────────────────────────────────────────

async def test_recording_agent_succeeds_immediately():
    fn = RecordingAgent()
    result = await fn("task", record_step=lambda s: None, update_state=lambda d: None)
    assert result == "ok"
    assert len(fn.calls) == 1


async def test_recording_agent_records_kwargs():
    fn = RecordingAgent()
    def sentinel(s):
        return None
    await fn("task", record_step=sentinel, update_state=lambda d: None, extra="val")
    assert fn.calls[0]["extra"] == "val"
    assert fn.calls[0]["record_step"] is sentinel


async def test_recording_agent_fails_then_succeeds():
    async def retry(ctx): return RecoveryAction.RETRY()
    fn = RecordingAgent(succeed_after=2)
    policy = FailurePolicy(default=FailurePolicy.escalate_by_default(), UNKNOWN=retry)
    ag = Agent(fn, policy, max_recovery_attempts=5)
    result = await ag.run("task")
    assert result == "ok"
    assert len(fn.calls) == 3  # 2 failures + 1 success


async def test_recording_agent_custom_result():
    fn = RecordingAgent(result={"answer": 42})
    result = await fn("task", record_step=lambda s: None, update_state=lambda d: None)
    assert result == {"answer": 42}


async def test_recording_agent_records_triage_context():
    async def retry(ctx): return RecoveryAction.RETRY(hint="try again")
    fn = RecordingAgent(succeed_after=1)
    policy = FailurePolicy(default=FailurePolicy.escalate_by_default(), UNKNOWN=retry)
    ag = Agent(fn, policy, max_recovery_attempts=3)
    await ag.run("task")

    # First call: no triage context
    assert fn.calls[0].get("_triage_context") is None
    # Second call: triage context injected
    tc = fn.calls[1].get("_triage_context")
    assert tc is not None
    assert tc.failure_type == FailureType.UNKNOWN
    assert tc.hint == "try again"
    assert tc.attempt_number == 0


async def test_recording_agent_custom_error():
    async def retry(ctx): return RecoveryAction.RETRY()
    fn = RecordingAgent(succeed_after=1, error=ValueError("bad input"))
    policy = FailurePolicy(default=FailurePolicy.escalate_by_default(), UNKNOWN=retry)
    ag = Agent(fn, policy, max_recovery_attempts=3)
    await ag.run("task")
    assert len(fn.calls) == 2


# ── assert_classifies_as ──────────────────────────────────────────────────────

def test_assert_classifies_as_loop_detected():
    steps = [
        make_step(index=0, tool_called="search", tool_input={"q": "x"}),
        make_step(index=1, tool_called="search", tool_input={"q": "x"}),
        make_step(index=2, tool_called="search", tool_input={"q": "x"}),
    ]
    assert_classifies_as(steps, "find info", FailureType.LOOP_DETECTED)


def test_assert_classifies_as_raises_on_mismatch():
    steps = [
        make_step(index=0, tool_called="search", tool_input={"q": "x"}),
        make_step(index=1, tool_called="search", tool_input={"q": "x"}),
        make_step(index=2, tool_called="search", tool_input={"q": "x"}),
    ]
    with pytest.raises(AssertionError, match="loop_detected"):
        assert_classifies_as(steps, "find info", FailureType.EXTERNAL_FAULT)


def test_assert_classifies_as_wrong_tool():
    steps = [make_step(index=0, error="tool 'magic' not found")]
    assert_classifies_as(steps, "do task", FailureType.WRONG_TOOL_CALLED)


def test_assert_classifies_as_unknown():
    steps = [make_step(index=0)]
    assert_classifies_as(steps, "task", FailureType.UNKNOWN)


def test_assert_classifies_as_custom_classifier():
    from triage.classifier.rules import RulesClassifier
    clf = RulesClassifier(loop_window=2)
    steps = [
        make_step(index=0, tool_called="search", tool_input={"q": "x"}),
        make_step(index=1, tool_called="search", tool_input={"q": "x"}),
    ]
    assert_classifies_as(steps, "task", FailureType.LOOP_DETECTED, classifier=clf)


def test_assert_classifies_as_error_message_includes_actual():
    steps = [make_step(index=0)]
    with pytest.raises(AssertionError, match="unknown"):
        assert_classifies_as(steps, "task", FailureType.LOOP_DETECTED)
