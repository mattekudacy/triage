"""
tests/test_agent.py
~~~~~~~~~~~~~~~~~~~
Tests for triage.agent.Agent — run loop, auto_checkpoint drain, recovery
actions, and escalation/abort boundaries.
"""

from __future__ import annotations

from typing import Any

import pytest

from triage.agent import Agent, TriageAbortError, TriageEscalationError
from triage.checkpoint import InMemoryCheckpointStore
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureType, Step


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


# ---------------------------------------------------------------------------
# Strategy helpers (must be async per StrategyFn type)
# ---------------------------------------------------------------------------

def retry_strategy():
    async def _fn(ctx):
        return RecoveryAction.RETRY()
    return _fn


def retry_with_hint(hint: str):
    async def _fn(ctx):
        return RecoveryAction.RETRY(hint=hint)
    return _fn


def replan_strategy(hint: str = "new plan"):
    async def _fn(ctx):
        return RecoveryAction.REPLAN(hint=hint)
    return _fn


def rollback_strategy():
    async def _fn(ctx):
        return RecoveryAction.ROLLBACK(checkpoint_id=None)
    return _fn


def resume_strategy(subgoal: str):
    async def _fn(ctx):
        return RecoveryAction.RESUME(from_subgoal=subgoal)
    return _fn


def escalate_strategy(message: str = "give up"):
    async def _fn(ctx):
        return RecoveryAction.ESCALATE(message=message)
    return _fn


def abort_strategy(reason: str = "no way"):
    async def _fn(ctx):
        return RecoveryAction.ABORT(reason=reason)
    return _fn


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------

async def test_run_returns_result_on_success():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0))
        return f"ok:{task}"

    ag = Agent(agent_fn, FailurePolicy())
    result = await ag.run("mytask")
    assert result == "ok:mytask"


async def test_call_delegates_to_run():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    ag = Agent(agent_fn, FailurePolicy())
    result = await ag("mytask")
    assert result == "ok"


# ---------------------------------------------------------------------------
# auto_checkpoint drain
# ---------------------------------------------------------------------------

async def test_auto_checkpoint_saves_after_successful_run():
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0))
        record_step(make_step(1))
        return "done"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    cp = await store.latest()
    assert cp is not None
    assert len(cp.trajectory_snapshot) == 2


async def test_auto_checkpoint_drained_even_on_failure():
    store = InMemoryCheckpointStore()
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        record_step(make_step(0))
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    cp = await store.latest()
    assert cp is not None


async def test_no_checkpoint_saved_when_auto_checkpoint_disabled():
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0))
        return "done"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=False)
    await ag.run("task")

    cp = await store.latest()
    assert cp is None


# ---------------------------------------------------------------------------
# Recovery — RETRY
# ---------------------------------------------------------------------------

async def test_retry_recovers_on_first_failure():
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("transient")
        return "retried"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy)
    result = await ag.run("task")
    assert result == "retried"
    assert call_count[0] == 2


async def test_hint_injected_on_retry():
    received_hint: list[str | None] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        received_hint.append(kw.get("_triage_hint"))
        if len(received_hint) == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_with_hint("use tool X"))
    ag = Agent(agent_fn, policy)
    await ag.run("task")
    assert received_hint[1] == "use tool X"


# ---------------------------------------------------------------------------
# Recovery — REPLAN
# ---------------------------------------------------------------------------

async def test_replan_injects_triage_hint():
    received: list[str | None] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        received.append(kw.get("_triage_hint"))
        if len(received) == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=replan_strategy("new plan"))
    ag = Agent(agent_fn, policy)
    await ag.run("task")
    assert received[1] == "new plan"


# ---------------------------------------------------------------------------
# Recovery — ROLLBACK
# ---------------------------------------------------------------------------

async def test_rollback_loads_latest_checkpoint():
    store = InMemoryCheckpointStore()
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        record_step(make_step(0))
        if call_count[0] == 1:
            raise RuntimeError("bad state")
        return kw.get("_triage_hint", "no-hint")

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    result = await ag.run("task")
    assert "Rolled back" in result


async def test_rollback_escalates_when_no_checkpoint():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("oops")

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy)
    with pytest.raises(TriageEscalationError):
        await ag.run("task")


# ---------------------------------------------------------------------------
# Recovery — RESUME
# ---------------------------------------------------------------------------

async def test_resume_injects_subgoal():
    received: list[str | None] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        received.append(kw.get("_triage_subgoal"))
        if len(received) == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=resume_strategy("step 3"))
    ag = Agent(agent_fn, policy)
    await ag.run("task")
    assert received[1] == "step 3"


# ---------------------------------------------------------------------------
# Recovery — ESCALATE / ABORT
# ---------------------------------------------------------------------------

async def test_escalate_raises_triage_escalation_error():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("bad")

    policy = FailurePolicy(UNKNOWN=escalate_strategy("give up"))
    ag = Agent(agent_fn, policy)
    with pytest.raises(TriageEscalationError, match="give up"):
        await ag.run("task")


async def test_abort_raises_triage_abort_error():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("bad")

    policy = FailurePolicy(UNKNOWN=abort_strategy("no way"))
    ag = Agent(agent_fn, policy)
    with pytest.raises(TriageAbortError, match="no way"):
        await ag.run("task")


# ---------------------------------------------------------------------------
# Max attempts / escalation
# ---------------------------------------------------------------------------

async def test_escalates_after_max_recovery_attempts():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("always fails")

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, max_recovery_attempts=2)
    with pytest.raises(TriageEscalationError, match="Max recovery attempts"):
        await ag.run("task")


async def test_escalation_error_carries_context():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("always fails")

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, max_recovery_attempts=1)
    with pytest.raises(TriageEscalationError) as exc_info:
        await ag.run("task")
    assert exc_info.value.context.original_task == "task"


# ---------------------------------------------------------------------------
# Trajectory reset between attempts
# ---------------------------------------------------------------------------

async def test_trajectory_resets_each_attempt():
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        record_step(make_step(0))
        record_step(make_step(1))
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy)
    await ag.run("task")
    # After the second attempt succeeds with 2 steps, trajectory has 2 steps (not 4)
    assert len(ag._trajectory.steps) == 2
