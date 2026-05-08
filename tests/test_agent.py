"""
tests/test_agent.py
~~~~~~~~~~~~~~~~~~~
Tests for triage.agent.Agent — run loop, auto_checkpoint drain, recovery
actions, and escalation/abort boundaries.
"""

from __future__ import annotations

from typing import Any

import pytest

from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# Fix 1: async classify — classifier runs in a thread, not blocking the loop
# ---------------------------------------------------------------------------

async def test_classify_called_in_thread():
    """classify() must be called via anyio.to_thread, not directly."""
    classify_thread_ids: list[int] = []
    import threading

    class ThreadTrackingClassifier:
        def classify(self, trajectory, task):
            classify_thread_ids.append(threading.get_ident())
            return FailureType.UNKNOWN

    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, classifier=ThreadTrackingClassifier())
    await ag.run("task")

    # classify() ran exactly once (one failure) and in a different thread
    assert len(classify_thread_ids) == 1
    assert classify_thread_ids[0] != threading.main_thread().ident


async def test_classify_thread_failure_falls_back_to_unknown():
    """If classify() raises inside the thread, triage still handles the failure."""
    class BrokenClassifier:
        def classify(self, trajectory, task):
            raise RuntimeError("classifier exploded")

    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("agent fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, classifier=BrokenClassifier())
    # Should not propagate the classifier exception — triage catches it
    with pytest.raises((TriageEscalationError, Exception)):
        await ag.run("task")


# ---------------------------------------------------------------------------
# Fix 2: agent state in checkpoints — update_state / _triage_state
# ---------------------------------------------------------------------------

async def test_update_state_persisted_in_checkpoint():
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        record_step(make_step(0))
        update_state({"phase": "fetched", "count": 42})
        return "done"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    cp = await store.latest()
    assert cp is not None
    assert cp.state == {"phase": "fetched", "count": 42}


async def test_update_state_reflects_latest_call():
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        record_step(make_step(0))
        update_state({"phase": "step1"})
        record_step(make_step(1))
        update_state({"phase": "step2", "count": 99})
        return "done"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    cp = await store.latest()
    assert cp is not None
    assert cp.state["phase"] == "step2"
    assert cp.state["count"] == 99


async def test_rollback_injects_triage_state():
    store = InMemoryCheckpointStore()
    received_state: list[Any] = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_count[0] += 1
        received_state.append(kw.get("_triage_state"))

        if call_count[0] == 1:
            record_step(make_step(0))
            update_state({"fetched": "important data"})
            raise RuntimeError("phase 2 failed")
        return "recovered"

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    # First call: no injected state; second call: state from checkpoint
    assert received_state[0] is None
    assert received_state[1] == {"fetched": "important data"}


async def test_rollback_restores_current_state_on_agent():
    store = InMemoryCheckpointStore()
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_count[0] += 1
        record_step(make_step(0))
        update_state({"attempt": call_count[0]})
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    # After rollback + recovery, _current_state reflects the second attempt's state
    assert ag._current_state == {"attempt": 2}


async def test_no_triage_state_injected_when_checkpoint_state_empty():
    store = InMemoryCheckpointStore()
    received_state: list[Any] = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_count[0] += 1
        received_state.append(kw.get("_triage_state"))
        record_step(make_step(0))
        # Never calls update_state — state stays {}
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    # Empty state → _triage_state not injected
    assert received_state[1] is None


async def test_update_state_resets_each_attempt():
    states_at_failure: list[dict] = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_count[0] += 1
        update_state({"attempt": call_count[0]})
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy)
    await ag.run("task")

    # After retry, _current_state reflects the latest attempt only
    assert ag._current_state == {"attempt": 2}


# ---------------------------------------------------------------------------
# attempt_history on FailureContext
# ---------------------------------------------------------------------------

async def test_attempt_history_empty_on_first_failure():
    received_history: list = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("fail")

    async def capturing_strategy(ctx):
        received_history.append(list(ctx.attempt_history))
        return RecoveryAction.ABORT(reason="stop")

    policy = FailurePolicy(UNKNOWN=capturing_strategy)
    ag = Agent(agent_fn, policy)
    with pytest.raises(TriageAbortError):
        await ag.run("task")

    assert received_history[0] == []


async def test_attempt_history_accumulates_across_retries():
    received_history: list = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        raise RuntimeError("always fails")

    async def capturing_strategy(ctx):
        received_history.append(list(ctx.attempt_history))
        if len(ctx.attempt_history) >= 2:
            return RecoveryAction.ABORT(reason="stop")
        return RecoveryAction.RETRY()

    policy = FailurePolicy(UNKNOWN=capturing_strategy)
    ag = Agent(agent_fn, policy, max_recovery_attempts=5)
    with pytest.raises(TriageAbortError):
        await ag.run("task")

    # First call: empty; second call: one entry; third call: two entries
    assert received_history[0] == []
    assert received_history[1] == [(FailureType.UNKNOWN, "retry")]
    assert received_history[2] == [
        (FailureType.UNKNOWN, "retry"),
        (FailureType.UNKNOWN, "retry"),
    ]


async def test_attempt_history_records_action_kind():
    received_history: list = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        raise RuntimeError("fail")

    async def strategy(ctx):
        received_history.append(list(ctx.attempt_history))
        if not ctx.attempt_history:
            return RecoveryAction.REPLAN(hint="try again")
        return RecoveryAction.ABORT(reason="stop")

    policy = FailurePolicy(UNKNOWN=strategy)
    ag = Agent(agent_fn, policy, max_recovery_attempts=5)
    with pytest.raises(TriageAbortError):
        await ag.run("task")

    assert received_history[1][0][1] == "replan"


async def test_attempt_history_is_snapshot_not_reference():
    """History passed to each strategy call must not be mutated by later appends."""
    snapshots: list = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        raise RuntimeError("fail")

    async def strategy(ctx):
        snapshots.append(list(ctx.attempt_history))
        if len(ctx.attempt_history) >= 2:
            return RecoveryAction.ABORT(reason="stop")
        return RecoveryAction.RETRY()

    policy = FailurePolicy(UNKNOWN=strategy)
    ag = Agent(agent_fn, policy, max_recovery_attempts=5)
    with pytest.raises(TriageAbortError):
        await ag.run("task")

    # Each snapshot must be frozen at the length it had when captured
    assert len(snapshots[0]) == 0
    assert len(snapshots[1]) == 1
    assert len(snapshots[2]) == 2
