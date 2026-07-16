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

from triage.agent import Agent, TriageAbortError, TriageEscalationError, get_recorder, get_state_updater
from triage.checkpoint import InMemoryCheckpointStore
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureType, Step, TriageContext


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


async def test_auto_checkpoint_snapshots_per_step_not_final_state():
    """Each checkpoint must capture the trajectory length and state at the time
    record_step() was called, not the collapsed final values at drain time."""
    store = InMemoryCheckpointStore()

    async def two_phase(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="phase-1"))
        update_state({"phase": 1})
        record_step(Step(index=1, action="phase-2"))
        update_state({"phase": 2})
        return "ok"

    ag = Agent(two_phase, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("demo")

    checkpoints = sorted(store._store.values(), key=lambda c: c.timestamp)
    assert len(checkpoints) == 2

    traj_lengths = [len(cp.trajectory_snapshot) for cp in checkpoints]
    states = [cp.state for cp in checkpoints]

    assert traj_lengths == [1, 2], f"expected [1, 2], got {traj_lengths}"
    assert states == [{"phase": 1}, {"phase": 2}], f"expected per-step states, got {states}"


async def test_auto_checkpoint_carries_state_forward_when_no_update():
    """A step with no following update_state() must record the last-known
    state (carried forward), not an empty dict."""
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        update_state({"phase": 1})
        record_step(Step(index=0, action="phase-1"))   # state set BEFORE record_step
        record_step(Step(index=1, action="phase-2"))   # no update_state after — carry forward
        return "ok"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("demo")

    checkpoints = sorted(store._store.values(), key=lambda c: c.timestamp)
    states = [cp.state for cp in checkpoints]
    assert states == [{"phase": 1}, {"phase": 1}], f"state not carried forward: {states}"


async def test_auto_checkpoint_state_snapshot_isolated_from_later_mutation():
    """An earlier checkpoint's state must not be mutated by a later
    update_state() call."""
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="phase-1"))
        update_state({"phase": 1})
        record_step(Step(index=1, action="phase-2"))
        update_state({"phase": 2})
        return "ok"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("demo")

    checkpoints = sorted(store._store.values(), key=lambda c: c.timestamp)
    # First checkpoint must still be phase 1 even though phase 2 was set later.
    assert checkpoints[0].state == {"phase": 1}
    assert checkpoints[1].state == {"phase": 2}


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
    with pytest.raises(TriageEscalationError, match="max_recovery_attempts"):
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
# Zero-trajectory fallback
# ---------------------------------------------------------------------------

async def test_zero_trajectory_fallback_synthesizes_step():
    """Agent raises before any record_step call — trajectory must not be empty."""
    captured_trajectories: list = []

    class CapturingClassifier:
        def classify(self, trajectory, task):
            captured_trajectories.append(list(trajectory.steps))
            return FailureType.UNKNOWN

    async def agent_fn(task: str, *, record_step, **kw) -> str:
        raise RuntimeError("503 service unavailable")  # fails before record_step

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, classifier=CapturingClassifier(), max_recovery_attempts=1)
    with pytest.raises((TriageEscalationError, Exception)):
        await ag.run("task")

    assert len(captured_trajectories) > 0
    assert len(captured_trajectories[0]) == 1  # sentinel step was injected
    assert captured_trajectories[0][0].action == "<no steps recorded>"
    assert "503" in captured_trajectories[0][0].error


async def test_zero_trajectory_external_fault_detected():
    """Sentinel step's error text feeds RulesClassifier — EXTERNAL_FAULT detected."""
    async def agent_fn(task: str, *, record_step, **kw) -> str:
        raise RuntimeError("HTTP 503 Service Unavailable")

    results: list = []

    async def capturing_strategy(ctx):
        results.append(ctx.failure_type)
        return RecoveryAction.ABORT(reason="stop")

    policy = FailurePolicy(EXTERNAL_FAULT=capturing_strategy, UNKNOWN=capturing_strategy)
    ag = Agent(agent_fn, policy)
    with pytest.raises(TriageAbortError):
        await ag.run("task")

    assert results[0] == FailureType.EXTERNAL_FAULT


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


async def test_aclassify_preferred_over_thread_classify_when_present():
    """When a classifier defines aclassify(), agent.py awaits it directly and
    never runs classify() in a thread."""
    import threading

    calls: dict[str, list] = {"aclassify": [], "classify": []}

    class AsyncCapableClassifier:
        def classify(self, trajectory, task):
            calls["classify"].append(threading.get_ident())
            return FailureType.UNKNOWN

        async def aclassify(self, trajectory, task):
            calls["aclassify"].append(threading.get_ident())
            return FailureType.UNKNOWN

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("fail")

    policy = FailurePolicy(UNKNOWN=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, classifier=AsyncCapableClassifier())
    with pytest.raises(TriageEscalationError):
        await ag.run("task")

    assert len(calls["aclassify"]) == 1
    assert len(calls["classify"]) == 0
    # aclassify ran on the main thread (awaited directly, no to_thread hop)
    assert calls["aclassify"][0] == threading.main_thread().ident


async def test_classify_thread_fallback_when_no_aclassify():
    """Classifiers without aclassify() keep using the anyio.to_thread path."""
    import threading

    calls: dict[str, list] = {"classify": []}

    class SyncOnlyClassifier:
        def classify(self, trajectory, task):
            calls["classify"].append(threading.get_ident())
            return FailureType.UNKNOWN

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        raise RuntimeError("fail")

    policy = FailurePolicy(UNKNOWN=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, classifier=SyncOnlyClassifier())
    with pytest.raises(TriageEscalationError):
        await ag.run("task")

    assert len(calls["classify"]) == 1
    assert calls["classify"][0] != threading.main_thread().ident


async def test_run_resets_classifier_call_count_when_present():
    """Agent.run() calls classifier.reset_call_count() (duck-typed) once at the
    start of every run — e.g. so HybridClassifier's max_llm_calls_per_run budget
    is scoped per run rather than per classifier lifetime."""
    reset_calls: list[int] = []

    class BudgetedClassifier:
        def reset_call_count(self) -> None:
            reset_calls.append(1)

        def classify(self, trajectory: Any, task: Any) -> FailureType:
            return FailureType.UNKNOWN

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, classifier=BudgetedClassifier())
    await ag.run("task")
    await ag.run("task")

    assert len(reset_calls) == 2


async def test_run_does_not_require_reset_call_count():
    """Classifiers without reset_call_count() (e.g. RulesClassifier) are unaffected."""
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy)  # default RulesClassifier — no reset_call_count
    result = await ag.run("task")
    assert result == "ok"


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


# ---------------------------------------------------------------------------
# max_total_attempts — cross-type global cap
# ---------------------------------------------------------------------------

async def test_max_total_attempts_escalates_before_loop_cap():
    """max_total_attempts=1 should escalate on the second failure, regardless of type."""
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        raise RuntimeError("always fails")

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, max_recovery_attempts=5, max_total_attempts=1)
    with pytest.raises(TriageEscalationError, match="max_total_attempts"):
        await ag.run("task")

    assert call_count[0] == 2  # initial + 1 retry before cap


async def test_max_total_attempts_none_disables_global_cap():
    """max_total_attempts=None should defer to max_recovery_attempts only."""
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy, max_recovery_attempts=3, max_total_attempts=None)
    result = await ag.run("task")
    assert result == "ok"


# ---------------------------------------------------------------------------
# Agent.clone()
# ---------------------------------------------------------------------------

async def test_clone_shares_policy_and_classifier():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_strategy())
    ag = Agent(agent_fn, policy)
    clone = ag.clone()

    assert clone._policy is ag._policy
    assert clone._classifier is ag._classifier
    assert clone._fn is ag._fn


async def test_clone_has_independent_run_state():
    """Clones must not share mutable run-state."""
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        call_count[0] += 1
        record_step(make_step(0))
        return f"ok:{task}"

    ag = Agent(agent_fn, FailurePolicy())
    clone = ag.clone()

    await ag.run("task-a")
    await clone.run("task-b")

    # Each instance tracked its own trajectory independently
    assert len(ag._trajectory.steps) == 1
    assert len(clone._trajectory.steps) == 1


async def test_clone_inherits_max_total_attempts():
    ag = Agent(lambda task, **kw: None, FailurePolicy(), max_total_attempts=7)
    clone = ag.clone()
    assert clone._max_total_attempts == 7


# ---------------------------------------------------------------------------
# _triage_context injection
# ---------------------------------------------------------------------------

async def test_triage_context_injected_on_retry():
    received: list[Any] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        received.append(kw.get("_triage_context"))
        if len(received) == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=retry_with_hint("use tool X"))
    ag = Agent(agent_fn, policy)
    await ag.run("task")

    assert received[0] is None  # first call: no context yet
    ctx: TriageContext = received[1]
    assert isinstance(ctx, TriageContext)
    assert ctx.failure_type == FailureType.UNKNOWN
    assert ctx.hint == "use tool X"
    assert ctx.attempt_number == 0


async def test_triage_context_injected_on_replan():
    received: list[Any] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        received.append(kw.get("_triage_context"))
        if len(received) == 1:
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=replan_strategy("new plan"))
    ag = Agent(agent_fn, policy)
    await ag.run("task")

    ctx: TriageContext = received[1]
    assert ctx.hint == "new plan"
    assert ctx.subgoal is None


async def test_triage_context_state_on_rollback():
    store = InMemoryCheckpointStore()
    received_ctx: list[Any] = []
    call_count = [0]

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_count[0] += 1
        received_ctx.append(kw.get("_triage_context"))
        if call_count[0] == 1:
            record_step(make_step(0))
            update_state({"key": "value"})
            raise RuntimeError("fail")
        return "ok"

    policy = FailurePolicy(UNKNOWN=rollback_strategy())
    ag = Agent(agent_fn, policy, checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    ctx: TriageContext = received_ctx[1]
    assert ctx.state == {"key": "value"}


# ---------------------------------------------------------------------------
# get_recorder() / get_state_updater() — contextvars injection
# ---------------------------------------------------------------------------

async def test_get_recorder_works_inside_run():
    recorded: list[Step] = []

    async def agent_fn(task: str, **kw: Any) -> str:
        # Does NOT accept record_step — uses contextvars instead
        record = get_recorder()
        s = make_step(0)
        record(s)
        recorded.append(s)
        return "ok"

    ag = Agent(agent_fn, FailurePolicy())
    await ag.run("task")
    assert len(recorded) == 1


async def test_get_state_updater_works_inside_run():
    store = InMemoryCheckpointStore()

    async def agent_fn(task: str, **kw: Any) -> str:
        upd = get_state_updater()
        rec = get_recorder()
        rec(make_step(0))
        upd({"from_contextvar": True})
        return "ok"

    ag = Agent(agent_fn, FailurePolicy(), checkpoint_store=store, auto_checkpoint=True)
    await ag.run("task")

    cp = await store.latest()
    assert cp is not None
    assert cp.state == {"from_contextvar": True}


async def test_get_recorder_raises_outside_run():
    """get_recorder() must raise when called outside Agent.run()."""
    with pytest.raises(RuntimeError, match="outside a triage"):
        get_recorder()


async def test_get_state_updater_raises_outside_run():
    with pytest.raises(RuntimeError, match="outside a triage"):
        get_state_updater()


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


async def test_cancelled_error_propagates_without_recovery():
    """CancelledError must never be caught and classified as a failure."""
    import asyncio

    calls = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        calls[0] += 1
        raise asyncio.CancelledError("task cancelled")

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, max_recovery_attempts=3)

    with pytest.raises(asyncio.CancelledError):
        await ag.run("task")

    # Must have run exactly once — no recovery attempts
    assert calls[0] == 1


# ── Lifecycle hooks ───────────────────────────────────────────────────────────

async def test_on_step_called_for_each_recorded_step():
    received: list[Step] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="a"))
        record_step(Step(index=1, action="b"))
        record_step(Step(index=2, action="c"))
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, on_step=received.append)
    await ag.run("task")

    assert len(received) == 3
    assert [s.action for s in received] == ["a", "b", "c"]


async def test_on_failure_called_with_failure_context():
    received: list[Any] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        if not kw.get("_triage_context"):
            raise RuntimeError("fail")
        return "ok"

    async def retry(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(UNKNOWN=retry)
    ag = Agent(agent_fn, policy, on_failure=received.append)
    await ag.run("task")

    assert len(received) == 1
    assert received[0].failure_type == FailureType.UNKNOWN


async def test_on_recovery_called_with_ctx_and_action():
    ctx_received: list[Any] = []
    action_received: list[Any] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        if not kw.get("_triage_context"):
            raise RuntimeError("fail")
        return "ok"

    async def retry(ctx: Any) -> Any:
        return RecoveryAction.RETRY(hint="go again")

    policy = FailurePolicy(UNKNOWN=retry)

    def hook(ctx: Any, action: Any) -> None:
        ctx_received.append(ctx)
        action_received.append(action)

    ag = Agent(agent_fn, policy, on_recovery=hook)
    await ag.run("task")

    assert len(action_received) == 1
    assert action_received[0].kind == "retry"
    assert ctx_received[0].failure_type == FailureType.UNKNOWN


async def test_hooks_not_called_on_clean_success():
    failure_calls: list[Any] = []
    recovery_calls: list[Any] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy,
               on_failure=failure_calls.append,
               on_recovery=recovery_calls.append)
    await ag.run("task")

    assert failure_calls == []
    assert recovery_calls == []


async def test_hook_exception_does_not_break_run():
    def exploding_hook(step: Step) -> None:
        raise ValueError("hook error")

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="work"))
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy, on_step=exploding_hook)
    result = await ag.run("task")
    assert result == "ok"


async def test_clone_copies_hooks():
    received: list[Step] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="x"))
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    original = Agent(agent_fn, policy, on_step=received.append)
    cloned = original.clone()

    await cloned.run("task")
    assert len(received) == 1  # hook fired on cloned agent


# ── Step.idempotent is informational, not enforced by agent.py ────────────────

async def test_retry_does_not_check_idempotency_automatically():
    """Agents that mark steps non-idempotent still get retried.
    The idempotent flag is informational — strategies must check it explicitly.
    """
    calls: list[int] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        calls.append(1)
        record_step(Step(index=0, action="charge_card", idempotent=False,
                         error="tool foo not found"))
        if len(calls) == 1:
            raise RuntimeError("tool foo not found")
        return "ok"

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy)
    result = await ag.run("task")
    assert result == "ok"
    assert len(calls) == 2  # retried despite idempotent=False


# ── strict_idempotency ────────────────────────────────────────────────────────

async def test_strict_idempotency_escalates_on_non_idempotent_step():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="charge_card", idempotent=False,
                         error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, strict_idempotency=True)
    with pytest.raises(TriageEscalationError):
        await ag.run("task")


async def test_strict_idempotency_message_lists_non_idempotent_steps():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="send_email", idempotent=False,
                         error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, strict_idempotency=True)
    with pytest.raises(TriageEscalationError, match="send_email"):
        await ag.run("task")


async def test_strict_idempotency_false_allows_retry():
    calls: list[int] = []

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        calls.append(1)
        record_step(Step(index=0, action="charge_card", idempotent=False,
                         error="tool foo not found"))
        if len(calls) == 1:
            raise RuntimeError("tool foo not found")
        return "ok"

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, strict_idempotency=False)
    result = await ag.run("task")
    assert result == "ok"
    assert len(calls) == 2


async def test_strict_idempotency_only_checks_on_retry():
    """Non-retry actions (replan) are not blocked by strict_idempotency."""
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="charge_card", idempotent=False))
        raise RuntimeError("plan failed")

    async def replan_strategy(ctx: Any) -> Any:
        return RecoveryAction.REPLAN(hint="try again")

    calls: list[str] = []

    async def agent_fn2(task: str, *, record_step: Any, **kw: Any) -> str:
        calls.append("called")
        if len(calls) == 1:
            record_step(Step(index=0, action="charge_card", idempotent=False))
            raise RuntimeError("plan failed")
        return "ok"

    policy = FailurePolicy(default=replan_strategy)
    ag = Agent(agent_fn2, policy, strict_idempotency=True)
    result = await ag.run("task")
    assert result == "ok"


async def test_clone_copies_strict_idempotency():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    original = Agent(agent_fn, policy, strict_idempotency=True)
    cloned = original.clone()
    assert cloned._strict_idempotency is True


# ── max_recovery_seconds ──────────────────────────────────────────────────────

async def test_max_recovery_seconds_escalates_when_exceeded(monkeypatch: Any):
    import time as _time
    calls: list[float] = [0.0]

    def fake_monotonic() -> float:
        # First call returns 0, second returns 100 (exceeds any cap)
        val = calls[0]
        calls[0] += 100.0
        return val

    monkeypatch.setattr(_time, "monotonic", fake_monotonic)

    attempt_count: list[int] = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        attempt_count[0] += 1
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, max_recovery_seconds=10.0, max_recovery_attempts=10)
    with pytest.raises(TriageEscalationError, match="max_recovery_seconds"):
        await ag.run("task")


async def test_max_recovery_seconds_none_does_not_cap():
    """Default (None) never triggers time-based escalation."""
    calls: list[int] = [0]

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        calls[0] += 1
        record_step(Step(index=0, action="step", error="tool foo not found"))
        if calls[0] < 3:
            raise RuntimeError("tool foo not found")
        return "ok"

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, max_recovery_seconds=None)
    result = await ag.run("task")
    assert result == "ok"


async def test_clone_copies_max_recovery_seconds():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    original = Agent(agent_fn, policy, max_recovery_seconds=30.0)
    cloned = original.clone()
    assert cloned._max_recovery_seconds == 30.0


# ── structured logs ───────────────────────────────────────────────────────────

async def test_structured_log_failure_classified(caplog: Any):
    import logging

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, max_recovery_attempts=1)

    with caplog.at_level(logging.INFO, logger="triage"):
        with pytest.raises(TriageEscalationError):
            await ag.run("task")

    events = [r for r in caplog.records if getattr(r, "triage_event", None) == "failure_classified"]
    assert events, "Expected at least one failure_classified log record"
    assert events[0].__dict__["failure_type"] == "wrong_tool_called"


async def test_structured_log_action_dispatched(caplog: Any):
    import logging

    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy, max_recovery_attempts=1)

    with caplog.at_level(logging.INFO, logger="triage"):
        with pytest.raises(TriageEscalationError):
            await ag.run("task")

    events = [r for r in caplog.records if getattr(r, "triage_event", None) == "action_dispatched"]
    assert events, "Expected at least one action_dispatched log record"
    assert events[0].__dict__["action_kind"] == "retry"


# ── multi-agent context propagation ──────────────────────────────────────────

async def test_child_triage_escalation_propagates_to_parent():
    """Child TriageEscalationError propagates unchanged through the parent agent.

    The parent's except (TriageEscalationError, TriageAbortError) re-raise clause
    ensures the child's exception is never re-classified — it surfaces directly to
    the caller of the outer agent's run().
    """
    inner_policy = FailurePolicy(default=FailurePolicy.escalate_by_default())

    async def inner_agent(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    inner = Agent(inner_agent, inner_policy)

    async def outer_agent(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="outer step"))
        await inner.run(task)  # inner escalates; propagates out unchanged
        return "ok"

    outer_policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    outer = Agent(outer_agent, outer_policy)

    # The child's TriageEscalationError propagates through the outer agent
    # unchanged — the outer's re-raise clause handles it without re-classifying.
    with pytest.raises(TriageEscalationError) as exc_info:
        await outer.run("task")

    # Context comes from the inner agent's classification
    assert exc_info.value.context.failure_type == FailureType.WRONG_TOOL_CALLED


async def test_child_triage_context_reused_when_chained_exception():
    """If exception chains to TriageEscalationError, outer reuses child's failure_type."""
    inner_policy = FailurePolicy(default=FailurePolicy.escalate_by_default())

    async def inner_agent(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    inner = Agent(inner_agent, inner_policy)

    outer_received_types: list[str] = []

    async def outer_agent(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(Step(index=0, action="outer step"))
        try:
            await inner.run(task)
        except TriageEscalationError as e:
            # Wrap and re-raise — outer should reuse child's failure_type
            raise RuntimeError("outer wrapped child failure") from e
        return "ok"

    async def outer_recovery(ctx: Any) -> Any:
        outer_received_types.append(ctx.failure_type.value)
        return RecoveryAction.ABORT(reason="handled")

    outer_policy = FailurePolicy(default=outer_recovery)
    outer = Agent(outer_agent, outer_policy)

    with pytest.raises(TriageAbortError):
        await outer.run("task")

    # Outer should have reused the child's wrong_tool_called classification
    assert outer_received_types == ["wrong_tool_called"]


# ── report_misclassification ──────────────────────────────────────────────────

async def test_report_misclassification_raises_when_no_context():
    async def agent_fn(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    ag = Agent(agent_fn, policy)
    with pytest.raises(RuntimeError, match="No failure context"):
        ag.report_misclassification(FailureType.EXTERNAL_FAULT)


# ── step risk scoring ─────────────────────────────────────────────────────────

async def test_risk_scorer_aborts_on_high_score():
    from triage.scorer.base import RiskScore

    def always_risky(step, trajectory):
        return RiskScore(score=1.0, reason="always abort")

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="dangerous action"))
        return "should not reach"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=always_risky, risk_threshold=0.9)
    with pytest.raises(TriageAbortError) as exc_info:
        await ag.run("task")
    assert "always abort" in str(exc_info.value)


async def test_risk_scorer_does_not_abort_below_threshold():
    from triage.scorer.base import RiskScore

    def low_risk(step, trajectory):
        return RiskScore(score=0.5)

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="safe step"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=low_risk, risk_threshold=0.9)
    result = await ag.run("task")
    assert result == "ok"


async def test_risk_scorer_none_does_not_affect_run():
    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="step"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy)  # no risk_scorer
    result = await ag.run("task")
    assert result == "ok"


async def test_risk_scorer_receives_trajectory():
    received = []

    def capture_scorer(step, trajectory):
        from triage.scorer.base import RiskScore
        received.append(len(trajectory.steps))
        return RiskScore(score=0.0)

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="step 1"))
        record_step(Step(index=1, action="step 2"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=capture_scorer)
    await ag.run("task")
    assert received == [1, 2]  # trajectory grows with each step


def test_clone_copies_risk_scorer_and_threshold():
    from triage.scorer.base import RiskScore

    def scorer(step, trajectory):
        return RiskScore(score=0.0)

    policy = FailurePolicy()
    ag = Agent(lambda t, **kw: None, policy, risk_scorer=scorer, risk_threshold=0.8)
    cloned = ag.clone()
    assert cloned._risk_scorer is scorer
    assert cloned._risk_threshold == 0.8


async def test_risk_scorer_exception_is_sandboxed():
    """A buggy scorer that raises must not crash the run — skip score check and continue."""
    def buggy_scorer(step, trajectory):
        raise RuntimeError("scorer bug")

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="safe step"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=buggy_scorer)
    result = await ag.run("task")
    assert result == "ok"


# ---------------------------------------------------------------------------
# Concurrent run() calls on a single Agent instance
# ---------------------------------------------------------------------------

async def test_concurrent_runs_do_not_share_trajectory():
    """Two run() calls in flight at once on the same Agent must not see each
    other's steps — trajectory is isolated per concurrent task, not per Agent.
    """
    import anyio

    results: dict[str, list[str]] = {}

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        record_step(Step(index=0, action=f"start-{task}"))
        await anyio.sleep(0.02 if task == "slow" else 0.0)
        record_step(Step(index=1, action=f"end-{task}"))
        return task

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy)

    async def run_and_capture(task: str) -> None:
        await ag.run(task)
        results[task] = [s.action for s in ag._trajectory.steps]

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_and_capture, "slow")
        tg.start_soon(run_and_capture, "fast")

    assert results["slow"] == ["start-slow", "end-slow"]
    assert results["fast"] == ["start-fast", "end-fast"]


async def test_concurrent_runs_do_not_share_current_state():
    import anyio

    captured: dict[str, dict] = {}

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        update_state({"task": task})
        await anyio.sleep(0.02 if task == "slow" else 0.0)
        return task

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy)

    async def run_and_capture(task: str) -> None:
        await ag.run(task)
        captured[task] = dict(ag._current_state)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_and_capture, "slow")
        tg.start_soon(run_and_capture, "fast")

    assert captured["slow"] == {"task": "slow"}
    assert captured["fast"] == {"task": "fast"}


async def test_concurrent_runs_recover_independently():
    """A failure + recovery in one concurrent run() must not affect another's
    attempt_history or trajectory.
    """
    import anyio

    call_counts = {"slow": 0, "fast": 0}

    async def agent_fn(task: str, *, record_step: Any, update_state: Any, **kw: Any) -> str:
        call_counts[task] += 1
        record_step(Step(index=0, action="step", error="tool foo not found"))
        if task == "slow":
            await anyio.sleep(0.02)
        if call_counts[task] == 1:
            raise RuntimeError("tool foo not found")
        return task

    async def retry_strategy(ctx: Any) -> Any:
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    ag = Agent(agent_fn, policy)

    results: dict[str, str] = {}

    async def run_and_capture(task: str) -> None:
        results[task] = await ag.run(task)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_and_capture, "slow")
        tg.start_soon(run_and_capture, "fast")

    assert results == {"slow": "slow", "fast": "fast"}
    assert call_counts == {"slow": 2, "fast": 2}
