"""
tests/test_suspension.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for human-in-the-loop pause/resume:
  - SuspendedRun / SuspensionStore protocol
  - InMemorySuspensionStore
  - RecoveryAction.SUSPEND()
  - TriageSuspendedError
  - Agent.resume(token, action=...)
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from triage.agent import Agent, TriageAbortError, TriageEscalationError, TriageSuspendedError
from triage.policy import FailurePolicy, RecoveryAction
from triage.suspension import (
    InMemorySuspensionStore,
    SuspendedRun,
    SuspensionStore,
    deserialize_run,
    serialize_run,
)
from triage.taxonomy import FailureContext, FailureType, Step
from triage.usage import Usage


def make_step(
    index: int = 0,
    error: str | None = None,
    tool_called: str | None = None,
) -> Step:
    return Step(index=index, action="test step", error=error, tool_called=tool_called)


# ── InMemorySuspensionStore ───────────────────────────────────────────────────


async def test_store_save_and_load():
    store = InMemorySuspensionStore()
    run = SuspendedRun(
        token="tok-1",
        context=None,  # type: ignore[arg-type]
        task="t",
        kwargs={},
        attempt=0,
        attempt_history=[],
    )
    await store.save(run)
    loaded = await store.load("tok-1")
    assert loaded.token == "tok-1"


async def test_store_load_missing_raises():
    store = InMemorySuspensionStore()
    with pytest.raises(KeyError, match="no-such-token"):
        await store.load("no-such-token")


async def test_store_delete_removes_entry():
    store = InMemorySuspensionStore()
    run = SuspendedRun(
        token="tok-del",
        context=None,  # type: ignore[arg-type]
        task="t",
        kwargs={},
        attempt=0,
        attempt_history=[],
    )
    await store.save(run)
    await store.delete("tok-del")
    with pytest.raises(KeyError):
        await store.load("tok-del")


async def test_store_delete_missing_is_noop():
    store = InMemorySuspensionStore()
    await store.delete("never-existed")  # must not raise


def test_suspension_store_protocol():
    """InMemorySuspensionStore satisfies the SuspensionStore Protocol."""
    assert isinstance(InMemorySuspensionStore(), SuspensionStore)


# ── RecoveryAction.SUSPEND ────────────────────────────────────────────────────


def test_suspend_action_kind():
    a = RecoveryAction.SUSPEND()
    assert a.kind == "suspend"


def test_suspend_action_message():
    a = RecoveryAction.SUSPEND(message="needs approval")
    assert a.params["message"] == "needs approval"


def test_suspend_action_metadata():
    a = RecoveryAction.SUSPEND(metadata={"channel": "#ops"})
    assert a.params["metadata"] == {"channel": "#ops"}


def test_suspend_action_no_message_is_excluded():
    a = RecoveryAction.SUSPEND()
    assert "message" not in a.params


# ── TriageSuspendedError ──────────────────────────────────────────────────────


async def test_suspend_raises_triage_suspended_error():
    """Policy returning SUSPEND raises TriageSuspendedError with a token."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="needs human"))
        raise RuntimeError("needs human")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND(message="approve this")

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    err = exc_info.value
    assert err.token
    assert err.run.task == "task"
    assert err.run.message == "approve this"


async def test_suspend_stores_run_in_store():
    """The SuspendedRun is persisted before TriageSuspendedError is raised."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="down"))
        raise RuntimeError("down")

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=lambda ctx: RecoveryAction.SUSPEND())  # type: ignore[arg-type, return-value]

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    token = exc_info.value.token
    loaded = await store.load(token)
    assert loaded.token == token
    assert loaded.task == "task"


async def test_suspend_carries_failure_context():
    """SuspendedRun.context has the correct failure type."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="HTTP 503"))
        raise RuntimeError("HTTP 503")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    store = InMemorySuspensionStore()
    policy = FailurePolicy(EXTERNAL_FAULT=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    assert exc_info.value.run.context.failure_type == FailureType.EXTERNAL_FAULT


async def test_suspend_metadata_stored():
    """Metadata from SUSPEND() is preserved in the SuspendedRun."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="boom"))
        raise RuntimeError("boom")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND(metadata={"channel": "#ops", "user": "alice"})

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    assert exc_info.value.run.metadata == {"channel": "#ops", "user": "alice"}


# ── Agent.resume ──────────────────────────────────────────────────────────────


async def test_resume_with_retry_completes_run():
    """Human approves RETRY → agent continues and succeeds."""
    calls: list[int] = []

    async def flaky(task: str, *, record_step: Any, **kw: Any) -> str:
        calls.append(len(calls))
        record_step(make_step(0, error="boom" if len(calls) == 1 else None))
        if len(calls) == 1:
            raise RuntimeError("first call fails")
        return "done"

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND(message="please decide")

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(flaky, policy, suspension_store=store, max_recovery_attempts=3)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    token = exc_info.value.token
    result = await ag.resume(token, action=RecoveryAction.RETRY())
    assert result == "done"
    assert len(calls) == 2


async def test_resume_with_abort_raises_triage_abort():
    """Human chooses ABORT → TriageAbortError is raised."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="boom"))
        raise RuntimeError("boom")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    with pytest.raises(TriageAbortError, match="human rejected"):
        await ag.resume(
            exc_info.value.token,
            action=RecoveryAction.ABORT(reason="human rejected"),
        )


async def test_resume_deletes_token_after_load():
    """Token is single-use — second resume raises KeyError."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="boom"))
        raise RuntimeError("boom")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    token = exc_info.value.token

    # First resume — consumes the token
    with pytest.raises(TriageAbortError):
        await ag.resume(token, action=RecoveryAction.ABORT(reason="done"))

    # Second resume — token is gone
    with pytest.raises(KeyError):
        await ag.resume(token, action=RecoveryAction.RETRY())


async def test_resume_invalid_token_raises_key_error():
    async def ok_agent(task: str, *, record_step: Any, **kw: Any) -> str:
        return "ok"

    store = InMemorySuspensionStore()
    ag = Agent(ok_agent, FailurePolicy(), suspension_store=store)

    with pytest.raises(KeyError):
        await ag.resume("no-such-token", action=RecoveryAction.RETRY())


async def test_resume_with_replan_hint():
    """Human supplies a REPLAN hint → agent receives it as _triage_hint."""
    hints_received: list[str] = []
    calls: list[int] = []

    async def flaky(task: str, *, record_step: Any, **kw: Any) -> str:
        calls.append(len(calls))
        if len(calls) == 1:
            record_step(make_step(0, error="lost"))
            raise RuntimeError("lost context")
        hints_received.append(kw.get("_triage_hint", ""))
        record_step(make_step(0))
        return "recovered"

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    ag = Agent(flaky, policy, suspension_store=store, max_recovery_attempts=3)

    with pytest.raises(TriageSuspendedError) as exc_info:
        await ag.run("task")

    result = await ag.resume(
        exc_info.value.token,
        action=RecoveryAction.REPLAN(hint="start over with fresh context"),
    )
    assert result == "recovered"
    assert hints_received == ["start over with fresh context"]


async def test_suspend_does_not_fire_for_normal_escalate():
    """ESCALATE (not SUSPEND) raises TriageEscalationError, not TriageSuspendedError."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="boom"))
        raise RuntimeError("boom")

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    store = InMemorySuspensionStore()
    ag = Agent(always_fails, policy, suspension_store=store)

    with pytest.raises(TriageEscalationError):
        await ag.run("task")


async def test_suspension_store_copied_by_clone():
    """clone() uses the same suspension_store so tokens work across clones."""

    async def always_fails(task: str, *, record_step: Any, **kw: Any) -> str:
        record_step(make_step(0, error="boom"))
        raise RuntimeError("boom")

    async def suspend_strategy(ctx: Any) -> RecoveryAction:
        return RecoveryAction.SUSPEND()

    store = InMemorySuspensionStore()
    policy = FailurePolicy(default=suspend_strategy)
    original = Agent(always_fails, policy, suspension_store=store)
    cloned = original.clone()

    # Suspend via original
    with pytest.raises(TriageSuspendedError) as exc_info:
        await original.run("task")

    token = exc_info.value.token

    # Resume via clone — same store, so the token is accessible
    with pytest.raises(TriageAbortError):
        await cloned.resume(token, action=RecoveryAction.ABORT(reason="done"))


# ── serialize_run / deserialize_run round-trip ────────────────────────────────


def _make_suspended_run(*, kwargs: dict[str, Any] | None = None) -> SuspendedRun:
    # Sets every non-default field so the round-trip test exercises the full surface.
    step = Step(
        index=0,
        action="test step",
        tool_called="search",
        tool_input={"query": "hello"},
        tool_output="some result",
        llm_output="thinking...",
        error=None,
        timestamp=1.0,
        state_hash="abc123",
        metadata={"source": "unit-test"},
        idempotent=True,
        partial=False,
    )
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[step],
        critical_step_index=0,
        original_task="do the thing",
        last_checkpoint_id="ckpt-1",
        loop_steps=[0, 1, 2],
        violated_constraint="no markdown",
        expected_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        metadata={"attempt_number": 1, "run_id": "run-abc"},
        attempt_history=[(FailureType.TIMEOUT, "retry")],
    )
    return SuspendedRun(
        token="tok-rt",
        context=ctx,
        task="do the thing",
        kwargs=kwargs or {"user": "alice", "dry_run": True},
        attempt=1,
        attempt_history=[(FailureType.TIMEOUT, "retry")],
        timestamp=1000.0,
        message="needs approval",
        metadata={"channel": "#ops"},
        usage_snapshot=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, calls=1),
    )


def test_serialize_deserialize_round_trip():
    """All scalar and nested fields survive a JSON round-trip without loss."""
    run = _make_suspended_run()
    restored = deserialize_run(serialize_run(run))

    assert restored.token == run.token
    assert restored.task == run.task
    assert restored.attempt == run.attempt
    assert restored.timestamp == run.timestamp
    assert restored.message == run.message
    assert restored.metadata == run.metadata
    assert restored.kwargs == run.kwargs
    assert restored.attempt_history == run.attempt_history

    # Usage snapshot
    assert restored.usage_snapshot.input_tokens == run.usage_snapshot.input_tokens
    assert restored.usage_snapshot.output_tokens == run.usage_snapshot.output_tokens
    assert restored.usage_snapshot.cost_usd == run.usage_snapshot.cost_usd
    assert restored.usage_snapshot.calls == run.usage_snapshot.calls

    # FailureContext
    ctx = restored.context
    assert ctx.failure_type == FailureType.EXTERNAL_FAULT
    assert ctx.original_task == "do the thing"
    assert ctx.critical_step_index == 0
    assert ctx.last_checkpoint_id == "ckpt-1"
    assert ctx.loop_steps == [0, 1, 2]
    assert ctx.violated_constraint == "no markdown"
    assert ctx.expected_schema == {"type": "object", "properties": {"result": {"type": "string"}}}
    assert ctx.metadata == {"attempt_number": 1, "run_id": "run-abc"}
    assert ctx.attempt_history == [(FailureType.TIMEOUT, "retry")]

    # Trajectory
    assert len(ctx.trajectory) == 1
    s = ctx.trajectory[0]
    assert s.index == 0
    assert s.action == "test step"
    assert s.tool_called == "search"
    assert s.tool_input == {"query": "hello"}
    assert s.tool_output == "some result"
    assert s.llm_output == "thinking..."
    assert s.error is None
    assert s.timestamp == 1.0
    assert s.state_hash == "abc123"
    assert s.metadata == {"source": "unit-test"}
    assert s.idempotent is True
    assert s.partial is False


def test_serialize_non_primitive_tool_output_coerced_to_str():
    """tool_output values that are not JSON primitives are coerced to str()."""
    step = Step(index=0, action="step", tool_output=object())
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=[step],
        critical_step_index=0,
        original_task="task",
    )
    run = SuspendedRun(
        token="tok-coerce",
        context=ctx,
        task="task",
        kwargs={},
        attempt=0,
        attempt_history=[],
    )
    restored = deserialize_run(serialize_run(run))
    # After round-trip the value is a string (str(object()) form), not the original object.
    assert isinstance(restored.context.trajectory[0].tool_output, str)


def test_serialize_non_serializable_kwargs_are_dropped_with_warning(caplog: Any):
    """kwargs with non-primitive values are dropped and a warning is logged."""
    import logging

    run = _make_suspended_run(kwargs={"user": "alice", "client": object(), "timeout": 30})
    with caplog.at_level(logging.WARNING, logger="triage"):
        restored = deserialize_run(serialize_run(run))

    # Non-serializable "client" key is absent; primitives survive
    assert "client" not in restored.kwargs
    assert restored.kwargs == {"user": "alice", "timeout": 30}

    # Warning was emitted
    warnings = [
        r for r in caplog.records if getattr(r, "triage_event", None) == "serialize_kwargs_dropped"
    ]
    assert warnings, "Expected a serialize_kwargs_dropped warning"
    assert "client" in warnings[0].dropped_keys


def test_serialize_all_primitive_kwargs_no_warning(caplog: Any):
    """No warning is emitted when all kwargs are JSON-serializable."""
    import logging

    run = _make_suspended_run(kwargs={"user": "alice", "n": 3, "flag": True, "x": None})
    with caplog.at_level(logging.WARNING, logger="triage"):
        restored = deserialize_run(serialize_run(run))

    assert restored.kwargs == {"user": "alice", "n": 3, "flag": True, "x": None}
    warnings = [
        r for r in caplog.records if getattr(r, "triage_event", None) == "serialize_kwargs_dropped"
    ]
    assert not warnings


# ── field-drift guard tests ───────────────────────────────────────────────────
# These tests introspect dataclasses.fields() so that adding a new field to
# Step or FailureContext immediately fails CI if serialization isn't updated.
# "Intentionally omitted" sets document deliberate exclusions so the guard
# never needs to be weakened to pass.

_STEP_INTENTIONALLY_OMITTED: set[str] = set()
_CTX_INTENTIONALLY_OMITTED: set[str] = {"raw_error"}


def test_serialize_covers_all_step_fields():
    """Every Step field (except intentional omissions) appears in the serialized dict."""
    run = _make_suspended_run()
    doc = json.loads(serialize_run(run))
    step_doc = doc["context"]["trajectory"][0]
    for f in dataclasses.fields(Step):
        if f.name in _STEP_INTENTIONALLY_OMITTED:
            continue
        assert f.name in step_doc, f"Step.{f.name} is missing from serialize_run output"


def test_serialize_covers_all_context_fields():
    """Every FailureContext field (except raw_error) appears in the serialized dict."""
    run = _make_suspended_run()
    doc = json.loads(serialize_run(run))
    ctx_doc = doc["context"]
    for f in dataclasses.fields(FailureContext):
        if f.name in _CTX_INTENTIONALLY_OMITTED:
            continue
        assert f.name in ctx_doc, f"FailureContext.{f.name} is missing from serialize_run output"
