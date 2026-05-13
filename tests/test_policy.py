"""Tests for triage.policy — RecoveryAction, FailurePolicy."""

import pytest

from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureContext, FailureType, Step


def make_ctx(failure_type: FailureType = FailureType.UNKNOWN) -> FailureContext:
    return FailureContext(
        failure_type=failure_type,
        trajectory=[Step(index=0, action="step")],
        critical_step_index=0,
        original_task="test task",
    )


# ── RecoveryAction constructors ───────────────────────────────────────────────

def test_recovery_action_retry():
    action = RecoveryAction.RETRY(hint="try again")
    assert action.kind == "retry"
    assert action.params["hint"] == "try again"


def test_recovery_action_retry_with_delay():
    action = RecoveryAction.RETRY(delay=4.0)
    assert action.params["delay"] == 4.0


def test_recovery_action_replan():
    action = RecoveryAction.REPLAN(hint="rethink it")
    assert action.kind == "replan"
    assert action.params["hint"] == "rethink it"


def test_recovery_action_rollback():
    action = RecoveryAction.ROLLBACK(checkpoint_id="cp-1")
    assert action.kind == "rollback"
    assert action.params["checkpoint_id"] == "cp-1"


def test_recovery_action_resume():
    action = RecoveryAction.RESUME(from_subgoal="step-2")
    assert action.kind == "resume"
    assert action.params["from_subgoal"] == "step-2"


def test_recovery_action_escalate():
    action = RecoveryAction.ESCALATE(message="help!")
    assert action.kind == "escalate"
    assert action.params["message"] == "help!"


def test_recovery_action_abort():
    action = RecoveryAction.ABORT(reason="fatal")
    assert action.kind == "abort"
    assert action.params["reason"] == "fatal"


def test_recovery_action_none_params_excluded():
    action = RecoveryAction.RETRY()
    assert "hint" not in action.params
    assert "inject" not in action.params


def test_recovery_action_repr():
    action = RecoveryAction.ESCALATE(message="oops")
    assert "ESCALATE" in repr(action)
    assert "oops" in repr(action)


# ── FailurePolicy.resolve ─────────────────────────────────────────────────────

def test_resolve_exact_match():
    async def my_strategy(ctx):
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=my_strategy)
    assert policy.resolve(FailureType.WRONG_TOOL_CALLED) is my_strategy


def test_resolve_fallback_to_default():
    async def default_strategy(ctx):
        return RecoveryAction.ESCALATE()

    policy = FailurePolicy(default=default_strategy)
    assert policy.resolve(FailureType.LOOP_DETECTED) is default_strategy


def test_resolve_no_match_no_default_returns_none():
    policy = FailurePolicy()
    assert policy.resolve(FailureType.SCHEMA_MISMATCH) is None


def test_resolve_specific_beats_default():
    async def specific(ctx):
        return RecoveryAction.RETRY()

    async def default_fn(ctx):
        return RecoveryAction.ESCALATE()

    policy = FailurePolicy(EXTERNAL_FAULT=specific, default=default_fn)
    assert policy.resolve(FailureType.EXTERNAL_FAULT) is specific
    assert policy.resolve(FailureType.UNKNOWN) is default_fn


# ── FailurePolicy.dispatch ────────────────────────────────────────────────────

async def test_dispatch_calls_strategy_and_returns_action():
    async def my_strategy(ctx):
        return RecoveryAction.REPLAN(hint="redo")

    policy = FailurePolicy(LOOP_DETECTED=my_strategy)
    ctx = make_ctx(FailureType.LOOP_DETECTED)
    action = await policy.dispatch(ctx)
    assert action.kind == "replan"
    assert action.params["hint"] == "redo"


async def test_dispatch_no_strategy_escalates():
    policy = FailurePolicy()
    ctx = make_ctx(FailureType.HALLUCINATED_STATE)
    action = await policy.dispatch(ctx)
    assert action.kind == "escalate"
    assert "hallucinated_state" in action.params.get("message", "")


async def test_dispatch_uses_default_when_no_specific():
    async def fallback(ctx):
        return RecoveryAction.ABORT(reason="unhandled")

    policy = FailurePolicy(default=fallback)
    ctx = make_ctx(FailureType.GOAL_DRIFT)
    action = await policy.dispatch(ctx)
    assert action.kind == "abort"


# ── convenience factories ─────────────────────────────────────────────────────

async def test_escalate_by_default_factory():
    strategy = FailurePolicy.escalate_by_default()
    ctx = make_ctx(FailureType.CONTEXT_OVERFLOW)
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "context_overflow" in action.params.get("message", "")


async def test_abort_by_default_factory():
    strategy = FailurePolicy.abort_by_default()
    ctx = make_ctx(FailureType.PLAN_INCOMPLETE)
    action = await strategy(ctx)
    assert action.kind == "abort"
    assert action.params.get("reason") == "plan_incomplete"


# ── FailurePolicy.chain ───────────────────────────────────────────────────────

async def test_chain_uses_primary_when_not_escalating():
    async def primary(ctx):
        return RecoveryAction.RETRY(hint="retry")

    async def fallback(ctx):
        return RecoveryAction.ABORT(reason="fallback hit")

    chained = FailurePolicy.chain(primary, fallback)
    ctx = make_ctx(FailureType.EXTERNAL_FAULT)
    action = await chained(ctx)
    assert action.kind == "retry"


async def test_chain_falls_through_to_fallback_on_escalate():
    async def primary(ctx):
        return RecoveryAction.ESCALATE(message="giving up")

    async def fallback(ctx):
        return RecoveryAction.REPLAN(hint="fallback replan")

    chained = FailurePolicy.chain(primary, fallback)
    ctx = make_ctx(FailureType.LOOP_DETECTED)
    action = await chained(ctx)
    assert action.kind == "replan"
    assert action.params["hint"] == "fallback replan"


async def test_chain_custom_after_kinds():
    async def primary(ctx):
        return RecoveryAction.RETRY()

    async def fallback(ctx):
        return RecoveryAction.ABORT(reason="custom fallback")

    # Fall through on "retry" (non-standard usage for test)
    chained = FailurePolicy.chain(primary, fallback, after_kinds=("retry",))
    ctx = make_ctx(FailureType.UNKNOWN)
    action = await chained(ctx)
    assert action.kind == "abort"


async def test_chain_does_not_call_fallback_on_replan():
    calls = []

    async def primary(ctx):
        return RecoveryAction.REPLAN()

    async def fallback(ctx):
        calls.append("fallback")
        return RecoveryAction.ABORT(reason="should not reach")

    chained = FailurePolicy.chain(primary, fallback)
    ctx = make_ctx(FailureType.GOAL_DRIFT)
    action = await chained(ctx)
    assert action.kind == "replan"
    assert calls == []


# ── TIMEOUT field ─────────────────────────────────────────────────────────────

async def test_timeout_field_resolves_strategy():
    async def strategy(ctx):
        return RecoveryAction.RETRY()

    policy = FailurePolicy(TIMEOUT=strategy)
    ctx = make_ctx(FailureType.TIMEOUT)
    action = await policy.dispatch(ctx)
    assert action.kind == "retry"


def test_timeout_falls_through_to_default():
    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    assert policy.resolve(FailureType.TIMEOUT) is not None  # default is set
