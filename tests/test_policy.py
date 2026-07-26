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
    ctx = make_ctx(FailureType.SCHEMA_MISMATCH)
    action = await policy.dispatch(ctx)
    assert action.kind == "escalate"
    assert "schema_mismatch" in action.params.get("message", "")


async def test_dispatch_uses_default_when_no_specific():
    async def fallback(ctx):
        return RecoveryAction.ABORT(reason="unhandled")

    policy = FailurePolicy(default=fallback)
    ctx = make_ctx(FailureType.CONTEXT_OVERFLOW)
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
    ctx = make_ctx(FailureType.UNKNOWN)
    action = await chained(ctx)
    assert action.kind == "replan"
    assert calls == []


# ── FailurePolicy.sequence ───────────────────────────────────────────────────

async def test_sequence_first_strategy_on_first_failure():
    async def s1(ctx): return RecoveryAction.RETRY(hint="s1")
    async def s2(ctx): return RecoveryAction.REPLAN(hint="s2")

    seq = FailurePolicy.sequence(s1, s2)
    ctx = make_ctx(FailureType.UNKNOWN)  # no prior attempts
    action = await seq(ctx)
    assert action.kind == "retry"
    assert action.params["hint"] == "s1"


async def test_sequence_second_strategy_on_second_failure():
    async def s1(ctx): return RecoveryAction.RETRY(hint="s1")
    async def s2(ctx): return RecoveryAction.REPLAN(hint="s2")

    seq = FailurePolicy.sequence(s1, s2)
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=[Step(index=0, action="step")],
        critical_step_index=0,
        original_task="task",
        attempt_history=[(FailureType.UNKNOWN, "retry")],  # one prior attempt
    )
    action = await seq(ctx)
    assert action.kind == "replan"
    assert action.params["hint"] == "s2"


async def test_sequence_escalates_after_exhaustion():
    async def s1(ctx): return RecoveryAction.RETRY()

    seq = FailurePolicy.sequence(s1)
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=[Step(index=0, action="step")],
        critical_step_index=0,
        original_task="task",
        attempt_history=[(FailureType.UNKNOWN, "retry")],  # s1 already used
    )
    action = await seq(ctx)
    assert action.kind == "escalate"
    assert "exhausted" in action.params.get("message", "")


async def test_sequence_counts_only_matching_failure_type():
    """Prior attempts of a different failure type must not advance the index."""
    async def s1(ctx): return RecoveryAction.RETRY(hint="s1")
    async def s2(ctx): return RecoveryAction.REPLAN(hint="s2")

    seq = FailurePolicy.sequence(s1, s2)
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=[Step(index=0, action="step")],
        critical_step_index=0,
        original_task="task",
        # prior attempt was for a DIFFERENT type — should not count
        attempt_history=[(FailureType.EXTERNAL_FAULT, "retry")],
    )
    action = await seq(ctx)
    assert action.kind == "retry"
    assert action.params["hint"] == "s1"


async def test_sequence_three_strategies_in_order():
    results = []

    async def s1(ctx):
        results.append("s1")
        return RecoveryAction.RETRY()

    async def s2(ctx):
        results.append("s2")
        return RecoveryAction.REPLAN()

    async def s3(ctx):
        results.append("s3")
        return RecoveryAction.ROLLBACK()

    seq = FailurePolicy.sequence(s1, s2, s3)
    ft = FailureType.UNKNOWN

    for i in range(3):
        ctx = FailureContext(
            failure_type=ft,
            trajectory=[Step(index=0, action="step")],
            critical_step_index=0,
            original_task="task",
            attempt_history=[(ft, "x")] * i,
        )
        await seq(ctx)

    assert results == ["s1", "s2", "s3"]


def test_sequence_requires_at_least_one_strategy():
    with pytest.raises(ValueError, match="at least one"):
        FailurePolicy.sequence()


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


# ── FailurePolicy.from_dict ───────────────────────────────────────────────────

async def test_from_dict_resolves_strategy():
    async def strategy(ctx): return RecoveryAction.RETRY()
    policy = FailurePolicy.from_dict({FailureType.EXTERNAL_FAULT: strategy})
    ctx = make_ctx(FailureType.EXTERNAL_FAULT)
    action = await policy.dispatch(ctx)
    assert action.kind == "retry"


async def test_from_dict_with_default():
    async def strategy(ctx): return RecoveryAction.RETRY()
    policy = FailurePolicy.from_dict(
        {FailureType.EXTERNAL_FAULT: strategy},
        default=FailurePolicy.escalate_by_default(),
    )
    ctx = make_ctx(FailureType.UNKNOWN)
    action = await policy.dispatch(ctx)
    assert action.kind == "escalate"


async def test_from_dict_unregistered_type_escalates_without_default():
    policy = FailurePolicy.from_dict(
        {FailureType.EXTERNAL_FAULT: FailurePolicy.escalate_by_default()}
    )
    ctx = make_ctx(FailureType.TIMEOUT)
    action = await policy.dispatch(ctx)
    assert action.kind == "escalate"


def test_from_dict_multiple_types():
    async def s(ctx): return RecoveryAction.RETRY()
    policy = FailurePolicy.from_dict({
        FailureType.EXTERNAL_FAULT: s,
        FailureType.TIMEOUT: s,
        FailureType.SCHEMA_MISMATCH: s,
    })
    assert policy.resolve(FailureType.EXTERNAL_FAULT) is s
    assert policy.resolve(FailureType.TIMEOUT) is s
    assert policy.resolve(FailureType.SCHEMA_MISMATCH) is s
    assert policy.resolve(FailureType.UNKNOWN) is None


# ── FailurePolicy.from_yaml ───────────────────────────────────────────────────

def test_from_yaml_toml_format(tmp_path):
    config = tmp_path / "policy.toml"
    config.write_text(
        '[EXTERNAL_FAULT]\nstrategy = "backoff_and_retry"\nmax_attempts = 3\n'
        '\n[WRONG_TOOL_CALLED]\nstrategy = "retry_with_tool_manifest"\nmax_attempts = 2\n'
    )
    policy = FailurePolicy.from_yaml(str(config))
    assert policy.resolve(FailureType.EXTERNAL_FAULT) is not None
    assert policy.resolve(FailureType.WRONG_TOOL_CALLED) is not None
    assert policy.resolve(FailureType.UNKNOWN) is None


def test_from_yaml_default_escalate(tmp_path):
    config = tmp_path / "policy.toml"
    config.write_text('default = "escalate"\n')
    policy = FailurePolicy.from_yaml(str(config))
    assert policy.resolve(FailureType.UNKNOWN) is not None


def test_from_yaml_unknown_strategy_raises(tmp_path):
    import pytest
    config = tmp_path / "policy.toml"
    config.write_text('[EXTERNAL_FAULT]\nstrategy = "does_not_exist"\n')
    with pytest.raises(ValueError, match="Unknown strategy"):
        FailurePolicy.from_yaml(str(config))


def test_from_yaml_custom_registry(tmp_path):
    async def my_strategy(ctx): return RecoveryAction.RETRY()
    config = tmp_path / "policy.toml"
    config.write_text('[EXTERNAL_FAULT]\nstrategy = "my_custom"\n')
    policy = FailurePolicy.from_yaml(
        str(config), strategy_registry={"my_custom": lambda **_: my_strategy}
    )
    assert policy.resolve(FailureType.EXTERNAL_FAULT) is my_strategy


def test_from_yaml_unknown_failure_type_raises(tmp_path):
    import pytest
    config = tmp_path / "policy.toml"
    config.write_text('[NOT_A_REAL_TYPE]\nstrategy = "escalate"\n')
    with pytest.raises(ValueError, match="Unknown FailureType"):
        FailurePolicy.from_yaml(str(config))


def test_from_yaml_unsupported_extension_raises(tmp_path):
    import pytest
    config = tmp_path / "policy.json"
    config.write_text('{}')
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        FailurePolicy.from_yaml(str(config))
