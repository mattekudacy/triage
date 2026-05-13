"""Tests for triage.strategies — retry, replan, rollback."""

import pytest

from triage.policy import RecoveryAction
from triage.strategies.replan import replan, resume_from_subgoal
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.taxonomy import FailureContext, FailureType, Step


def make_ctx(
    failure_type: FailureType = FailureType.UNKNOWN,
    attempt_history: list[tuple[FailureType, str]] | None = None,
    metadata: dict | None = None,
) -> FailureContext:
    return FailureContext(
        failure_type=failure_type,
        trajectory=[Step(index=0, action="step")],
        critical_step_index=0,
        original_task="test task",
        attempt_history=attempt_history or [],
        metadata=metadata or {},
    )


# ── replan() ─────────────────────────────────────────────────────────────────

async def test_replan_returns_replan_action():
    strategy = replan()
    ctx = make_ctx()
    action = await strategy(ctx)
    assert action.kind == "replan"


async def test_replan_injects_hint():
    strategy = replan(hint="try again differently")
    ctx = make_ctx()
    action = await strategy(ctx)
    assert action.params["hint"] == "try again differently"


async def test_replan_default_hint_when_none():
    strategy = replan()
    ctx = make_ctx()
    action = await strategy(ctx)
    assert "hint" in action.params
    assert action.params["hint"]  # non-empty


async def test_replan_escalates_after_max_replans():
    strategy = replan(max_replans=2)
    history = [
        (FailureType.UNKNOWN, "replan"),
        (FailureType.UNKNOWN, "replan"),
    ]
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "max_replans" in action.params.get("message", "")


async def test_replan_counts_only_replan_entries():
    strategy = replan(max_replans=2)
    # 2 retries + 1 replan = only 1 replan counted, should still replan
    history = [
        (FailureType.UNKNOWN, "retry"),
        (FailureType.UNKNOWN, "retry"),
        (FailureType.UNKNOWN, "replan"),
    ]
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "replan"


async def test_replan_max_replans_zero_escalates_immediately():
    strategy = replan(max_replans=0)
    ctx = make_ctx()
    action = await strategy(ctx)
    assert action.kind == "escalate"


async def test_replan_exactly_at_limit_escalates():
    strategy = replan(max_replans=3)
    history = [(FailureType.UNKNOWN, "replan")] * 3
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"


async def test_replan_one_below_limit_replans():
    strategy = replan(max_replans=3)
    history = [(FailureType.UNKNOWN, "replan")] * 2
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "replan"


# ── resume_from_subgoal() ─────────────────────────────────────────────────────

async def test_resume_returns_resume_action():
    strategy = resume_from_subgoal()
    ctx = make_ctx(metadata={"incomplete_subgoal": "step-3"})
    action = await strategy(ctx)
    assert action.kind == "resume"
    assert action.params["from_subgoal"] == "step-3"


async def test_resume_escalates_when_key_missing():
    strategy = resume_from_subgoal()
    ctx = make_ctx()  # no metadata
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "incomplete_subgoal" in action.params.get("message", "")


async def test_resume_escalates_when_metadata_empty():
    strategy = resume_from_subgoal()
    ctx = make_ctx(metadata={})
    action = await strategy(ctx)
    assert action.kind == "escalate"


# ── retry_with_tool_manifest() ────────────────────────────────────────────────

async def test_retry_with_tool_manifest_returns_retry():
    strategy = retry_with_tool_manifest()
    ctx = make_ctx()
    action = await strategy(ctx)
    assert action.kind == "retry"


async def test_retry_with_tool_manifest_hint_present():
    strategy = retry_with_tool_manifest()
    ctx = make_ctx()
    action = await strategy(ctx)
    assert "hint" in action.params


async def test_retry_with_tool_manifest_escalates_after_max():
    strategy = retry_with_tool_manifest(max_attempts=2)
    history = [
        (FailureType.WRONG_TOOL_CALLED, "retry"),
        (FailureType.WRONG_TOOL_CALLED, "retry"),
    ]
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "max_attempts" in action.params.get("message", "")


async def test_retry_with_tool_manifest_counts_only_retry_entries():
    strategy = retry_with_tool_manifest(max_attempts=2)
    history = [
        (FailureType.UNKNOWN, "replan"),
        (FailureType.UNKNOWN, "replan"),
        (FailureType.WRONG_TOOL_CALLED, "retry"),
    ]
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "retry"


async def test_retry_with_tool_manifest_exactly_at_limit_escalates():
    strategy = retry_with_tool_manifest(max_attempts=3)
    history = [(FailureType.WRONG_TOOL_CALLED, "retry")] * 3
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"


# ── backoff_and_retry() ───────────────────────────────────────────────────────

async def test_backoff_and_retry_returns_retry():
    strategy = backoff_and_retry()
    ctx = make_ctx()
    action = await strategy(ctx)
    assert action.kind == "retry"


async def test_backoff_and_retry_has_delay():
    strategy = backoff_and_retry()
    ctx = make_ctx(metadata={"attempt_number": 1})
    action = await strategy(ctx)
    assert action.params.get("delay", 0) > 0


async def test_backoff_and_retry_delay_grows_exponentially():
    strategy = backoff_and_retry()
    ctx0 = make_ctx(metadata={"attempt_number": 0})
    ctx1 = make_ctx(metadata={"attempt_number": 1})
    ctx2 = make_ctx(metadata={"attempt_number": 2})
    a0 = await strategy(ctx0)
    a1 = await strategy(ctx1)
    a2 = await strategy(ctx2)
    assert a1.params["delay"] > a0.params["delay"]
    assert a2.params["delay"] > a1.params["delay"]


async def test_backoff_and_retry_escalates_after_max():
    strategy = backoff_and_retry(max_attempts=3)
    history = [(FailureType.EXTERNAL_FAULT, "retry")] * 3
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "max_attempts" in action.params.get("message", "")


async def test_backoff_and_retry_counts_only_retry_entries():
    strategy = backoff_and_retry(max_attempts=2)
    history = [
        (FailureType.EXTERNAL_FAULT, "replan"),
        (FailureType.EXTERNAL_FAULT, "retry"),
    ]
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "retry"


async def test_backoff_and_retry_exactly_at_limit_escalates():
    strategy = backoff_and_retry(max_attempts=5)
    history = [(FailureType.EXTERNAL_FAULT, "retry")] * 5
    ctx = make_ctx(attempt_history=history)
    action = await strategy(ctx)
    assert action.kind == "escalate"
