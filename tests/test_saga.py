"""
tests/test_saga.py
~~~~~~~~~~~~~~~~~~
Tests for saga / compensating rollback:
- Compensators run in reverse step-index order on ROLLBACK
- Compensator errors are swallowed; recovery continues
- Compensators are cleared between loop iterations
- Async compensators are awaited correctly
- Both kwarg injection and get_compensator_recorder() contextvar work
- Stream path also clears compensators per iteration
- No compensators registered → ROLLBACK is a no-op (passes through)
"""

from __future__ import annotations

import pytest

from triage.agent import Agent, get_compensator_recorder
from triage.policy import FailurePolicy
from triage.strategies.saga import compensating_rollback
from triage.streaming import StreamRetryEvent
from triage.taxonomy import FailureContext, FailureType, Step


def make_step(index: int = 0, error: str | None = None) -> Step:
    return Step(index=index, action="test step", error=error)


def _rollback_policy() -> FailurePolicy:
    return FailurePolicy(default=compensating_rollback())


# ── Reverse-order execution ───────────────────────────────────────────────────


async def test_compensators_run_in_reverse_step_order():
    """Compensators registered for later steps fire before earlier ones."""
    order: list[int] = []
    call_count = 0

    async def my_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(Step(index=1, action="step1"))
            record_compensator(1, lambda: order.append(1))
            record_step(Step(index=2, action="step2"))
            record_compensator(2, lambda: order.append(2))
            record_step(Step(index=3, action="step3"))
            record_compensator(3, lambda: order.append(3))
            raise RuntimeError("boom")
        return "recovered"

    agent = Agent(
        my_agent,
        _rollback_policy(),
        auto_checkpoint=True,
        max_recovery_attempts=3,
    )
    result = await agent.run("t")
    assert result == "recovered"
    assert order == [3, 2, 1]


async def test_compensators_registered_out_of_order_still_reverse():
    """Registration order doesn't matter — it's step_index that determines sequence."""
    order: list[int] = []
    call_count = 0

    async def my_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Register step 3 before step 1
            record_compensator(3, lambda: order.append(3))
            record_compensator(1, lambda: order.append(1))
            record_compensator(2, lambda: order.append(2))
            record_step(Step(index=1, action="s1"))
            raise RuntimeError("fail")
        return "ok"

    agent = Agent(my_agent, _rollback_policy(), auto_checkpoint=True)
    await agent.run("t")
    assert order == [3, 2, 1]


# ── Compensator errors don't abort recovery ───────────────────────────────────


async def test_compensator_error_is_swallowed():
    """A failing compensator is logged but recovery continues."""
    call_count = 0

    async def my_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(Step(index=1, action="s1"))

            def _bad():
                raise ValueError("undo failed")

            record_compensator(1, _bad)
            raise RuntimeError("fail")
        return "recovered"

    agent = Agent(my_agent, _rollback_policy(), auto_checkpoint=True)
    result = await agent.run("t")
    assert result == "recovered"


# ── Async compensators ────────────────────────────────────────────────────────


async def test_async_compensator_is_awaited():
    """Compensators that return awaitables are properly awaited."""
    awaited: list[bool] = []
    call_count = 0

    async def my_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(Step(index=1, action="s1"))

            async def _undo():
                awaited.append(True)

            record_compensator(1, _undo)
            raise RuntimeError("fail")
        return "ok"

    agent = Agent(my_agent, _rollback_policy(), auto_checkpoint=True)
    await agent.run("t")
    assert awaited == [True]


# ── Cleared between iterations ────────────────────────────────────────────────


async def test_compensators_cleared_between_attempts():
    """After a ROLLBACK, the compensator list resets so attempt 2 starts fresh."""
    compensation_calls: list[int] = []
    call_count = 0

    async def my_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        record_step(Step(index=call_count, action=f"step-{call_count}"))
        record_compensator(call_count, lambda n=call_count: compensation_calls.append(n))
        if call_count < 3:
            raise RuntimeError("keep failing")
        return "ok"

    agent = Agent(
        my_agent,
        _rollback_policy(),
        auto_checkpoint=True,
        max_recovery_attempts=5,
    )
    await agent.run("t")
    # Attempt 1 registers compensator 1 → fires on ROLLBACK
    # Attempt 2 registers compensator 2 → fires on ROLLBACK
    # Attempt 3 succeeds, no compensators fired
    assert compensation_calls == [1, 2]


# ── contextvar accessor ───────────────────────────────────────────────────────


async def test_get_compensator_recorder_works_inside_run():
    """get_compensator_recorder() contextvar binds the same recorder as the kwarg."""
    order: list[int] = []
    call_count = 0

    async def my_agent(task, *, record_step, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(Step(index=1, action="s1"))
            # use contextvar instead of kwarg
            get_compensator_recorder()(1, lambda: order.append(1))
            raise RuntimeError("fail")
        return "ok"

    agent = Agent(my_agent, _rollback_policy(), auto_checkpoint=True)
    await agent.run("t")
    assert order == [1]


async def test_get_compensator_recorder_raises_outside_run():
    with pytest.raises(RuntimeError, match="outside a triage Agent.run"):
        get_compensator_recorder()


# ── No compensators → normal ROLLBACK ────────────────────────────────────────


async def test_rollback_with_no_compensators_works():
    """compensating_rollback() with no registered compensators behaves like ROLLBACK."""
    call_count = 0

    async def my_agent(task, *, record_step, **kw):
        nonlocal call_count
        call_count += 1
        record_step(Step(index=1, action="s1"))
        if call_count == 1:
            raise RuntimeError("fail")
        return "recovered"

    agent = Agent(my_agent, _rollback_policy(), auto_checkpoint=True)
    result = await agent.run("t")
    assert result == "recovered"


# ── Stream path ───────────────────────────────────────────────────────────────


async def test_stream_path_clears_compensators_between_iterations():
    """In the stream path, compensators also reset on each iteration."""
    compensation_calls: list[int] = []
    call_count = 0

    async def my_stream_agent(task, *, record_step, record_compensator, **kw):
        nonlocal call_count
        call_count += 1
        record_step(Step(index=call_count, action=f"s{call_count}"))
        record_compensator(call_count, lambda n=call_count: compensation_calls.append(n))
        if call_count < 2:
            raise RuntimeError("fail once")
        yield "chunk"

    agent = Agent(my_stream_agent, _rollback_policy(), auto_checkpoint=True)
    chunks = [c async for c in agent.stream("t") if not isinstance(c, StreamRetryEvent)]
    assert chunks == ["chunk"]
    assert compensation_calls == [1]


# ── compensating_rollback strategy ───────────────────────────────────────────


async def test_compensating_rollback_returns_rollback_action():
    """compensating_rollback() returns RecoveryAction.ROLLBACK."""
    strategy = compensating_rollback()
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[make_step(0)],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    assert action.kind == "rollback"


async def test_compensating_rollback_uses_explicit_checkpoint_id():
    strategy = compensating_rollback(checkpoint_id="cp-abc")
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[make_step(0)],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    assert action.params.get("checkpoint_id") == "cp-abc"
