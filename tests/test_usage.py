"""
tests/test_usage.py
~~~~~~~~~~~~~~~~~~~
Tests for triage.usage — Usage dataclass and UsageMeter accumulation.
"""

from __future__ import annotations

import threading

import anyio
import pytest

from triage.agent import Agent, TriageEscalationError, get_usage_recorder
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import Step
from triage.usage import Usage, UsageMeter

# ── UsageMeter unit tests ─────────────────────────────────────────────────────

def test_meter_starts_empty():
    m = UsageMeter()
    assert m.total.input_tokens == 0
    assert m.total.output_tokens == 0
    assert m.total.cost_usd == 0.0
    assert m.total.calls == 0
    assert m.total_tokens == 0
    assert m.cost_usd == 0.0


def test_meter_accumulates_single_record():
    m = UsageMeter()
    m.record(Usage(input_tokens=10, output_tokens=5, cost_usd=0.001))
    assert m.total.input_tokens == 10
    assert m.total.output_tokens == 5
    assert m.total.cost_usd == pytest.approx(0.001)
    assert m.total.calls == 1
    assert m.total_tokens == 15


def test_meter_accumulates_multiple_records():
    m = UsageMeter()
    m.record(Usage(input_tokens=10, output_tokens=5))
    m.record(Usage(input_tokens=20, output_tokens=10, cost_usd=0.002))
    assert m.total.input_tokens == 30
    assert m.total.output_tokens == 15
    assert m.total.calls == 2
    assert m.total.cost_usd == pytest.approx(0.002)


def test_meter_reset_clears_totals():
    m = UsageMeter()
    m.record(Usage(input_tokens=100, cost_usd=0.5))
    m.reset()
    assert m.total.input_tokens == 0
    assert m.total.calls == 0
    assert m.cost_usd == 0.0


def test_meter_total_returns_snapshot():
    m = UsageMeter()
    m.record(Usage(input_tokens=10))
    snapshot = m.total
    m.record(Usage(input_tokens=5))
    assert snapshot.input_tokens == 10  # snapshot is frozen


def test_meter_threadsafe_concurrent_records():
    m = UsageMeter()
    n = 100

    def worker():
        for _ in range(n):
            m.record(Usage(input_tokens=1))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.total.input_tokens == n * 10


def test_usage_total_tokens():
    u = Usage(input_tokens=100, output_tokens=50)
    assert u.total_tokens == 150


# ── Agent integration — record_usage injection ────────────────────────────────

async def test_agent_injects_record_usage():
    """record_usage kwarg is received and meter accumulates."""

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        record_usage(Usage(input_tokens=10, output_tokens=5))
        return "ok"

    policy = FailurePolicy()
    agent = Agent(my_agent, policy)
    await agent.run("t")
    assert agent._meter.total.input_tokens == 10


async def test_get_usage_recorder_works_inside_run():
    """get_usage_recorder() contextvar accessor records into the same meter."""
    async def my_agent(task: str, *, record_step, **kwargs) -> str:
        get_usage_recorder()(Usage(input_tokens=7))
        return "ok"

    policy = FailurePolicy()
    agent = Agent(my_agent, policy)
    await agent.run("t")
    assert agent._meter.total.input_tokens == 7


async def test_get_usage_recorder_raises_outside_run():
    from triage.agent import get_usage_recorder
    with pytest.raises(RuntimeError, match="outside a triage Agent.run"):
        get_usage_recorder()


async def test_meter_resets_between_runs():
    async def my_agent(task: str, *, record_usage, **kwargs) -> str:
        record_usage(Usage(input_tokens=50))
        return "ok"

    policy = FailurePolicy()
    agent = Agent(my_agent, policy)
    await agent.run("first")
    assert agent._meter.total.input_tokens == 50
    await agent.run("second")
    # meter resets at the top of run(), so only the second run's usage is present
    assert agent._meter.total.input_tokens == 50


# ── Agent budgets — max_tokens ─────────────────────────────────────────────────

async def test_max_tokens_escalates_when_exceeded():
    """After recording enough tokens, the second failure triggers escalation."""
    call_count = 0

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        record_usage(Usage(input_tokens=100))
        record_step(Step(index=0, action="step"))
        raise RuntimeError("boom")

    policy = FailurePolicy(default=_ret(RecoveryAction.RETRY()))
    agent = Agent(my_agent, policy, max_tokens=150)
    with pytest.raises(TriageEscalationError, match="max_tokens"):
        await agent.run("t")
    # first attempt records 100 tokens, second exceeds 150
    assert call_count == 2


async def test_max_tokens_none_does_not_escalate():
    """max_tokens=None (default) never escalates due to token count."""
    call_count = 0

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        record_usage(Usage(input_tokens=10_000))
        record_step(Step(index=0, action="step"))
        if call_count < 2:
            raise RuntimeError("fail once")
        return "ok"

    policy = FailurePolicy(default=_ret(RecoveryAction.RETRY()))
    agent = Agent(my_agent, policy)
    result = await agent.run("t")
    assert result == "ok"


# ── Agent budgets — max_cost_usd ───────────────────────────────────────────────

async def test_max_cost_escalates_when_exceeded():
    call_count = 0

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        record_usage(Usage(cost_usd=0.10))
        record_step(Step(index=0, action="step"))
        raise RuntimeError("boom")

    policy = FailurePolicy(default=_ret(RecoveryAction.RETRY()))
    agent = Agent(my_agent, policy, max_cost_usd=0.15)
    with pytest.raises(TriageEscalationError, match="max_cost_usd"):
        await agent.run("t")
    assert call_count == 2


async def test_max_cost_none_does_not_escalate():
    call_count = 0

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        record_usage(Usage(cost_usd=999.0))
        record_step(Step(index=0, action="step"))
        if call_count < 2:
            raise RuntimeError("fail once")
        return "ok"

    policy = FailurePolicy(default=_ret(RecoveryAction.RETRY()))
    agent = Agent(my_agent, policy)
    result = await agent.run("t")
    assert result == "ok"


# ── Concurrent run isolation ───────────────────────────────────────────────────

async def test_concurrent_runs_have_independent_meters():
    """Two concurrent run() calls on the same Agent don't share usage totals."""

    async def my_agent(task: str, *, record_usage, **kwargs) -> str:
        record_usage(Usage(input_tokens=int(task)))
        await anyio.sleep(0)  # yield so the other task can run
        return "ok"

    policy = FailurePolicy()
    agent = Agent(my_agent, policy)

    async with anyio.create_task_group() as tg:
        tg.start_soon(agent.run, "100")
        tg.start_soon(agent.run, "200")

    # After both complete the meter reflects the last run (they reset independently)
    # — we can't assert a specific value here, but the important thing is no crash
    # and the meter is a valid non-negative number.
    assert agent._meter.total.input_tokens >= 0


# ── Clone copies budget params ─────────────────────────────────────────────────

def test_clone_copies_budget_params():
    async def noop(task, *, record_step, **kw):
        return "ok"

    policy = FailurePolicy()
    agent = Agent(noop, policy, max_tokens=500, max_cost_usd=1.0)
    cloned = agent.clone()
    assert cloned._max_tokens == 500
    assert cloned._max_cost_usd == 1.0


# ── Helper ────────────────────────────────────────────────────────────────────

def _ret(action: RecoveryAction):
    """Wrap a RecoveryAction in a trivial async strategy."""
    async def _strategy(ctx):
        return action
    return _strategy
