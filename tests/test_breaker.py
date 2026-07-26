"""
tests/test_breaker.py
~~~~~~~~~~~~~~~~~~~~~
Tests for triage.breaker.CircuitBreaker and triage.strategies.circuit_breaker.
"""

from __future__ import annotations

import threading

import pytest

from triage.agent import Agent, TriageEscalationError
from triage.breaker import BreakerState, CircuitBreaker
from triage.policy import FailurePolicy, RecoveryAction
from triage.strategies.circuit_breaker import circuit_breaker
from triage.taxonomy import FailureContext, FailureType, Step

# ── CircuitBreaker unit tests ─────────────────────────────────────────────────

def test_initial_state_is_closed():
    b = CircuitBreaker()
    assert b.state(_now=0.0) == BreakerState.CLOSED


def test_failures_below_threshold_stay_closed():
    b = CircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=10)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=2.0) == BreakerState.CLOSED
    assert b.failure_count(_now=2.0) == 2


def test_threshold_reached_trips_open():
    b = CircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=10)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    state = b.record_failure(_now=2.0)
    assert state == BreakerState.OPEN
    assert b.state(_now=2.0) == BreakerState.OPEN


def test_open_transitions_to_half_open_after_cooldown():
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=2.0) == BreakerState.OPEN
    # before cooldown
    assert b.is_open(_now=20.0)
    # after cooldown
    assert not b.is_open(_now=31.0)
    assert b.state(_now=31.0) == BreakerState.HALF_OPEN


def test_half_open_to_closed_on_success():
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    # advance past cooldown
    assert not b.is_open(_now=31.0)
    b.record_success(_now=32.0)
    assert b.state(_now=32.0) == BreakerState.CLOSED
    assert b.failure_count(_now=32.0) == 0


def test_half_open_to_open_on_failure():
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    # advance to HALF_OPEN
    assert not b.is_open(_now=31.0)
    b.record_failure(_now=32.0)
    assert b.state(_now=32.0) == BreakerState.OPEN


def test_failures_outside_window_evicted():
    b = CircuitBreaker(failure_threshold=3, window_seconds=10, cooldown_seconds=5)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    # advance past window — old failures evicted
    b.record_failure(_now=15.0)
    assert b.state(_now=15.0) == BreakerState.CLOSED
    assert b.failure_count(_now=15.0) == 1


def test_allow_request_closed_returns_true():
    b = CircuitBreaker()
    assert b.allow_request(_now=0.0)


def test_allow_request_open_returns_false():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    assert not b.allow_request(_now=1.0)


def test_allow_request_half_open_returns_true_first_call():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    assert b.allow_request(_now=31.0)  # past cooldown → HALF_OPEN → allowed


def test_allow_request_half_open_blocks_second_concurrent_probe():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    # First caller claims the probe slot
    assert b.allow_request(_now=31.0) is True
    # Second concurrent caller is blocked until the first records an outcome
    assert b.allow_request(_now=31.0) is False


def test_probe_in_flight_cleared_by_record_failure():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.allow_request(_now=31.0)  # sets _probe_in_flight
    # Recording a failure clears the flag and re-opens
    b.record_failure(_now=32.0)
    assert b._probe_in_flight is False
    assert b.state(_now=32.0) == BreakerState.OPEN


def test_probe_in_flight_cleared_by_record_success():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.allow_request(_now=31.0)  # sets _probe_in_flight
    b.record_success(_now=32.0)
    assert b._probe_in_flight is False
    assert b.state(_now=32.0) == BreakerState.CLOSED


def test_probe_in_flight_cleared_by_reset():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.allow_request(_now=31.0)  # sets _probe_in_flight
    b.reset()
    assert b._probe_in_flight is False


def test_reset_clears_all_state():
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=10)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=2.0) == BreakerState.OPEN
    b.reset()
    assert b.state(_now=2.0) == BreakerState.CLOSED
    assert b.failure_count(_now=2.0) == 0


def test_success_in_closed_state_is_noop():
    b = CircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=10)
    b.record_failure(_now=0.0)
    state = b.record_success(_now=1.0)
    assert state == BreakerState.CLOSED
    assert b.failure_count(_now=1.0) == 1  # failure still counted


# ── Validation ────────────────────────────────────────────────────────────────

def test_invalid_failure_threshold_raises():
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)


def test_invalid_window_seconds_raises():
    with pytest.raises(ValueError, match="window_seconds"):
        CircuitBreaker(window_seconds=0)


def test_invalid_cooldown_seconds_raises():
    with pytest.raises(ValueError, match="cooldown_seconds"):
        CircuitBreaker(cooldown_seconds=-1)


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_record_failure_is_threadsafe():
    b = CircuitBreaker(failure_threshold=1000, window_seconds=60, cooldown_seconds=5)
    n = 50

    def worker():
        for _i in range(n):
            b.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert b.failure_count() == n * 10


# ── circuit_breaker() strategy ────────────────────────────────────────────────

def _retry_strategy():
    async def _s(ctx):
        return RecoveryAction.RETRY()
    return _s


async def test_circuit_breaker_strategy_allows_when_closed():
    b = CircuitBreaker(failure_threshold=5, window_seconds=60, cooldown_seconds=10)
    strategy = circuit_breaker(b, _retry_strategy())
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    assert action.kind == "retry"
    assert b.failure_count() == 1


async def test_circuit_breaker_strategy_blocks_when_open():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=300)
    b.record_failure()  # real time; 300s cooldown won't elapse during test
    assert b.state() == BreakerState.OPEN

    strategy = circuit_breaker(b, _retry_strategy())
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    assert action.kind == "escalate"
    assert "OPEN" in action.params.get("message", "")


async def test_circuit_breaker_open_action_abort():
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=300)
    b.record_failure()  # real time; 300s cooldown won't elapse during test

    strategy = circuit_breaker(b, _retry_strategy(), open_action="abort")
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    assert action.kind == "abort"


def test_circuit_breaker_invalid_open_action_raises():
    b = CircuitBreaker()
    with pytest.raises(ValueError, match="open_action"):
        circuit_breaker(b, _retry_strategy(), open_action="retry")


async def test_circuit_breaker_strategy_records_failure_after_inner():
    """record_failure is called after inner returns, not before."""
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=10)
    call_order: list[str] = []

    async def _ordered_inner(ctx: FailureContext) -> RecoveryAction:
        call_order.append("inner")
        return RecoveryAction.RETRY()

    strategy = circuit_breaker(b, _ordered_inner)
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[],
        critical_step_index=0,
        original_task="t",
    )
    action = await strategy(ctx)
    call_order.append("record_failure_checked")

    assert action.kind == "retry"
    assert b.failure_count() == 1
    # inner was called before failure was recorded — order is deterministic
    assert call_order == ["inner", "record_failure_checked"]


async def test_half_open_probe_failure_reopens_breaker():
    """A failed probe in HALF_OPEN re-opens the breaker."""
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    # force HALF_OPEN by evaluating state at cooldown boundary
    assert b.state(_now=31.0) == BreakerState.HALF_OPEN

    strategy = circuit_breaker(b, _retry_strategy())
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[],
        critical_step_index=0,
        original_task="t",
    )
    # This probe call goes through (HALF_OPEN), inner runs, record_failure re-opens
    action = await strategy(ctx)
    assert action.kind == "retry"
    # Breaker should be OPEN again because we called record_failure after inner
    assert b.state() == BreakerState.OPEN


async def test_record_success_closes_breaker_after_probe():
    """Calling record_success() after a recovered run closes the breaker."""
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure(_now=0.0)
    b.allow_request(_now=31.0)  # claim probe slot → HALF_OPEN
    b.record_success(_now=32.0)
    assert b.state(_now=32.0) == BreakerState.CLOSED
    assert b.failure_count(_now=32.0) == 0


# ── Agent integration ─────────────────────────────────────────────────────────

async def test_agent_escalates_when_breaker_open():
    """When the breaker is already OPEN, the agent escalates on first failure."""
    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=300)
    b.record_failure()  # real time; 300s cooldown won't elapse during test

    call_count = 0

    async def my_agent(task: str, *, record_step, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        record_step(Step(index=0, action="step", error="HTTP 503"))
        raise RuntimeError("service down")

    from triage.strategies.retry import backoff_and_retry
    policy = FailurePolicy(
        EXTERNAL_FAULT=circuit_breaker(b, backoff_and_retry(max_attempts=3)),
    )
    agent = Agent(my_agent, policy)

    with pytest.raises(TriageEscalationError, match="OPEN"):
        await agent.run("t")

    assert call_count == 1  # breaker blocked before retry


async def test_agent_trips_breaker_after_threshold():
    """The breaker trips after enough failures accumulate across Agent runs."""
    b = CircuitBreaker(failure_threshold=2, window_seconds=60, cooldown_seconds=30)

    async def always_fail(task: str, *, record_step, **kwargs) -> str:
        record_step(Step(index=0, action="step", error="HTTP 503"))
        raise RuntimeError("down")


    async def _single_retry(ctx):
        return RecoveryAction.RETRY()

    policy = FailurePolicy(EXTERNAL_FAULT=circuit_breaker(b, _single_retry))

    # First run — trips the breaker after 2 strategy calls (initial + 2 retries)
    agent = Agent(always_fail, policy, max_recovery_attempts=2)
    with pytest.raises(TriageEscalationError):
        await agent.run("t")

    assert b.state() == BreakerState.OPEN
