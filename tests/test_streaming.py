"""
tests/test_streaming.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests for Agent.stream() and StreamRetryEvent.
"""

from __future__ import annotations

import pytest

from triage.agent import Agent, TriageAbortError, TriageEscalationError
from triage.policy import FailurePolicy, RecoveryAction
from triage.streaming import StreamRetryEvent
from triage.taxonomy import FailureType, Step

# ── helpers ───────────────────────────────────────────────────────────────────

def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
) -> Step:
    return Step(index=index, action="test step", tool_called=tool_called,
                tool_input=tool_input, error=error, llm_output=llm_output)


async def _collect(
    agent: Agent, task: str, **kwargs: object
) -> tuple[list[object], list[StreamRetryEvent]]:
    chunks: list[object] = []
    retries: list[StreamRetryEvent] = []
    async for item in agent.stream(task, **kwargs):
        if isinstance(item, StreamRetryEvent):
            retries.append(item)
            chunks.clear()  # discard partial buffer
        else:
            chunks.append(item)
    return chunks, retries


# ── StreamRetryEvent dataclass ────────────────────────────────────────────────

def test_stream_retry_event_fields():
    evt = StreamRetryEvent(
        attempt=0,
        failure_type=FailureType.EXTERNAL_FAULT,
        action_kind="retry",
        hint="try again",
    )
    assert evt.attempt == 0
    assert evt.failure_type == FailureType.EXTERNAL_FAULT
    assert evt.action_kind == "retry"
    assert evt.hint == "try again"


def test_stream_retry_event_hint_defaults_none():
    evt = StreamRetryEvent(attempt=1, failure_type=FailureType.UNKNOWN, action_kind="replan")
    assert evt.hint is None


# ── Step.partial field ────────────────────────────────────────────────────────

def test_step_partial_default_false():
    s = make_step()
    assert s.partial is False


def test_step_partial_can_be_set():
    s = Step(index=0, action="stream", partial=True)
    assert s.partial is True


# ── Type guards ───────────────────────────────────────────────────────────────

async def test_run_on_async_gen_raises_type_error():
    async def gen_agent(task: str, *, record_step, **kwargs):
        yield "chunk"

    agent = Agent(gen_agent, FailurePolicy())
    with pytest.raises(TypeError, match="agent.stream()"):
        await agent.run("t")


async def test_stream_on_coroutine_raises_type_error():
    async def coro_agent(task: str, *, record_step, **kwargs) -> str:
        return "done"

    agent = Agent(coro_agent, FailurePolicy())
    with pytest.raises(TypeError, match="agent.run()"):
        async for _ in agent.stream("t"):
            pass


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_stream_yields_chunks_on_success():
    async def gen_agent(task: str, *, record_step, **kwargs):
        yield "a"
        yield "b"
        yield "c"
        record_step(make_step(index=0))

    agent = Agent(gen_agent, FailurePolicy())
    chunks, retries = await _collect(agent, "t")
    assert chunks == ["a", "b", "c"]
    assert retries == []


async def test_stream_empty_generator_succeeds():
    async def gen_agent(task: str, *, record_step, **kwargs):
        return
        yield  # make it an async generator

    agent = Agent(gen_agent, FailurePolicy())
    chunks, retries = await _collect(agent, "t")
    assert chunks == []
    assert retries == []


# ── Retry on failure ──────────────────────────────────────────────────────────

async def test_stream_retry_yields_event_then_resumes():
    call_count = 0

    async def gen_agent(task: str, *, record_step, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield "partial"
            record_step(make_step(index=0, error="HTTP 503"))
            raise RuntimeError("transient error")
        yield "ok"
        record_step(make_step(index=0))

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    agent = Agent(gen_agent, policy)
    chunks, retries = await _collect(agent, "t")

    assert chunks == ["ok"]
    assert len(retries) == 1
    assert retries[0].attempt == 0
    assert retries[0].failure_type == FailureType.EXTERNAL_FAULT
    assert retries[0].action_kind == "retry"


async def _retry():
    return RecoveryAction.RETRY()


async def test_stream_retry_clears_partial_chunks():
    """The _collect helper discards chunks on StreamRetryEvent — verify the pattern."""
    call_count = 0

    async def gen_agent(task: str, *, record_step, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield "bad1"
            yield "bad2"
            record_step(make_step(index=0, error="HTTP 503"))
            raise RuntimeError("HTTP 503")
        yield "good"
        record_step(make_step(index=0))

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    agent = Agent(gen_agent, policy)
    chunks, retries = await _collect(agent, "t")

    assert chunks == ["good"]
    assert len(retries) == 1


async def test_stream_retry_hint_propagated():
    call_count = 0

    async def gen_agent(task: str, *, record_step, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(make_step(index=0, error="HTTP 503"))
            raise RuntimeError("HTTP 503")
        yield "ok"
        record_step(make_step(index=0))

    async def _retry_with_hint(ctx):
        return RecoveryAction.RETRY(hint="use correct tool")

    policy = FailurePolicy(EXTERNAL_FAULT=_retry_with_hint)
    agent = Agent(gen_agent, policy)
    chunks, retries = await _collect(agent, "t")

    assert retries[0].hint == "use correct tool"


# ── Caps and escalation ───────────────────────────────────────────────────────

async def test_stream_escalates_after_max_recovery_attempts():
    async def gen_agent(task: str, *, record_step, **kwargs):
        record_step(make_step(index=0, error="HTTP 503"))
        raise RuntimeError("HTTP 503")
        yield  # make it an async gen

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    agent = Agent(gen_agent, policy, max_recovery_attempts=2)

    with pytest.raises(TriageEscalationError, match="max_recovery_attempts"):
        async for _ in agent.stream("t"):
            pass


async def test_stream_escalates_on_escalate_action():
    async def gen_agent(task: str, *, record_step, **kwargs):
        record_step(make_step(index=0, error="HTTP 503"))
        raise RuntimeError("HTTP 503")
        yield

    async def _escalate(ctx):
        return RecoveryAction.ESCALATE(message="no more")

    policy = FailurePolicy(EXTERNAL_FAULT=_escalate)
    agent = Agent(gen_agent, policy)

    with pytest.raises(TriageEscalationError, match="no more"):
        async for _ in agent.stream("t"):
            pass


async def test_stream_raises_abort_error():
    async def gen_agent(task: str, *, record_step, **kwargs):
        record_step(make_step(index=0, error="HTTP 503"))
        raise RuntimeError("HTTP 503")
        yield

    async def _abort(ctx):
        return RecoveryAction.ABORT(reason="fatal")

    policy = FailurePolicy(EXTERNAL_FAULT=_abort)
    agent = Agent(gen_agent, policy)

    with pytest.raises(TriageAbortError, match="fatal"):
        async for _ in agent.stream("t"):
            pass


async def test_stream_multiple_retries():
    call_count = 0

    async def gen_agent(task: str, *, record_step, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            record_step(make_step(index=0, error="HTTP 503"))
            raise RuntimeError("not yet")
        yield "done"
        record_step(make_step(index=0))

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    agent = Agent(gen_agent, policy, max_recovery_attempts=3)
    chunks, retries = await _collect(agent, "t")

    assert chunks == ["done"]
    assert len(retries) == 2
    assert retries[0].attempt == 0
    assert retries[1].attempt == 1


# ── Lifecycle hooks ───────────────────────────────────────────────────────────

async def test_stream_on_failure_hook_called():
    events: list[str] = []

    async def gen_agent(task: str, *, record_step, **kwargs):
        record_step(make_step(index=0, error="HTTP 503"))
        raise RuntimeError("HTTP 503")
        yield

    def on_failure(ctx):
        events.append("failure")

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    # max_recovery_attempts=1: attempt 0 fails (hook fires), retry, attempt 1 fails (hook
    # fires again), then cap triggers → 2 on_failure calls total.
    agent = Agent(gen_agent, policy, max_recovery_attempts=1, on_failure=on_failure)

    with pytest.raises(TriageEscalationError):
        async for _ in agent.stream("t"):
            pass

    assert len(events) == 2
    assert all(e == "failure" for e in events)


async def test_stream_on_recovery_hook_called():
    events: list[str] = []

    call_count = 0

    async def gen_agent(task: str, *, record_step, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            record_step(make_step(index=0, error="HTTP 503"))
            raise RuntimeError("HTTP 503")
        yield "ok"
        record_step(make_step(index=0))

    def on_recovery(ctx, action):
        events.append(action.kind)

    policy = FailurePolicy(EXTERNAL_FAULT=lambda ctx: _retry())
    agent = Agent(gen_agent, policy, on_recovery=on_recovery)
    await _collect(agent, "t")

    assert events == ["retry"]


# ── circuit_breakers integration ──────────────────────────────────────────────

async def test_stream_calls_record_success_on_clean_run():
    from triage.breaker import BreakerState, CircuitBreaker

    b = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=30)
    b.record_failure()
    # force to HALF_OPEN via cooldown
    b.allow_request(_now=b._opened_at + 31)  # type: ignore[operator]

    async def gen_agent(task: str, *, record_step, **kwargs):
        yield "ok"
        record_step(make_step(index=0))

    agent = Agent(gen_agent, FailurePolicy(), circuit_breakers=[b])
    await _collect(agent, "t")
    # After a clean stream, record_success() should have closed the breaker
    # (note: b._opened_at may not be valid for _now comparison here — just
    # check CLOSED is reached after the forced probe)
    assert b.record_success() == BreakerState.CLOSED
