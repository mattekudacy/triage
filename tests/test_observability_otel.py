"""Tests for triage.observability.otel — OpenTelemetry span integration.

Tests that need OTel are skipped when opentelemetry-sdk is not installed.
One test verifies the zero-cost no-op path when OTel is absent or unconfigured.
"""

from __future__ import annotations

import pytest

from triage.taxonomy import Step

# ── helpers ───────────────────────────────────────────────────────────────────

def make_step(
    index: int = 0,
    tool_called: str | None = None,
    error: str | None = None,
) -> Step:
    return Step(index=index, action="test step", tool_called=tool_called, error=error)


def _otel_available() -> bool:
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


# ── no-op path (OTel absent or no provider) ──────────────────────────────────

def test_resolve_tracer_returns_none_when_otel_absent(monkeypatch):
    """resolve_tracer() must return None when opentelemetry is not installed."""
    import triage.observability.otel as otel_mod
    monkeypatch.setattr(otel_mod, "_OTEL_AVAILABLE", False)
    assert otel_mod.resolve_tracer(None) is None


def test_resolve_tracer_returns_explicit_even_without_provider(monkeypatch):
    """An explicitly passed tracer is always returned regardless of provider state."""
    import triage.observability.otel as otel_mod
    monkeypatch.setattr(otel_mod, "_OTEL_AVAILABLE", False)
    sentinel = object()
    assert otel_mod.resolve_tracer(sentinel) is sentinel


async def test_run_noop_when_no_tracer_does_not_raise():
    """run_span() with tracer=None must be a silent no-op."""
    from triage.observability.otel import run_span
    async with run_span(None, "run-id", "task") as span:
        assert span is None


def test_classify_noop_when_no_tracer_does_not_raise():
    """classify_span() with tracer=None must be a silent no-op."""
    from triage.observability.otel import classify_span
    with classify_span(None, "run-id") as span:
        assert span is None


def test_dispatch_noop_when_no_tracer_does_not_raise():
    """dispatch_span() with tracer=None must be a silent no-op."""
    from triage.observability.otel import dispatch_span
    with dispatch_span(None, "run-id", attempt=0) as span:
        assert span is None


def test_set_helpers_noop_on_none_span():
    """set_span_* helpers must be silent no-ops when span is None."""
    from triage.observability.otel import (
        set_span_classify_result,
        set_span_dispatch_result,
        set_span_run_outcome,
    )
    set_span_classify_result(None, "external_fault")
    set_span_dispatch_result(None, "retry", "external_fault")
    set_span_run_outcome(None)
    set_span_run_outcome(None, error=RuntimeError("boom"))


# ── agent integration with no tracer ─────────────────────────────────────────

async def test_agent_run_produces_correct_result_without_tracer():
    """Agent.run() output must be identical whether or not a tracer is active."""
    import triage
    from triage.policy import FailurePolicy

    async def my_agent(task: str, *, record_step, **kwargs) -> str:
        record_step(Step(index=0, action="work"))
        return f"done:{task}"

    agent = triage.Agent(my_agent, policy=FailurePolicy())
    result = await agent.run("hello")
    assert result == "done:hello"


# ── OTel span tree (skipped without opentelemetry-sdk) ───────────────────────

pytestmark_otel = pytest.mark.skipif(
    not _otel_available(),
    reason="opentelemetry-sdk not installed",
)


@pytestmark_otel
def test_resolve_tracer_returns_none_for_noop_provider():
    """With OTel installed but no provider set, resolve_tracer() returns None."""
    from opentelemetry import trace

    from triage.observability.otel import resolve_tracer

    # Save and reset to the default proxy provider
    original = trace.get_tracer_provider()
    try:
        # The default ProxyTracerProvider should yield None
        result = resolve_tracer(None)
        assert result is None
    finally:
        trace.set_tracer_provider(original)


@pytestmark_otel
def test_resolve_tracer_returns_explicit_tracer_directly():
    """An explicitly passed tracer is returned unchanged regardless of global state."""
    from opentelemetry.sdk.trace import TracerProvider

    from triage.observability.otel import resolve_tracer

    provider = TracerProvider()
    explicit_tracer = provider.get_tracer("triage-test")
    result = resolve_tracer(explicit_tracer)
    assert result is explicit_tracer


@pytestmark_otel
async def test_agent_emits_run_classify_dispatch_spans():
    """A full Agent.run() with failure + recovery must produce the expected span tree."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import triage
    from triage.policy import FailurePolicy, RecoveryAction

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("triage-test")

    calls = []

    async def flaky(task: str, *, record_step, **kwargs) -> str:
        calls.append(len(calls))
        record_step(Step(index=0, action="attempt"))
        if len(calls) == 1:
            raise RuntimeError("first failure")
        return "ok"

    async def retry_strategy(ctx):
        return RecoveryAction.RETRY()

    agent = triage.Agent(
        flaky,
        policy=FailurePolicy(default=retry_strategy),
        max_recovery_attempts=3,
        tracer=tracer,
    )
    result = await agent.run("task")
    assert result == "ok"

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert "triage.run" in span_names
    assert "triage.classify" in span_names
    assert "triage.dispatch" in span_names

    # run span must carry run_id and task attributes
    run_span_obj = next(s for s in spans if s.name == "triage.run")
    assert run_span_obj.attributes.get("triage.task") == "task"
    run_id = run_span_obj.attributes.get("triage.run_id")
    assert run_id  # non-empty

    # classify span must carry the failure_type
    classify_span_obj = next(s for s in spans if s.name == "triage.classify")
    assert classify_span_obj.attributes.get("triage.failure_type") is not None

    # dispatch span must carry action_kind
    dispatch_span_obj = next(s for s in spans if s.name == "triage.dispatch")
    assert dispatch_span_obj.attributes.get("triage.action_kind") == "retry"

    # all child spans must share the same trace_id as the root run span
    root_trace_id = run_span_obj.context.trace_id
    for s in spans:
        assert s.context.trace_id == root_trace_id, f"{s.name} has wrong trace_id"


@pytestmark_otel
async def test_escalate_marks_dispatch_span_as_error():
    """A dispatch that results in escalate must set the dispatch span status to ERROR."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    import triage
    from triage.policy import FailurePolicy

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("triage-test")

    async def always_fails(task: str, *, record_step, **kwargs) -> str:
        record_step(Step(index=0, action="boom"))
        raise RuntimeError("permanent failure")

    agent = triage.Agent(
        always_fails,
        policy=FailurePolicy(),  # no strategies -> escalate by default
        max_recovery_attempts=1,
        tracer=tracer,
    )

    with pytest.raises(triage.TriageEscalationError):
        await agent.run("task")

    spans = exporter.get_finished_spans()
    dispatch_spans = [s for s in spans if s.name == "triage.dispatch"]
    assert dispatch_spans, "expected at least one dispatch span"
    ds = dispatch_spans[0]
    assert ds.attributes.get("triage.action_kind") == "escalate"
    assert ds.status.status_code == StatusCode.ERROR


@pytestmark_otel
async def test_run_id_is_same_across_classify_and_dispatch_spans():
    """All spans from one Agent.run() must share the same triage.run_id attribute."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import triage
    from triage.policy import FailurePolicy

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("triage-test")

    calls = []

    async def flaky(task: str, *, record_step, **kwargs) -> str:
        calls.append(1)
        record_step(Step(index=0, action="step"))
        if len(calls) == 1:
            raise RuntimeError("fail once")
        return "ok"

    async def retry_strategy(ctx):
        return __import__("triage").RecoveryAction.RETRY()

    agent = triage.Agent(
        flaky,
        policy=FailurePolicy(default=retry_strategy),
        max_recovery_attempts=2,
        tracer=tracer,
    )
    await agent.run("task")

    spans = exporter.get_finished_spans()
    run_id = next(
        s.attributes["triage.run_id"]
        for s in spans
        if s.name == "triage.run"
    )
    for s in spans:
        if "triage.run_id" in (s.attributes or {}):
            assert s.attributes["triage.run_id"] == run_id, (
                f"{s.name} has different run_id"
            )
