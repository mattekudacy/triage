"""
tests/test_observability_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for triage.observability.metrics — OTel metrics integration.

Tests that need opentelemetry-sdk are skipped when it's not installed.
The no-op path (resolve_meter returns None, all record_* calls are silent)
runs unconditionally.
"""

from __future__ import annotations

import pytest

from triage.taxonomy import Step


# ── helpers ───────────────────────────────────────────────────────────────────

def make_step(index: int = 0, error: str | None = None) -> Step:
    return Step(index=index, action="test step", error=error)


def _otel_metrics_available() -> bool:
    try:
        from opentelemetry import metrics  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark_metrics = pytest.mark.skipif(
    not _otel_metrics_available(),
    reason="opentelemetry-sdk not installed",
)


# ── no-op path (OTel absent or no provider configured) ────────────────────────

def test_resolve_meter_returns_none_when_otel_absent(monkeypatch):
    import triage.observability.metrics as m
    monkeypatch.setattr(m, "_OTEL_METRICS_AVAILABLE", False)
    assert m.resolve_meter(None) is None


def test_resolve_meter_returns_explicit_without_provider(monkeypatch):
    import triage.observability.metrics as m
    monkeypatch.setattr(m, "_OTEL_METRICS_AVAILABLE", False)
    sentinel = object()
    assert m.resolve_meter(sentinel) is sentinel


def test_record_run_end_noop_on_none_meter():
    from triage.observability.metrics import record_run_end
    record_run_end(None, outcome="success", duration_s=1.5)
    record_run_end(None, outcome="error", duration_s=0.2)


def test_record_failure_noop_on_none_meter():
    from triage.observability.metrics import record_failure
    record_failure(None, failure_type="external_fault")


def test_record_recovery_noop_on_none_meter():
    from triage.observability.metrics import record_recovery
    record_recovery(None, failure_type="timeout", action_kind="retry")


def test_record_recovery_end_noop_on_none_meter():
    from triage.observability.metrics import record_recovery_end
    record_recovery_end(None, failure_type="unknown")


# ── agent integration without metrics ────────────────────────────────────────

async def test_agent_run_unaffected_without_meter():
    import triage
    from triage.policy import FailurePolicy

    async def my_agent(task: str, *, record_step, **kwargs) -> str:
        record_step(Step(index=0, action="work"))
        return f"done:{task}"

    agent = triage.Agent(my_agent, policy=FailurePolicy())
    result = await agent.run("hello")
    assert result == "done:hello"


# ── OTel metrics path (skipped without opentelemetry-sdk) ────────────────────

@pytestmark_metrics
def test_resolve_meter_returns_none_for_noop_provider():
    from triage.observability.metrics import resolve_meter
    # Without a configured MeterProvider, should return None
    result = resolve_meter(None)
    assert result is None


@pytestmark_metrics
def test_resolve_meter_returns_meter_when_provider_configured():
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from triage.observability.metrics import resolve_meter, _instrument_cache

    provider = MeterProvider()
    original = metrics.get_meter_provider()
    try:
        metrics.set_meter_provider(provider)
        result = resolve_meter(None)
        assert result is not None
    finally:
        metrics.set_meter_provider(original)
        # Clear the instrument cache so other tests get fresh instruments
        _instrument_cache.clear()


@pytestmark_metrics
async def test_agent_records_run_and_failure_metrics():
    """A failed + recovered run emits failures and recoveries metrics."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    import triage
    from triage.policy import FailurePolicy, RecoveryAction
    from triage.observability.metrics import _instrument_cache

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    original = metrics.get_meter_provider()
    try:
        metrics.set_meter_provider(provider)
        _instrument_cache.clear()

        calls: list[int] = []

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
        )
        result = await agent.run("task")
        assert result == "ok"

        data = reader.get_metrics_data()
        names = {
            m.name
            for rm in data.resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert "triage.runs" in names
        assert "triage.failures" in names
        assert "triage.recoveries" in names
        assert "triage.run.duration" in names

    finally:
        metrics.set_meter_provider(original)
        _instrument_cache.clear()


@pytestmark_metrics
async def test_run_counter_has_success_outcome_on_clean_run():
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    import triage
    from triage.policy import FailurePolicy
    from triage.observability.metrics import _instrument_cache

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    original = metrics.get_meter_provider()
    try:
        metrics.set_meter_provider(provider)
        _instrument_cache.clear()

        async def ok_agent(task: str, *, record_step, **kwargs) -> str:
            record_step(Step(index=0, action="work"))
            return "done"

        agent = triage.Agent(ok_agent, policy=FailurePolicy())
        await agent.run("t")

        data = reader.get_metrics_data()
        runs_points = [
            dp
            for rm in data.resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
            if m.name == "triage.runs"
            for dps in m.data.data_points
            for dp in [dps]
        ]
        outcomes = {dp.attributes.get("outcome") for dp in runs_points}
        assert "success" in outcomes

    finally:
        metrics.set_meter_provider(original)
        _instrument_cache.clear()
