"""
triage.observability.otel
~~~~~~~~~~~~~~~~~~~~~~~~~
Optional OpenTelemetry integration.

Importing this module never fails — if ``opentelemetry`` is not installed,
``resolve_tracer()`` returns ``None`` and all span helpers become no-ops.

Usage in agent.py::

    from triage.observability.otel import resolve_tracer, run_span, classify_span, dispatch_span

    tracer = resolve_tracer(explicit_tracer)   # once per Agent.__init__
    async with run_span(tracer, run_id, task): ...
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

# ── lazy OTel import ──────────────────────────────────────────────────────────

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-not-found]
    from opentelemetry.trace import (  # type: ignore[import-not-found]
        NonRecordingSpan,
        Span,
        Status,
        StatusCode,
        Tracer,
    )
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    _otel_trace = None
    Tracer = Any
    Span = Any
    Status = None
    StatusCode = None
    NonRecordingSpan = None


def resolve_tracer(explicit: Any = None) -> Any:
    """Return a tracer to use for this Agent instance.

    Priority:
    1. ``explicit`` — whatever the caller passed as ``Agent(tracer=...)``.
    2. Auto-detect: if OTel is importable *and* a real (non-proxy, non-NoOp)
       tracer provider has been configured, return ``trace.get_tracer("triage")``.
    3. ``None`` — tracing is a no-op; all span helpers return a null context-manager.

    A "real" tracer provider means ``get_tracer_provider()`` is not the default
    ``ProxyTracerProvider`` / ``NoOpTracerProvider`` — i.e. the user has called
    ``trace.set_tracer_provider(...)`` with an actual SDK provider.
    """
    if explicit is not None:
        return explicit

    if not _OTEL_AVAILABLE:
        return None

    provider = _otel_trace.get_tracer_provider()
    provider_type = type(provider).__name__
    # Proxy and NoOp providers are the default — treat them as "not configured"
    if provider_type in ("ProxyTracerProvider", "NoOpTracerProvider"):
        return None

    return _otel_trace.get_tracer("triage")


# ── span helpers ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def run_span(
    tracer: Any,
    run_id: str | None,
    task: str,
) -> AsyncIterator[Any]:
    """Async context manager for the root ``triage.run`` span."""
    if tracer is None or not _OTEL_AVAILABLE:
        yield None
        return

    with tracer.start_as_current_span(
        "triage.run",
        attributes={
            "triage.run_id": run_id or "",
            "triage.task": task,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            _mark_error(span, exc)
            raise


@contextmanager
def classify_span(
    tracer: Any,
    run_id: str | None,
) -> Iterator[Any]:
    """Sync context manager for the ``triage.classify`` span."""
    if tracer is None or not _OTEL_AVAILABLE:
        yield None
        return

    with tracer.start_as_current_span(
        "triage.classify",
        attributes={"triage.run_id": run_id or ""},
    ) as span:
        try:
            yield span
        except Exception as exc:
            _mark_error(span, exc)
            raise


@contextmanager
def dispatch_span(
    tracer: Any,
    run_id: str | None,
    attempt: int,
) -> Iterator[Any]:
    """Sync context manager for the ``triage.dispatch`` span."""
    if tracer is None or not _OTEL_AVAILABLE:
        yield None
        return

    with tracer.start_as_current_span(
        "triage.dispatch",
        attributes={
            "triage.run_id": run_id or "",
            "triage.attempt": attempt,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            _mark_error(span, exc)
            raise


def set_span_classify_result(span: Any, failure_type_value: str) -> None:
    """Record the classification result as a span attribute + event."""
    if span is None or not _OTEL_AVAILABLE:
        return
    span.set_attribute("triage.failure_type", failure_type_value)
    span.add_event("triage.classified", {"failure_type": failure_type_value})


def set_span_dispatch_result(span: Any, action_kind: str, failure_type_value: str) -> None:
    """Record the dispatched action as a span attribute + event."""
    if span is None or not _OTEL_AVAILABLE:
        return
    span.set_attribute("triage.action_kind", action_kind)
    span.set_attribute("triage.failure_type", failure_type_value)
    span.add_event("triage.dispatched", {
        "action_kind": action_kind,
        "failure_type": failure_type_value,
    })
    if action_kind in ("escalate", "abort"):
        _mark_span_error(span, f"{action_kind}: {failure_type_value}")


def set_span_run_outcome(span: Any, *, error: Exception | None = None) -> None:
    """Mark the root run span as succeeded or failed."""
    if span is None or not _OTEL_AVAILABLE:
        return
    if error is not None:
        _mark_error(span, error)
    else:
        span.set_status(Status(StatusCode.OK))


# ── internals ─────────────────────────────────────────────────────────────────

def _mark_error(span: Any, exc: Exception) -> None:
    if span is None or not _OTEL_AVAILABLE:
        return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _mark_span_error(span: Any, description: str) -> None:
    if span is None or not _OTEL_AVAILABLE:
        return
    span.set_status(Status(StatusCode.ERROR, description))
