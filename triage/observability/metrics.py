"""
triage.observability.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional OpenTelemetry metrics integration.

Importing this module never fails — if ``opentelemetry`` is not installed,
all helpers are silent no-ops with zero overhead.

Instruments
-----------
triage.runs            Counter   — incremented once per Agent.run() call;
                                   attributes: outcome ("success"|"error")
triage.failures        Counter   — incremented on every classified failure;
                                   attributes: failure_type
triage.recoveries      Counter   — incremented on every strategy dispatch;
                                   attributes: failure_type, action_kind
triage.run.duration    Histogram — wall-clock seconds for each Agent.run();
                                   attributes: outcome
triage.recovery.attempts UpDownCounter — net count of in-progress recovery
                                   loops; incremented on first failure,
                                   decremented when the run exits

Usage::

    from triage.observability.metrics import resolve_meter

    meter = resolve_meter(explicit_meter)   # once per Agent.__init__
    # then pass `meter` to record_* helpers throughout _run_loop
"""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.metrics import Meter

    _OTEL_METRICS_AVAILABLE = True
except ImportError:
    _OTEL_METRICS_AVAILABLE = False
    _otel_metrics = None
    Meter = Any


# ── meter resolution ──────────────────────────────────────────────────────────


def resolve_meter(explicit: Any = None) -> Any:
    """Return a Meter to use for this Agent instance.

    Priority:
    1. ``explicit`` — whatever the caller passed as ``Agent(meter=...)``.
    2. Auto-detect: if OTel metrics is importable *and* a non-default
       MeterProvider has been configured (i.e. not the NoOp/Proxy default),
       return ``metrics.get_meter("triage")``.
    3. ``None`` — metrics are a no-op; all helpers are silent.
    """
    if explicit is not None:
        return explicit

    if not _OTEL_METRICS_AVAILABLE:
        return None

    provider = _otel_metrics.get_meter_provider()
    provider_type = type(provider).__name__
    if provider_type in ("NoOpMeterProvider", "ProxyMeterProvider", "_ProxyMeterProvider"):
        return None

    return _otel_metrics.get_meter("triage")


# ── instrument cache ──────────────────────────────────────────────────────────


class _Instruments:
    """Lazily-built instruments for a given Meter instance."""

    def __init__(self, meter: Any) -> None:
        self.runs = meter.create_counter(
            "triage.runs",
            description="Total Agent.run() calls",
        )
        self.failures = meter.create_counter(
            "triage.failures",
            description="Classified failures inside Agent.run()",
        )
        self.recoveries = meter.create_counter(
            "triage.recoveries",
            description="Strategy dispatches (recovery attempts)",
        )
        self.run_duration = meter.create_histogram(
            "triage.run.duration",
            unit="s",
            description="Wall-clock seconds per Agent.run() call",
        )
        self.recovery_attempts = meter.create_up_down_counter(
            "triage.recovery.attempts",
            description="In-progress recovery loops",
        )


# Keyed on id(meter). This assumes OTel meters are long-lived (created once at
# provider setup, never replaced or garbage-collected during the process lifetime),
# which is the standard OTel usage pattern. If a meter were GC'd its id could be
# reused by an unrelated object, silently returning stale instruments for the new
# object — but that cannot happen in practice because the MeterProvider holds a
# strong reference to every meter it creates. No eviction is therefore needed.
_instrument_cache: dict[int, _Instruments] = {}


def _get_instruments(meter: Any) -> _Instruments | None:
    if meter is None or not _OTEL_METRICS_AVAILABLE:
        return None
    key = id(meter)
    if key not in _instrument_cache:
        _instrument_cache[key] = _Instruments(meter)
    return _instrument_cache[key]


# ── record helpers ────────────────────────────────────────────────────────────


def record_run_start(meter: Any) -> None:
    """Called at the very start of Agent.run() — no attributes yet."""
    # Nothing to record on start; duration is recorded on finish.


def record_run_end(meter: Any, *, outcome: str, duration_s: float) -> None:
    """Record run completion. ``outcome`` is ``"success"``, ``"error"``, or ``"cancelled"``."""
    inst = _get_instruments(meter)
    if inst is None:
        return
    attrs = {"outcome": outcome}
    inst.runs.add(1, attrs)
    inst.run_duration.record(duration_s, attrs)


def record_failure(meter: Any, *, failure_type: str) -> None:
    """Record a classified failure."""
    inst = _get_instruments(meter)
    if inst is None:
        return
    inst.failures.add(1, {"failure_type": failure_type})
    inst.recovery_attempts.add(1, {"failure_type": failure_type})


def record_recovery(meter: Any, *, failure_type: str, action_kind: str) -> None:
    """Record a strategy dispatch."""
    inst = _get_instruments(meter)
    if inst is None:
        return
    inst.recoveries.add(1, {"failure_type": failure_type, "action_kind": action_kind})


def record_recovery_end(meter: Any, *, failure_type: str) -> None:
    """Decrement the in-progress recovery counter when a run exits."""
    inst = _get_instruments(meter)
    if inst is None:
        return
    inst.recovery_attempts.add(-1, {"failure_type": failure_type})
