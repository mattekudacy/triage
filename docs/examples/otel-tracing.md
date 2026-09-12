# OpenTelemetry Tracing Example

Demo of the spans triage emits per `Agent.run()` call — auto-detected from a configured `TracerProvider`, or via an explicit `tracer=` passed to `Agent`.

**Source:** [`examples/otel_tracing.py`](https://github.com/mattekudacy/triage/blob/main/examples/otel_tracing.py)

## Requirements

```bash
pip install "triage-agent[otel]"
```

No collector needed — the example configures an in-memory exporter so spans print to stdout.

## What it demonstrates

triage emits three span types per `run()` call, all sharing one `trace_id` and a `triage.run_id` attribute:

| Span | Attributes | When |
|---|---|---|
| `triage.run` | `triage.run_id`, `triage.task` | Root span for the whole call |
| `triage.classify` | `triage.failure_type` | Wraps each failure classification |
| `triage.dispatch` | `triage.action_kind`, `triage.failure_type`, `triage.attempt` | Wraps each strategy dispatch; `ESCALATE`/`ABORT` set the span status to `ERROR` |

Two modes, both run in this example:

1. **Auto-detect** — with a real `TracerProvider` configured globally (as this example does), triage picks it up with no `Agent()` change.
2. **Explicit tracer** — pass `tracer=trace.get_tracer("my-app")` to `Agent()` to use a specific tracer regardless of the global provider, e.g. for per-component filtering.

See the README's Observability section for the full span/metric reference (`triage/observability/otel.py` and `metrics.py`), and the [OTel Trajectory example](otel-trajectory.md) for the inverse direction — building a `Trajectory` *from* spans a framework already emits, rather than emitting spans from triage's own recovery loop.

## Run

```bash
python examples/otel_tracing.py
```
