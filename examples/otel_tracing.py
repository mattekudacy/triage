"""
examples/otel_tracing.py
~~~~~~~~~~~~~~~~~~~~~~~~
Demo: OpenTelemetry span integration (v0.14).

Run with:
    pip install triage-agent[otel]
    python examples/otel_tracing.py

What this shows:

  triage emits three span types per run() call:
    - triage.run     — root span; attributes: triage.run_id, triage.task
    - triage.classify — wraps each failure classification; attribute: triage.failure_type
    - triage.dispatch — wraps each strategy dispatch; attributes: triage.action_kind,
                        triage.failure_type, triage.attempt

  All spans from one run() share the same trace_id and triage.run_id.
  Escalate/abort dispatches set the triage.dispatch span status to ERROR.

  Two modes:

  Auto-detect:  if opentelemetry-sdk is installed and a TracerProvider is
                configured, spans are emitted automatically — no Agent() change.

  Explicit:     pass tracer=trace.get_tracer("my-app") to Agent() to use a
                specific tracer regardless of the global provider.

  This example configures an in-memory exporter so you can see the spans
  printed to stdout without needing a collector.
"""

from __future__ import annotations

import asyncio

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

import triage
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

# ── Agent ─────────────────────────────────────────────────────────────────────

_attempt = [0]


async def flaky_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1
    attempt = _attempt[0]

    record_step(Step(
        index=0,
        action="fetch",
        tool_called="fetch_data",
        tool_input={"q": task},
        error="HTTP 503" if attempt == 1 else None,
        tool_output="data" if attempt > 1 else None,
    ))

    if attempt == 1:
        raise RuntimeError("HTTP 503 Service Unavailable")

    return f"Completed: {task}"


# ── Run ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not _OTEL_AVAILABLE:
        print(
            "opentelemetry-sdk is not installed.\n"
            "Install it with:  pip install triage-agent[otel]\n"
            "Then re-run this example."
        )
        return

    # Configure an in-memory exporter so spans are visible in this process
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=backoff_and_retry(max_attempts=2),
        default=triage.FailurePolicy.escalate_by_default(),
    )

    # --- Mode 1: auto-detect ---
    # Because we set a real TracerProvider above, triage picks it up automatically.
    print("── Mode 1: auto-detect (TracerProvider configured globally) ─────────────────\n")
    _attempt[0] = 0
    agent = triage.Agent(flaky_agent, policy=policy, max_recovery_attempts=3)
    result = await agent.run("analyse Q1 data")
    print(f"Result: {result}\n")

    _print_spans(exporter)
    exporter.clear()

    # --- Mode 2: explicit tracer ---
    # Pass tracer= directly; useful when you want a named tracer for filtering.
    print("── Mode 2: explicit tracer ──────────────────────────────────────────────────\n")
    _attempt[0] = 0
    my_tracer = trace.get_tracer("my-app")
    agent2 = triage.Agent(flaky_agent, policy=policy, max_recovery_attempts=3, tracer=my_tracer)
    result2 = await agent2.run("analyse Q2 data")
    print(f"Result: {result2}\n")

    _print_spans(exporter)
    exporter.clear()


def _print_spans(exporter: InMemorySpanExporter) -> None:
    spans = exporter.get_finished_spans()
    print(f"Spans emitted: {len(spans)}\n")
    for span in spans:
        attrs = dict(span.attributes or {})
        status = span.status.status_code.name
        run_id = attrs.get("triage.run_id", "")
        print(
            f"  {span.name:<22}"
            f"  status={status:<6}"
            f"  run_id={run_id[:8]}…"
            + (f"  failure_type={attrs['triage.failure_type']}"
               if "triage.failure_type" in attrs else "")
            + (f"  action_kind={attrs['triage.action_kind']}"
               if "triage.action_kind" in attrs else "")
        )
    print()


if __name__ == "__main__":
    asyncio.run(main())
