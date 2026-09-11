"""
examples/otel_trajectory.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: build a Trajectory from OTel spans instead of hand-written
record_step() calls (v1.2).

Every other example in this directory has the wrapped agent construct
Step objects itself. That's a real cost: you have to instrument your
agent before triage can tell you anything. This example shows the
alternative — if your framework already emits OpenTelemetry spans for its
tool calls and errors (openllmetry/traceloop-style auto-instrumentation,
or the OTel GenAI semantic conventions directly), triage can build the
Trajectory from those spans instead, and record_step() becomes a thin
replay loop rather than something you write by hand.

Run with:
    pip install triage-agent[otel]
    python examples/otel_trajectory.py

What this shows:

  1. A stand-in for an "already-instrumented" framework — code you do NOT
     own or modify, e.g. a LangChain/LangGraph run, or any OpenAI-compatible
     client with openllmetry attached. It emits real OTel spans for its
     tool call and (on the first attempt) its failure — nothing here is
     triage-aware.
  2. A triage-wrapped agent that runs that framework unmodified, then reads
     back the spans it just emitted and converts them into Steps via
     trajectory_from_spans() — one line of glue, not a Step construction
     per tool call.
  3. A normal triage.Agent.run() with a recovery policy, showing the
     classification and recovery work correctly from OTel-derived Steps,
     exactly as if they'd been hand-written.
"""

from __future__ import annotations

import asyncio

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

import triage
from triage.observability.otel_ingest import trajectory_from_spans
from triage.strategies.retry import retry_with_tool_manifest

# ── stand-in for a framework triage does not control ────────────────────────
# This function is deliberately NOT triage-aware — no record_step, no Step
# import. It represents whatever your agent framework's own instrumentation
# already does. The only thing that matters here is that it's a real OTel
# span with real gen_ai.* attributes and a real exception recorded on error.

_attempt = [0]


async def already_instrumented_search(tracer, query: str) -> str:
    """A tool call your framework already wraps in an OTel span — you did
    not write this instrumentation, you just have to be able to read it."""
    _attempt[0] += 1
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "search",
            "gen_ai.tool.call.arguments": f'{{"query": "{query}"}}',
        },
    ) as span:
        if _attempt[0] == 1:
            exc = RuntimeError("tool 'search' not found in the provided tool definitions")
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise exc
        return f"3 results for {query!r}"


# ── the triage-wrapped agent — glue only, no manual Step construction ───────


def make_agent(tracer, exporter):
    async def agent(task: str, *, record_step, **_kwargs) -> str:
        exporter.clear()  # only spans from this attempt, not previous ones
        try:
            return await already_instrumented_search(tracer, task)
        finally:
            # This is the entire integration: read back what the framework's
            # own instrumentation already emitted, replay it as Steps.
            trajectory = trajectory_from_spans(exporter.get_finished_spans())
            for step in trajectory.steps:
                record_step(step)

    return agent


async def main() -> None:
    if not _OTEL_AVAILABLE:
        print("This example requires: pip install triage-agent[otel]")
        return

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("otel-trajectory-example")

    policy = triage.FailurePolicy(
        WRONG_TOOL_CALLED=retry_with_tool_manifest(),
        default=triage.FailurePolicy.escalate_by_default(),
    )
    agent = triage.Agent(
        make_agent(tracer, exporter),
        policy=policy,
        max_recovery_attempts=3,
    )

    print("=" * 65)
    print("Running an agent whose tool-call instrumentation triage never")
    print("touches — no Step objects are constructed by hand anywhere in")
    print("already_instrumented_search(). record_step() only replays spans")
    print("that already existed.")
    print("=" * 65)
    print()

    result = await agent.run("revenue Q1")
    print(f"Result: {result!r}")
    print()
    print("The first attempt's WRONG_TOOL_CALLED classification and the")
    print("retry that recovered from it were both driven entirely by")
    print("Step objects built from real OTel spans — see")
    print("triage/observability/otel_ingest.py for the conversion logic.")


if __name__ == "__main__":
    asyncio.run(main())
