"""
examples/failure_distribution.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: aggregate triage OTel spans into a failure-type distribution report (v0.18).

Run with:
    pip install triage-agent[otel]
    python examples/failure_distribution.py

What this shows:

  triage emits a ``triage.classify`` span for every classified failure, carrying
  ``triage.failure_type`` as an attribute.  A ``triage.dispatch`` span follows each
  classification, carrying ``triage.action_kind`` and ``triage.failure_type``.

  This example:
    1. Runs 20 synthetic agent calls across four failure types.
    2. Reads the finished spans from an in-memory exporter.
    3. Produces a frequency table: input counts, failure-type mix, recovery rate.

  No new library code is needed — everything uses the existing OTel span attributes.

Reading the output columns:
    FAILURE TYPE   — the classified FailureType (or "success" for clean runs)
    COUNT          — how many runs hit this type
    %              — share of total runs
    RECOVERED      — runs where the strategy succeeded (action != escalate/abort)
    RECOVERY RATE  — RECOVERED / COUNT
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

import triage
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

# ── Synthetic agents ───────────────────────────────────────────────────────────
# Each agent raises a specific error on its first call, then succeeds.

def _make_agent(error_msg: str, *, succeed_after: int = 1):
    """Return an agent that fails ``succeed_after`` times then succeeds."""
    attempt = [0]

    async def _agent(task: str, *, record_step, **_kwargs) -> str:
        attempt[0] += 1
        n = attempt[0]

        step = Step(
            index=0,
            action="do_work",
            tool_called="tool",
            tool_input={"task": task},
            error=error_msg if n <= succeed_after else None,
        )
        record_step(step)

        if n <= succeed_after:
            raise RuntimeError(error_msg)
        return f"ok:{task}"

    return _agent


def _make_always_failing_agent(error_msg: str):
    """Return an agent that always fails (forces escalation)."""
    async def _agent(task: str, *, record_step, **_kwargs) -> str:
        record_step(Step(index=0, action="do_work", error=error_msg))
        raise RuntimeError(error_msg)

    return _agent


# ── Run population ─────────────────────────────────────────────────────────────

async def _run_population(
    exporter: InMemorySpanExporter,
) -> tuple[int, int]:
    """
    Execute 20 synthetic runs and return (total_runs, successful_runs).

    Failure mix injected:
        5 × EXTERNAL_FAULT  — backoff_and_retry recovers all 5
        5 × WRONG_TOOL      — replan recovers all 5
        5 × SCHEMA_MISMATCH — backoff_and_retry, but capped at 1 attempt → 5 escalate
        5 × success         — clean runs, no classification span emitted
    """
    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
        WRONG_TOOL_CALLED=replan(),
        SCHEMA_MISMATCH=backoff_and_retry(max_attempts=1),
        default=triage.FailurePolicy.escalate_by_default(),
    )

    total = 0
    successes = 0

    # ── external fault — recovers ──────────────────────────────────────────────
    for i in range(5):
        agent = triage.Agent(
            _make_agent("HTTP 503 Service Unavailable"),
            policy=policy,
            max_recovery_attempts=3,
        )
        try:
            await agent.run(f"ext-task-{i}")
            successes += 1
        except (triage.TriageEscalationError, triage.TriageAbortError):
            pass
        total += 1

    # ── wrong tool — recovers ──────────────────────────────────────────────────
    for i in range(5):
        agent = triage.Agent(
            _make_agent("no tool named missing_tool"),
            policy=policy,
            max_recovery_attempts=3,
        )
        try:
            await agent.run(f"tool-task-{i}")
            successes += 1
        except (triage.TriageEscalationError, triage.TriageAbortError):
            pass
        total += 1

    # ── schema mismatch — escalates (max_attempts=1 means one retry only) ──────
    for i in range(5):
        agent = triage.Agent(
            _make_always_failing_agent("JSONDecodeError: Expecting value at line 1 column 1"),
            policy=policy,
            max_recovery_attempts=1,
        )
        try:
            await agent.run(f"schema-task-{i}")
            successes += 1
        except (triage.TriageEscalationError, triage.TriageAbortError):
            pass
        total += 1

    # ── clean runs — no failure span ───────────────────────────────────────────
    async def _clean_agent(task: str, *, record_step, **_kwargs) -> str:
        record_step(Step(index=0, action="do_work"))
        return f"ok:{task}"

    for i in range(5):
        agent = triage.Agent(_clean_agent, policy=policy)
        try:
            await agent.run(f"clean-task-{i}")
            successes += 1
        except (triage.TriageEscalationError, triage.TriageAbortError):
            pass
        total += 1

    return total, successes


# ── Report builder ─────────────────────────────────────────────────────────────

def _build_report(
    spans: list,
    total_runs: int,
    successful_runs: int,
) -> None:
    """Aggregate span attributes into a failure-distribution report."""

    # Index spans by run_id
    # run_id → list of classified failure_type strings
    run_failures: dict[str, list[str]] = defaultdict(list)
    # run_id → True if the triage.run span ended with ERROR status
    run_errored: dict[str, bool] = {}

    for span in spans:
        attrs = dict(span.attributes or {})
        run_id = attrs.get("triage.run_id", "")
        if span.name == "triage.classify":
            ft = attrs.get("triage.failure_type")
            if ft:
                run_failures[run_id].append(ft)
        elif span.name == "triage.run":
            # status_code is an enum; .name is "OK", "ERROR", or "UNSET"
            errored = span.status.status_code.name == "ERROR"
            run_errored[run_id] = errored

    # Per-failure-type aggregation
    # failure_type → {"runs": int, "recovered": int}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "recovered": 0})

    all_failing_run_ids = set(run_failures.keys())

    for run_id in all_failing_run_ids:
        # Use the last classified failure type as the label for the run.
        # Runs with multiple attempts may classify the same type several times.
        failure_types = run_failures[run_id]
        primary = failure_types[-1] if failure_types else "unknown"
        stats[primary]["runs"] += 1

        # A run "recovered" if its root triage.run span did NOT end with ERROR.
        # This covers both explicit escalate/abort dispatches and the internal
        # max_recovery_attempts / max_total_attempts exceeded paths.
        if not run_errored.get(run_id, True):
            stats[primary]["recovered"] += 1

    # Clean runs (no classify span)
    clean_count = total_runs - len(all_failing_run_ids)

    # ── Print report ──────────────────────────────────────────────────────────
    col_ft = 26
    col_count = 7
    col_pct = 7
    col_rec = 11
    col_rate = 15

    sep = "-" * (col_ft + col_count + col_pct + col_rec + col_rate + 10)
    header = (
        f"{'FAILURE TYPE':<{col_ft}}"
        f"  {'COUNT':>{col_count}}"
        f"  {'%':>{col_pct}}"
        f"  {'RECOVERED':>{col_rec}}"
        f"  {'RECOVERY RATE':>{col_rate}}"
    )

    print()
    print("── Failure distribution ─────────────────────────────────────────────────────")
    print()
    print(f"  Total runs   : {total_runs}")
    print(f"  Successful   : {successful_runs}  ({100 * successful_runs / total_runs:.0f}%)")
    print(f"  With failures: {total_runs - successful_runs}")
    print()
    print(f"  {header}")
    print(f"  {sep}")

    rows = sorted(stats.items(), key=lambda x: x[1]["runs"], reverse=True)

    for ft, s in rows:
        count = s["runs"]
        recovered = s["recovered"]
        pct = 100 * count / total_runs
        rate = f"{100 * recovered // count}%" if count else "—"
        rec_str = f"{recovered}/{count}"
        bar = "█" * recovered + "░" * (count - recovered)
        print(
            f"  {ft:<{col_ft}}"
            f"  {count:>{col_count}}"
            f"  {pct:>{col_pct}.0f}%"
            f"  {rec_str:>{col_rec}}"
            f"  {rate:>{col_rate}}  {bar}"
        )

    if clean_count:
        print(
            f"  {'(no failure)':<{col_ft}}"
            f"  {clean_count:>{col_count}}"
            f"  {100 * clean_count / total_runs:>{col_pct}.0f}%"
            f"  {'—':>{col_rec}}"
            f"  {'—':>{col_rate}}"
        )

    print(f"  {sep}")
    print()

    # ── Span inventory ────────────────────────────────────────────────────────
    name_counts: dict[str, int] = defaultdict(int)
    for span in spans:
        name_counts[span.name] += 1

    print("── Span inventory ───────────────────────────────────────────────────────────")
    print()
    for name, count in sorted(name_counts.items()):
        print(f"  {name:<24}  {count:>4} spans")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not _OTEL_AVAILABLE:
        print(
            "opentelemetry-sdk is not installed.\n"
            "Install it with:  pip install triage-agent[otel]\n"
            "Then re-run this example."
        )
        return

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    print("Running 20 synthetic agent calls …")
    total_runs, successful_runs = await _run_population(exporter)

    spans = exporter.get_finished_spans()
    _build_report(spans, total_runs, successful_runs)


if __name__ == "__main__":
    asyncio.run(main())
