"""
scripts/bench_synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthetic benchmark — no API keys required.

Simulates three failure modes (transient external fault, wrong tool, schema
mismatch) and compares triage recovery against a blind-retry baseline.

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_synthetic.py
"""

from __future__ import annotations

import asyncio
import random

from triage.bench import run_benchmark
from triage.policy import FailurePolicy, RecoveryAction
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.taxonomy import FailureContext, Step

# ── synthetic tasks ────────────────────────────────────────────────────────────

TASKS = [
    "external_fault:fetch_weather",    # recoverable with backoff
    "external_fault:call_payments",    # recoverable with backoff
    "wrong_tool:lookup_user",          # recoverable with hint
    "schema_mismatch:parse_response",  # recoverable with hint
    "external_fault:send_email",       # recoverable with backoff
    "wrong_tool:create_ticket",        # recoverable with hint
]


# ── triage-wrapped agent ───────────────────────────────────────────────────────

async def triage_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    hint = kwargs.get("_triage_hint", "")
    kind = task.split(":")[0]

    if kind == "external_fault":
        # First attempt always fails; hint (from backoff_and_retry) means retry
        if not hint:
            record_step(Step(index=0, action=task, error="503 Service Unavailable"))
            raise RuntimeError("503 Service Unavailable")
        return f"ok:{task}"

    if kind == "wrong_tool":
        if not hint:
            record_step(Step(index=0, action=task, tool_called="deprecated_tool",
                             error="Tool 'deprecated_tool' not found in manifest"))
            raise RuntimeError("Tool 'deprecated_tool' not found in manifest")
        return f"ok:{task}"

    if kind == "schema_mismatch":
        if not hint:
            record_step(Step(index=0, action=task, error="JSON parse error: unexpected token"))
            raise RuntimeError("JSON parse error: unexpected token")
        return f"ok:{task}"

    return f"ok:{task}"


# ── blind-retry baseline ───────────────────────────────────────────────────────

_baseline_attempts: dict[str, int] = {}


async def baseline_agent(task: str) -> str:
    """Always retries blindly up to 3 times with no classification."""
    key = task
    _baseline_attempts[key] = _baseline_attempts.get(key, 0) + 1
    kind = task.split(":")[0]

    # Baseline has a 33% chance of succeeding blind on each attempt after the first
    if _baseline_attempts[key] == 1 and kind != "schema_mismatch":
        raise RuntimeError("first attempt fails")
    if kind == "schema_mismatch" and _baseline_attempts[key] < 3:
        raise RuntimeError("schema errors don't self-heal")
    return f"ok:{task}"


async def _schema_strategy(ctx: FailureContext) -> RecoveryAction:
    return RecoveryAction.RETRY(hint="Use strict JSON schema")


# ── main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    random.seed(42)
    _baseline_attempts.clear()

    # delay=0 keeps the bench fast and avoids skewing the latency comparison
    async def backoff_nodelay(ctx: FailureContext) -> RecoveryAction:
        prior = sum(1 for _, k in ctx.attempt_history if k == "retry")
        if prior >= 2:
            return RecoveryAction.ESCALATE("max attempts reached")
        return RecoveryAction.RETRY(hint="External fault. Retry.", delay=0.0)

    policy = FailurePolicy(
        EXTERNAL_FAULT=backoff_nodelay,
        WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=2),
        SCHEMA_MISMATCH=_schema_strategy,
    )

    report = await run_benchmark(
        triage_agent,
        tasks=TASKS,
        policy=policy,
        n_runs=5,
        label="triage",
        max_recovery_attempts=3,
        baseline_fn=baseline_agent,
        baseline_label="blind-retry",
    )

    print(report.summary())
    print()
    print(report.compare())
    print()

    # Emit the Markdown table for the README
    bsr = report._baseline_success_rate
    tsr = report.success_rate
    bml = report._baseline_mean_latency_s
    tml = report.mean_latency_s
    rec = report.total_recoveries

    print("Markdown table:")
    print()
    print("| Scenario | blind-retry success | triage success | triage recoveries | mean latency |")
    print("|---|---|---|---|---|")
    print(f"| 6 tasks × 5 runs (mixed failure modes) | {bsr:.0%} | {tsr:.0%} | {rec} | {tml:.3f}s |")
    print()
    print("Failure type breakdown:")
    for ft, n in sorted(report.failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {n}")


if __name__ == "__main__":
    asyncio.run(main())
