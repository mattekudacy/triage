"""
scripts/bench_synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Mechanism demo (synthetic) — no API keys required.

This is a routing demonstration, not a real-world accuracy measurement.
The tasks are constructed so that the correct recovery hint changes the
outcome — a misclassification that delivers the wrong hint still fails.
Do not read the numbers as evidence of production accuracy.

Design
------
Both arms (triage + no-recovery baseline) call the same _task_body(),
which ensures the only variable between them is triage's classification
and routing.

  * external_fault: transient — any retry heals it, hint irrelevant.
  * wrong_tool:     requires "manifest" in the hint to succeed.
  * schema_mismatch: requires "schema" in the hint to succeed.

The no-recovery baseline retries up to 3 times with no hint, so it
recovers the transient faults (~50%) but not the routing-sensitive ones.
The gap is attributable precisely to the two types where classification
matters. Conceding the transient case makes the comparison honest.

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_synthetic.py
"""

from __future__ import annotations

import asyncio

from triage.bench import run_benchmark
from triage.policy import FailurePolicy, RecoveryAction
from triage.strategies.retry import retry_with_tool_manifest
from triage.taxonomy import FailureContext, Step

# ── shared failure logic ───────────────────────────────────────────────────────

TASKS = [
    "external_fault:fetch_weather",
    "external_fault:call_payments",
    "wrong_tool:lookup_user",
    "schema_mismatch:parse_response",
    "external_fault:send_email",
    "wrong_tool:create_ticket",
]


def _task_body(task: str, attempt: int, hint: str) -> str:
    """Shared success/failure logic used by both arms.

    First attempt always raises.  Recovery succeeds only when the hint
    matches the failure type:
    - external_fault:  any retry heals it (transient by definition)
    - wrong_tool:      hint must contain "manifest"
    - schema_mismatch: hint must contain "schema"
    """
    kind = task.split(":")[0]

    if attempt == 0:
        error_map = {
            "external_fault": "503 Service Unavailable",
            "wrong_tool": "Tool 'deprecated_tool' not found in manifest",
            "schema_mismatch": "JSON parse error: unexpected token",
        }
        raise RuntimeError(error_map.get(kind, "unknown error"))

    if kind == "external_fault":
        return f"ok:{task}"  # transient — heals on any retry, no hint required
    if kind == "wrong_tool" and "manifest" in hint:
        return f"ok:{task}"
    if kind == "schema_mismatch" and "schema" in hint:
        return f"ok:{task}"

    raise RuntimeError(f"hint {hint!r} does not resolve {kind}")


# ── triage-wrapped agent ───────────────────────────────────────────────────────


async def triage_agent(task: str, *, record_step, **kwargs) -> str:
    hint = kwargs.get("_triage_hint", "")
    attempt = 0 if not hint else 1
    if attempt == 0:
        kind = task.split(":")[0]
        errors = {
            "external_fault": "503 Service Unavailable",
            "wrong_tool": "Tool 'deprecated_tool' not found in manifest",
            "schema_mismatch": "JSON parse error: unexpected token",
        }
        record_step(Step(index=0, action=task, error=errors.get(kind, "")))
    return _task_body(task, attempt, hint)


# ── no-recovery baseline ───────────────────────────────────────────────────────
#
# Retries up to 3 times with no hint. Recovers transient faults (external_fault),
# cannot recover routing-sensitive failures (wrong_tool, schema_mismatch).


async def baseline_agent(task: str) -> str:
    for attempt in range(3):
        try:
            return _task_body(task, attempt, hint="")
        except RuntimeError:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


# ── strategies ─────────────────────────────────────────────────────────────────


async def _schema_strategy(ctx: FailureContext) -> RecoveryAction:
    return RecoveryAction.RETRY(hint="Use strict JSON schema validation")


# ── main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    async def backoff_nodelay(ctx: FailureContext) -> RecoveryAction:
        prior = sum(1 for _, k in ctx.attempt_history if k == "retry")
        if prior >= 2:
            return RecoveryAction.ESCALATE("max attempts reached")
        return RecoveryAction.RETRY(hint="External fault. Retry with backoff.", delay=0.0)

    policy = FailurePolicy(
        EXTERNAL_FAULT=backoff_nodelay,
        WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=2),
        SCHEMA_MISMATCH=_schema_strategy,
    )

    report = await run_benchmark(
        triage_agent,
        tasks=TASKS,
        policy=policy,
        n_runs=1,
        label="triage",
        max_recovery_attempts=3,
        baseline_fn=baseline_agent,
        baseline_label="no-recovery",
    )

    print("── Routing demo (synthetic) ─────────────────────────────────────────")
    print("NOTE: constructed scenarios showing routing correctness, not")
    print("evidence of real-world classification accuracy.")
    print()
    print(report.summary())
    print()

    bsr = report._baseline_success_rate
    tsr = report.success_rate
    rec = report.total_recoveries
    print(f"{'':22} no-recovery    triage")
    print(f"{'success rate:':<22} {bsr:.0%}            {tsr:.0%}")
    print(f"{'recoveries:':<22} —               {rec}")
    print()
    print("Failure type breakdown (triage arm):")
    for ft, n in sorted(report.failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {n}")
    print()
    print("Gap: triage beats no-recovery only on wrong_tool and schema_mismatch,")
    print("where classification delivers the hint that changes the outcome.")
    print("external_fault heals on any retry — no classification needed.")


if __name__ == "__main__":
    asyncio.run(main())
