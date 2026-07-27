"""
scripts/bench_synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Mechanism demo (synthetic) — no API keys required.

This is a routing demonstration, not a real-world accuracy measurement.
The tasks are constructed so that the correct recovery hint actually changes
the outcome — a misclassification that delivers the wrong hint will still
fail — but the failure bodies are synthetic. Do not read the numbers as
evidence of production accuracy.

To measure classification accuracy against real traces, use the labeled
corpus in tests/test_classifier_rules.py or build a labeled JSONL dataset
and call RulesClassifier.fit() / LLMClassifier on it.

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_synthetic.py
"""

from __future__ import annotations

import asyncio

from triage.bench import run_benchmark
from triage.policy import FailurePolicy, RecoveryAction
from triage.strategies.retry import retry_with_tool_manifest
from triage.taxonomy import FailureContext, Step

# ── shared task logic ──────────────────────────────────────────────────────────
#
# Both arms call _task_body with the same task string.  Success depends on
# the hint being *correct for the failure type*:
#   - external_fault  → hint must contain "backoff" or "retry"
#   - wrong_tool      → hint must contain "manifest"
#   - schema_mismatch → hint must contain "schema"
#
# A misclassification that routes a schema error through backoff_and_retry
# delivers a hint like "External fault. Retry." — that hint does NOT contain
# "schema", so the task stays failed.  The number therefore measures routing
# correctness, not just hint presence.

_ATTEMPT: dict[str, int] = {}


def _task_body(task: str, attempt: int, hint: str) -> str:
    """Shared failure/success logic for both arms.

    First call always raises.  Subsequent calls succeed only if the hint
    matches the failure type that was injected.
    """
    kind = task.split(":")[0]

    if attempt == 0:
        if kind == "external_fault":
            raise RuntimeError("503 Service Unavailable")
        if kind == "wrong_tool":
            raise RuntimeError("Tool 'deprecated_tool' not found in manifest")
        if kind == "schema_mismatch":
            raise RuntimeError("JSON parse error: unexpected token")

    # Recovery attempt — hint must be correct for the failure type
    if kind == "external_fault" and ("retry" in hint or "backoff" in hint):
        return f"ok:{task}"
    if kind == "wrong_tool" and "manifest" in hint:
        return f"ok:{task}"
    if kind == "schema_mismatch" and "schema" in hint:
        return f"ok:{task}"

    # Wrong hint (misclassification) or no hint at all → still failing
    raise RuntimeError(f"hint {hint!r} does not resolve {kind}")


TASKS = [
    "external_fault:fetch_weather",
    "external_fault:call_payments",
    "wrong_tool:lookup_user",
    "schema_mismatch:parse_response",
    "external_fault:send_email",
    "wrong_tool:create_ticket",
]


# ── triage-wrapped agent ───────────────────────────────────────────────────────


async def triage_agent(task: str, *, record_step, **kwargs) -> str:
    hint = kwargs.get("_triage_hint", "")
    is_recovery = bool(hint)

    if not is_recovery:
        kind = task.split(":")[0]
        error_map = {
            "external_fault": "503 Service Unavailable",
            "wrong_tool": "Tool 'deprecated_tool' not found in manifest",
            "schema_mismatch": "JSON parse error: unexpected token",
        }
        error = error_map.get(kind, "unknown error")
        record_step(Step(index=0, action=task, error=error))
        raise RuntimeError(error)

    return _task_body(task, 1, hint)


# ── blind-retry baseline ───────────────────────────────────────────────────────
#
# Same task logic, no classification, no hint — always retries with hint="".
# Because the hint never matches, it will always fail on hint-sensitive tasks.

_baseline_call: dict[str, int] = {}


async def baseline_agent(task: str) -> str:
    _baseline_call[task] = _baseline_call.get(task, 0) + 1
    return _task_body(task, _baseline_call[task] - 1, hint="")


async def _schema_strategy(ctx: FailureContext) -> RecoveryAction:
    return RecoveryAction.RETRY(hint="Use strict JSON schema validation")


# ── main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    _baseline_call.clear()

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
        baseline_label="blind-retry",
    )

    print("── Mechanism demo (synthetic) ───────────────────────────────────────")
    print("NOTE: constructed scenarios demonstrating correct routing, not")
    print("evidence of real-world classification accuracy.")
    print()
    print(report.summary())
    print()
    print(report.compare())
    print()
    print("Failure type breakdown (triage arm):")
    for ft, n in sorted(report.failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {n}")


if __name__ == "__main__":
    asyncio.run(main())
