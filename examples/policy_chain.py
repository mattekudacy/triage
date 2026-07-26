"""
examples/policy_chain.py
~~~~~~~~~~~~~~~~~~~~~~~~
Demo: FailurePolicy.chain — cascading recovery strategies.

Run with:
    python examples/policy_chain.py

What this shows:

  FailurePolicy.chain lets you compose two strategies for the same failure
  type: try the primary first, fall through to the fallback when the primary
  exhausts itself.

  This example runs two scenarios side by side:

  Scenario A — LOOP_DETECTED: replan first, rollback if replan fails.
    - Attempt 1: agent loops → LOOP_DETECTED → replan (hint injected)
    - Attempt 2: agent loops again despite hint → LOOP_DETECTED → replan
      has exhausted its budget → chain falls through to rollback
    - Attempt 3: agent restores from checkpoint and succeeds

  Scenario B — EXTERNAL_FAULT: retry with backoff (×2), then replan.
    - Attempts 1-2: 503 errors → backoff_and_retry
    - Attempt 3: backoff_and_retry exhausted → chain falls through to replan
    - Attempt 4: agent uses alternative approach from hint → succeeds

  Also demonstrates auto_checkpoint=True writing a checkpoint after every
  step so rollback has a clean restore point.
"""

from __future__ import annotations

import asyncio
import logging

import triage
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.strategies.rollback import rollback_to_checkpoint
from triage.taxonomy import Step

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Scenario A: LOOP_DETECTED → replan → rollback ─────────────────────────────

_a_attempt = [0]


async def looping_agent(
    task: str,
    *,
    record_step,
    update_state,
    _triage_hint: str | None = None,
    _triage_state: dict | None = None,
    **_kwargs,
) -> str:
    _a_attempt[0] += 1
    attempt = _a_attempt[0]
    print(f"  [A] attempt {attempt}, hint={_triage_hint!r}, state_restored={bool(_triage_state)}")

    if _triage_state:
        # Restored from checkpoint — continue from where we left off
        print(f"  [A] resuming with state: {_triage_state}")
        record_step(
            Step(
                index=2,
                action="finalize",
                tool_called="summarize",
                tool_input={"data": _triage_state.get("fetched")},
                tool_output="summary complete",
            )
        )
        return f"Completed from checkpoint: {task}"

    # Phase 1: always succeeds — save state
    record_step(
        Step(
            index=0,
            action="fetch_data",
            tool_called="fetch",
            tool_input={"q": task},
            tool_output="raw data",
        )
    )
    update_state({"fetched": "raw data", "phase": "fetched"})

    # Phase 2: first two attempts loop (ignore hint), third attempt won't reach here
    if attempt <= 2:
        for i in range(3):
            record_step(
                Step(
                    index=1 + i,
                    action="analyze",
                    tool_called="analyze",
                    tool_input={"method": "default"},  # identical inputs → loop
                )
            )
        raise RuntimeError("Analysis stuck: same tool called 3 times with identical inputs")

    # Should not be reached given the rollback above
    return f"Done: {task}"


# ── Scenario B: EXTERNAL_FAULT → retry → replan ───────────────────────────────

_b_attempt = [0]


async def flaky_api_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _b_attempt[0] += 1
    attempt = _b_attempt[0]
    print(f"  [B] attempt {attempt}, hint={_triage_hint!r}")

    if _triage_hint and "alternative" in _triage_hint.lower():
        # Replan hint received — use a different data source
        record_step(
            Step(
                index=0,
                action="fetch_from_backup",
                tool_called="backup_api",
                tool_input={"q": task},
                tool_output="data from backup source",
            )
        )
        return f"Completed via backup API: {task}"

    # First two attempts: primary API is down
    record_step(
        Step(
            index=0,
            action="fetch_from_primary",
            tool_called="primary_api",
            tool_input={"q": task},
            error="HTTP 503 Service Unavailable",
        )
    )
    raise RuntimeError("HTTP 503 Service Unavailable")


# ── Wire up triage ────────────────────────────────────────────────────────────

# Scenario A: replan up to 1 time, then rollback
loop_strategy = triage.FailurePolicy.chain(
    replan(hint="You are repeating the same analysis. Try a different method.", max_replans=1),
    rollback_to_checkpoint(),
)

# Scenario B: retry with backoff up to 2 times, then replan to alternative
fault_strategy = triage.FailurePolicy.chain(
    backoff_and_retry(max_attempts=2),
    replan(hint="Primary API exhausted. Use the alternative data source instead."),
)

policy_a = triage.FailurePolicy(
    LOOP_DETECTED=loop_strategy,
    default=triage.FailurePolicy.escalate_by_default(),
)

policy_b = triage.FailurePolicy(
    EXTERNAL_FAULT=fault_strategy,
    default=triage.FailurePolicy.escalate_by_default(),
)

agent_a = triage.Agent(
    looping_agent,
    policy=policy_a,
    max_recovery_attempts=4,
    auto_checkpoint=True,  # saves after every record_step
)

agent_b = triage.Agent(
    flaky_api_agent,
    policy=policy_b,
    max_recovery_attempts=4,
)


# ── Run ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Scenario A
    print("\n── Scenario A: LOOP_DETECTED → replan → rollback ────────────────────────────\n")
    _a_attempt[0] = 0
    try:
        result = await agent_a.run("analyse sales data")
        print(f"\n  Result: {result}")
    except triage.TriageEscalationError as exc:
        print(f"\n  Escalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\n  Aborted: {exc}")

    # Scenario B
    print("\n── Scenario B: EXTERNAL_FAULT → retry (×2) → replan ────────────────────────\n")
    _b_attempt[0] = 0
    try:
        result = await agent_b.run("fetch market data")
        print(f"\n  Result: {result}")
    except triage.TriageEscalationError as exc:
        print(f"\n  Escalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\n  Aborted: {exc}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
