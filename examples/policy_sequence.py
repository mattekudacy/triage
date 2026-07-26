"""
examples/policy_sequence.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: FailurePolicy.sequence — ordered multi-step recovery.

Run with:
    python examples/policy_sequence.py

What this shows:

  FailurePolicy.sequence lets you step through an ordered list of strategies
  across successive failures of the same type. The first failure uses strategy
  0, the second uses strategy 1, and so on. Once all strategies are exhausted
  it escalates.

  Unlike chain() (which falls through within a single attempt), sequence()
  advances one step per failure — i.e. across attempts.

  Two scenarios run side by side:

  Scenario A — EXTERNAL_FAULT: retry → replan → escalate
    - Attempt 1: 503 error → sequence picks strategy 0: backoff_and_retry
    - Attempt 2: still failing → sequence picks strategy 1: replan
    - Attempt 3: agent reads hint, uses backup API, succeeds

  Scenario B — LOOP_DETECTED: replan → rollback → escalate
    - Attempt 1: agent loops → sequence picks strategy 0: replan(hint)
    - Attempt 2: still loops despite hint → sequence picks strategy 1: rollback
    - Attempt 3: checkpoint restored, agent continues from saved state, succeeds
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

# ── Scenario A: EXTERNAL_FAULT → retry → replan → (success) ──────────────────

_a_attempt = [0]


async def flaky_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _a_attempt[0] += 1
    attempt = _a_attempt[0]
    print(f"  [A] attempt {attempt}, hint={_triage_hint!r}")

    if _triage_hint and "backup" in _triage_hint.lower():
        record_step(Step(
            index=0,
            action="fetch_from_backup",
            tool_called="backup_api",
            tool_input={"q": task},
            tool_output="data from backup",
        ))
        return f"Completed via backup API: {task}"

    record_step(Step(
        index=0,
        action="fetch_from_primary",
        tool_called="primary_api",
        tool_input={"q": task},
        error="HTTP 503 Service Unavailable",
    ))
    raise RuntimeError("HTTP 503 Service Unavailable")


# ── Scenario B: LOOP_DETECTED → replan → rollback → (success) ────────────────

_b_attempt = [0]


async def looping_agent(
    task: str,
    *,
    record_step,
    update_state,
    _triage_hint: str | None = None,
    _triage_state: dict | None = None,
    **_kwargs,
) -> str:
    _b_attempt[0] += 1
    attempt = _b_attempt[0]
    print(f"  [B] attempt {attempt}, hint={_triage_hint!r}, "
          f"state_restored={bool(_triage_state)}")

    if _triage_state:
        # Rolled back — pick up from saved checkpoint
        print(f"  [B] resuming from checkpoint state: {_triage_state}")
        record_step(Step(
            index=2,
            action="finalize",
            tool_called="summarize",
            tool_input={"data": _triage_state.get("fetched")},
            tool_output="summary complete",
        ))
        return f"Completed from rollback checkpoint: {task}"

    # Phase 1: fetch succeeds — persist state so rollback has something to restore
    record_step(Step(
        index=0,
        action="fetch_data",
        tool_called="fetch",
        tool_input={"q": task},
        tool_output="raw data",
    ))
    update_state({"fetched": "raw data", "phase": "fetched"})

    # Phase 2: loop on the first two attempts regardless of hint
    if attempt <= 2:
        for i in range(3):
            record_step(Step(
                index=1 + i,
                action="analyze",
                tool_called="analyze",
                tool_input={"method": "default"},  # identical inputs → LOOP_DETECTED
            ))
        raise RuntimeError("Analysis stuck: same tool called 3 times with identical inputs")

    return f"Done: {task}"


# ── Wire up triage ────────────────────────────────────────────────────────────

# Scenario A: first failure → retry with backoff; second failure → replan to backup
fault_sequence = triage.FailurePolicy.sequence(
    backoff_and_retry(max_attempts=1),
    replan(hint="Primary API is unavailable. Switch to the backup API instead."),
)

# Scenario B: first loop → replan with different approach hint;
#             second loop → rollback to last clean checkpoint
loop_sequence = triage.FailurePolicy.sequence(
    replan(hint="You are repeating the same analysis. Try a different method.", max_replans=1),
    rollback_to_checkpoint(),
)

policy_a = triage.FailurePolicy(
    EXTERNAL_FAULT=fault_sequence,
    default=triage.FailurePolicy.escalate_by_default(),
)

policy_b = triage.FailurePolicy(
    LOOP_DETECTED=loop_sequence,
    default=triage.FailurePolicy.escalate_by_default(),
)

agent_a = triage.Agent(
    flaky_agent,
    policy=policy_a,
    max_recovery_attempts=4,
)

agent_b = triage.Agent(
    looping_agent,
    policy=policy_b,
    max_recovery_attempts=4,
    auto_checkpoint=True,  # saves after every record_step so rollback has a restore point
)


# ── Run ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n── Scenario A: EXTERNAL_FAULT → backoff_and_retry → replan ─────────────────\n")
    _a_attempt[0] = 0
    try:
        result = await agent_a.run("fetch market data")
        print(f"\n  Result: {result}")
    except triage.TriageEscalationError as exc:
        print(f"\n  Escalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\n  Aborted: {exc}")

    print("\n── Scenario B: LOOP_DETECTED → replan → rollback ───────────────────────────\n")
    _b_attempt[0] = 0
    try:
        result = await agent_b.run("analyse sales data")
        print(f"\n  Result: {result}")
    except triage.TriageEscalationError as exc:
        print(f"\n  Escalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\n  Aborted: {exc}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
