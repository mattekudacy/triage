"""
examples/risk_guard.py
~~~~~~~~~~~~~~~~~~~~~~
Demo: RulesRiskScorer + strict_idempotency + lifecycle hooks.

Run with:
    python examples/risk_guard.py

What this shows:

  1. RulesRiskScorer intercepts a destructive step (send_email) before it
     executes. triage aborts the run instead of letting the agent fire an
     irreversible action.

  2. strict_idempotency=True prevents a retry after any non-idempotent step
     has already executed — guarantees you never charge a card or send a
     message twice by accident.

  3. Lifecycle hooks (on_step, on_failure, on_recovery) let you build an
     audit trail without touching agent code.

Two runs are demonstrated:

  Run A — agent tries to send an email mid-loop.
           RulesRiskScorer catches it (score 0.95 >= threshold 0.9) and
           raises TriageAbortError before the action executes.

  Run B — agent executes a non-idempotent payment step, then fails.
           strict_idempotency=True blocks the retry and escalates to human
           review rather than charging the card twice.
"""

from __future__ import annotations

import asyncio
import logging

import triage
from triage.policy import RecoveryAction
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import FailureContext, Step

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Audit log (built from lifecycle hooks) ────────────────────────────────────

audit_log: list[str] = []


def on_step(step: Step) -> None:
    audit_log.append(f"STEP  [{step.index}] {step.action}")


def on_failure(ctx: FailureContext) -> None:
    audit_log.append(f"FAIL  {ctx.failure_type.value} at step {ctx.critical_step_index}")


def on_recovery(ctx: FailureContext, action: RecoveryAction) -> None:
    audit_log.append(f"RECV  {action.kind} for {ctx.failure_type.value}")


# ── Run A: agent tries to send an email ──────────────────────────────────────


async def email_agent(task: str, *, record_step, **_kwargs) -> str:
    # Safe step
    record_step(
        Step(
            index=0,
            action="fetch_data",
            tool_called="fetch",
            tool_input={"url": "https://api.example.com/report"},
            tool_output="report data",
        )
    )

    # Destructive step — RulesRiskScorer fires here, score 0.95
    record_step(
        Step(
            index=1,
            action="send_email",  # matches high-risk pattern
            tool_called="send_email",
            tool_input={"to": "ceo@corp.com", "body": "Quarterly report attached."},
            idempotent=False,
        )
    )

    # We never reach this line — TriageAbortError raised during record_step above
    return "done"


# ── Run B: agent charges card, then fails ─────────────────────────────────────

_run_b_attempt = [0]


async def payment_agent(task: str, *, record_step, **_kwargs) -> str:
    _run_b_attempt[0] += 1

    # Non-idempotent step — money has moved
    record_step(
        Step(
            index=0,
            action="charge_card",
            tool_called="charge_card",
            tool_input={"amount": 99.00, "currency": "USD"},
            tool_output="charge_id: chg_abc123",
            idempotent=False,  # explicitly marked non-idempotent
        )
    )

    # Confirmation endpoint fails — record the 503 so the classifier can see it
    record_step(
        Step(
            index=1,
            action="confirm_charge",
            tool_called="confirm_api",
            tool_input={"charge_id": "chg_abc123"},
            error="HTTP 503 Service Unavailable — confirmation endpoint down",
        )
    )
    raise RuntimeError("HTTP 503 Service Unavailable — confirmation endpoint down")


# ── Wire up triage ────────────────────────────────────────────────────────────

scorer = triage.RulesRiskScorer()

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    LOOP_DETECTED=replan(),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent_a = triage.Agent(
    email_agent,
    policy=policy,
    risk_scorer=scorer,
    risk_threshold=0.9,
    on_step=on_step,
    on_failure=on_failure,
    on_recovery=on_recovery,
)

agent_b = triage.Agent(
    payment_agent,
    policy=triage.FailurePolicy(
        EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),  # would retry, but idempotency blocks it
        default=triage.FailurePolicy.escalate_by_default(),
    ),
    strict_idempotency=True,  # never retry after a non-idempotent step
    on_step=on_step,
    on_failure=on_failure,
    on_recovery=on_recovery,
)


async def main() -> None:
    # ── Run A ─────────────────────────────────────────────────────────────────
    print("\n── Run A: destructive step interception ─────────────────────────────────────\n")
    audit_log.clear()
    try:
        await agent_a.run("send quarterly report")
        print("  [unexpected] agent completed — abort did not fire")
    except triage.TriageAbortError as exc:
        print(f"  Aborted: {exc}")
        print("  (email was never sent)")

    print("\n  Audit trail:")
    for entry in audit_log:
        print(f"    {entry}")

    # ── Run B ─────────────────────────────────────────────────────────────────
    print("\n── Run B: strict idempotency blocks retry after payment ─────────────────────\n")
    audit_log.clear()
    try:
        await agent_b.run("charge customer for subscription")
        print("  [unexpected] agent completed")
    except triage.TriageEscalationError as exc:
        print(f"  Escalated: {exc}")
        print("  (card was charged once; retry blocked to prevent double-charge)")

    print("\n  Audit trail:")
    for entry in audit_log:
        print(f"    {entry}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
