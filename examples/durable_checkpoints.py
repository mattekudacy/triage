"""
examples/durable_checkpoints.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: durable SQLite checkpoints + auto_checkpoint + ROLLBACK recovery.

Run with:
    pip install "triage-agent[sqlite]"
    python examples/durable_checkpoints.py

What happens:
  A multi-phase agent runs three phases. After each phase it records a step.
  auto_checkpoint=True saves a checkpoint after every step.

  Phase 2 deliberately raises an error on the first attempt.
  triage classifies it as UNKNOWN (no pattern match), rolls back to the last
  checkpoint (end of phase 1), and re-runs from there.

  The SQLite database is written to a temp file so the example is self-contained
  and cleans up after itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(message)s")

import triage  # noqa: E402
from triage.strategies.rollback import rollback_to_checkpoint  # noqa: E402
from triage.taxonomy import Step  # noqa: E402

try:
    from triage.checkpoint.sqlite import SQLiteCheckpointStore
except ImportError:
    raise SystemExit(
        "Missing dependency. Run:\n"
        "  pip install 'triage-agent[sqlite]'"
    ) from None

# ── Synthetic multi-phase agent ───────────────────────────────────────────────

_attempt = [0]


async def multi_phase_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1
    print(f"\n  [agent] Attempt {_attempt[0]}")

    if _triage_hint:
        print(f"  [agent] Recovery hint: {_triage_hint!r}")

    # Phase 1 — always succeeds
    print("  [agent] Phase 1: fetching data...")
    record_step(Step(
        index=0,
        action="fetch_data",
        tool_called="fetch",
        tool_input={"url": "https://api.example.com/data"},
        tool_output='{"records": 42}',
    ))

    # Phase 2 — fails on first attempt with a hallucinated-state-style error
    print("  [agent] Phase 2: processing data...")
    if _attempt[0] == 1:
        record_step(Step(
            index=1,
            action="process_data",
            tool_called="process",
            error="AssertionError: agent claimed 100 records but tool returned 42",
            llm_output="I have processed all 100 records successfully.",
        ))
        raise RuntimeError(
            "AssertionError: agent claimed 100 records but tool returned 42"
        )

    # Phase 2 succeeds on retry (with hint)
    record_step(Step(
        index=1,
        action="process_data",
        tool_called="process",
        tool_input={"records": 42},
        tool_output="processed 42 records",
    ))

    # Phase 3
    print("  [agent] Phase 3: generating report...")
    record_step(Step(
        index=2,
        action="generate_report",
        tool_called="report",
        tool_input={"count": 42},
        tool_output="Report generated.",
    ))

    return f"Done. Processed 42 records for task: {task}"


# ── Wire up triage with SQLite checkpoints ────────────────────────────────────

async def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SQLiteCheckpointStore(db_path)
        print(f"\nCheckpoint DB: {db_path}")

        # Map UNKNOWN → rollback so the demo works with the default RulesClassifier.
        # For semantically ambiguous failures (PLAN_INCOMPLETE, CONTEXT_OVERFLOW),
        # swap in HybridClassifier — see examples/llm_classifier.py.
        policy = triage.FailurePolicy(
            UNKNOWN=rollback_to_checkpoint(),
            default=triage.FailurePolicy.escalate_by_default(),
        )

        agent = triage.Agent(
            multi_phase_agent,
            policy=policy,
            checkpoint_store=store,
            auto_checkpoint=True,
            max_recovery_attempts=3,
        )

        task = "Analyse Q1 sales data"
        print(f"\nTask: {task}\n")

        try:
            result = await agent.run(task)
            print(f"\n{result}")

            # Show what was persisted
            latest = await store.latest()
            if latest:
                print(f"\nLatest checkpoint: {latest.id}")
                print(f"  trajectory steps : {len(latest.trajectory_snapshot)}")
                print(f"  timestamp        : {latest.timestamp:.3f}")
        except triage.TriageEscalationError as exc:
            print(f"\nEscalated: {exc}")
        except triage.TriageAbortError as exc:
            print(f"\nAborted: {exc}")

    finally:
        os.unlink(db_path)
        print(f"\nCleaned up {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
