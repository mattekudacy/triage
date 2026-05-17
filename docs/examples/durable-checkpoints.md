# Durable Checkpoints Example

Demo: multi-phase agent with SQLite checkpoints and `ROLLBACK` recovery. Shows how state is preserved across failures.

**Source:** [`examples/durable_checkpoints.py`](https://github.com/mattekudacy/triage/blob/main/examples/durable_checkpoints.py)

## Requirements

```bash
pip install "triage-agent[sqlite]"
```

No API key needed — uses `RulesClassifier` and maps `UNKNOWN` to rollback for the demo.

## What it demonstrates

1. A 3-phase agent runs with `auto_checkpoint=True` — a checkpoint is saved after each `record_step()` call.
2. Phase 2 deliberately fails on attempt 1, simulating a hallucinated-state error.
3. triage rolls back to the last checkpoint (end of phase 1) and re-runs from there with a corrective hint.
4. Phase 2 succeeds on attempt 2; phase 3 completes normally.
5. The SQLite database is written to a temp file and cleaned up after the run.

## Code

```python
import asyncio
import os
import tempfile
import triage
from triage.checkpoint.sqlite import SQLiteCheckpointStore
from triage.strategies.rollback import rollback_to_checkpoint
from triage.taxonomy import Step

_attempt = [0]

async def multi_phase_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    _attempt[0] += 1

    # Phase 1 — always succeeds
    record_step(Step(
        index=0, action="fetch_data", tool_called="fetch",
        tool_output='{"records": 42}',
    ))

    # Phase 2 — fails on first attempt
    if _attempt[0] == 1:
        record_step(Step(
            index=1, action="process_data", tool_called="process",
            error="AssertionError: agent claimed 100 records but tool returned 42",
            llm_output="I have processed all 100 records successfully.",
        ))
        raise RuntimeError("AssertionError: agent claimed 100 records but tool returned 42")

    record_step(Step(index=1, action="process_data", tool_called="process",
                     tool_input={"records": 42}, tool_output="processed 42 records"))

    # Phase 3
    record_step(Step(index=2, action="generate_report", tool_called="report",
                     tool_output="Report generated."))
    return f"Done. Processed 42 records for task: {task}"

async def main():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SQLiteCheckpointStore(db_path)

        # Map UNKNOWN → rollback so the demo works without LLMClassifier.
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

        result = await agent.run("Analyse Q1 sales data")
        print(result)

        latest = await store.latest()
        if latest:
            print(f"Latest checkpoint: {latest.id}")
            print(f"Trajectory steps: {len(latest.trajectory_snapshot)}")
    finally:
        os.unlink(db_path)

asyncio.run(main())
```

## Expected output

```
[triage] UNKNOWN detected at step 1
[triage] Dispatching: RecoveryAction.ROLLBACK()
[triage] Attempt 2...
Done. Processed 42 records for task: Analyse Q1 sales data
Latest checkpoint: <uuid>
Trajectory steps: 3
```

## Run

```bash
python examples/durable_checkpoints.py
```

## Production setup

For a real production agent, pair rollback with `HybridClassifier` so the LLM can semantically detect failures that pattern-matching misses (e.g. `PLAN_INCOMPLETE`, `CONTEXT_OVERFLOW`):

```python
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier
from triage.strategies.rollback import rollback_to_checkpoint

classifier = HybridClassifier(llm=LLMClassifier())
policy = triage.FailurePolicy(
    LOOP_DETECTED=rollback_to_checkpoint(),
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)
agent = triage.Agent(my_agent, policy=policy, classifier=classifier,
                     checkpoint_store=SQLiteCheckpointStore("prod.db"),
                     auto_checkpoint=True)
```
