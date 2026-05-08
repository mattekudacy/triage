# Failure Types

`FailureType` is the core enum that drives every recovery decision. The classifier assigns one value per failure; the policy maps each value to a strategy.

## All 10 types

| Member | String value | When it occurs | Default recovery |
|---|---|---|---|
| `WRONG_TOOL_CALLED` | `wrong_tool_called` | Agent called a tool that doesn't exist or isn't in the manifest | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | `constraint_ignored` | LLM output violates an explicit constraint from the task | Replan with constraint reminder |
| `LOOP_DETECTED` | `loop_detected` | Agent repeating the same tool + input across 3+ steps | Replan or rollback |
| `HALLUCINATED_STATE` | `hallucinated_state` | Agent asserts facts that contradict tool outputs | Rollback to checkpoint |
| `PLAN_INCOMPLETE` | `plan_incomplete` | Agent declared success before completing all sub-goals | Resume from subgoal |
| `SCHEMA_MISMATCH` | `schema_mismatch` | Tool output or LLM response didn't match expected structure | Retry with schema hint |
| `CONTEXT_OVERFLOW` | `context_overflow` | Agent lost earlier task context due to long-horizon drift | Replan with compressed context |
| `GOAL_DRIFT` | `goal_drift` | Agent making progress toward the wrong interpretation of the goal | Replan with goal restatement |
| `EXTERNAL_FAULT` | `external_fault` | Tool/API returned a transient error (429, 500, 502, 503) | Backoff and retry |
| `UNKNOWN` | `unknown` | Classifier could not determine the failure type | Escalate |

## Design rules

- **Order is stable** — do not reorder members. String values are used in logs, serialized state, and external systems.
- **`UNKNOWN` is always last** — new types are inserted before it.
- **String values are the stable public identifier** — never change them.

## Using failure types in strategies

```python
from triage.taxonomy import FailureType, FailureContext
from triage.policy import RecoveryAction

async def smart_strategy(ctx: FailureContext) -> RecoveryAction:
    if ctx.failure_type == FailureType.EXTERNAL_FAULT:
        # Check attempt history to avoid infinite backoff loops
        external_faults = sum(
            1 for ft, _ in ctx.attempt_history
            if ft == FailureType.EXTERNAL_FAULT
        )
        if external_faults >= 3:
            return RecoveryAction.ESCALATE(message="External service unavailable after 3 retries.")
    return RecoveryAction.RETRY(delay=2.0 ** len(ctx.attempt_history))
```
