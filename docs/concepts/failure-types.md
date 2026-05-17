# Failure Types

`FailureType` is the core enum that drives every recovery decision. The classifier assigns one value per failure; the policy maps each value to a strategy.

## All 9 types

| Member | String value | When it occurs | Default recovery |
|---|---|---|---|
| `WRONG_TOOL_CALLED` | `wrong_tool_called` | Agent called a tool that doesn't exist or isn't in the manifest | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | `constraint_ignored` | LLM output violates an explicit constraint from the task | Replan with constraint reminder |
| `LOOP_DETECTED` | `loop_detected` | Agent repeating the same tool + input across 3+ steps | Replan or rollback |
| `PLAN_INCOMPLETE` | `plan_incomplete` | Agent declared success before completing all sub-goals | Resume from subgoal |
| `SCHEMA_MISMATCH` | `schema_mismatch` | Tool output or LLM response didn't match expected structure | Retry with schema hint |
| `CONTEXT_OVERFLOW` | `context_overflow` | Agent lost earlier task context due to long-horizon drift | Replan with compressed context |
| `EXTERNAL_FAULT` | `external_fault` | Tool/API returned a transient error (429, 500, 502, 503) | Backoff and retry |
| `TIMEOUT` | `timeout` | Wall-clock or async deadline exceeded | Backoff and retry |
| `UNKNOWN` | `unknown` | Classifier could not determine the failure type | Escalate |

## Classifier coverage

Not all failure types are detectable by the pattern-based `RulesClassifier`. The table below shows which classifier is required for each type:

| Type | RulesClassifier | LLMClassifier / HybridClassifier |
|---|---|---|
| `WRONG_TOOL_CALLED` | ✓ | ✓ |
| `SCHEMA_MISMATCH` | ✓ | ✓ |
| `EXTERNAL_FAULT` | ✓ | ✓ |
| `TIMEOUT` | ✓ | ✓ |
| `LOOP_DETECTED` | ✓ (window configurable) | ✓ |
| `CONSTRAINT_IGNORED` | ✓ (requires `constraints=` arg) | ✓ |
| `PLAN_INCOMPLETE` | ✗ → `UNKNOWN` | ✓ |
| `CONTEXT_OVERFLOW` | ✗ → `UNKNOWN` | ✓ |

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` require semantic understanding of the trajectory — pattern-matching cannot detect them. If you use `RulesClassifier` alone, these failures arrive at your `UNKNOWN` strategy. Use `HybridClassifier` (rules first, LLM on `UNKNOWN`) to get full coverage without paying for an LLM call on every failure.

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
