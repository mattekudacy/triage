# FailureType & Step

Core data types used throughout the triage API.

## FailureType

```python
from triage.taxonomy import FailureType
```

An `Enum` with 10 members. The classifier assigns one value per failure; the policy maps each to a strategy.

| Member | String value | Default recovery |
|---|---|---|
| `WRONG_TOOL_CALLED` | `wrong_tool_called` | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | `constraint_ignored` | Replan with constraint reminder |
| `LOOP_DETECTED` | `loop_detected` | Replan or rollback |
| `HALLUCINATED_STATE` | `hallucinated_state` | Rollback to checkpoint |
| `PLAN_INCOMPLETE` | `plan_incomplete` | Resume from subgoal |
| `SCHEMA_MISMATCH` | `schema_mismatch` | Retry with schema hint |
| `CONTEXT_OVERFLOW` | `context_overflow` | Replan with compressed context |
| `GOAL_DRIFT` | `goal_drift` | Replan with goal restatement |
| `EXTERNAL_FAULT` | `external_fault` | Backoff and retry |
| `UNKNOWN` | `unknown` | Escalate |

String values are the stable public identifiers used in logs and serialized state. Never change them.

## Step

```python
from triage.taxonomy import Step
```

A single recorded action in the agent's trajectory.

```python
@dataclass
class Step:
    index: int                          # position in the trajectory
    action: str                         # human-readable description
    tool_called: str | None = None      # tool name if a tool was invoked
    tool_input: dict | None = None      # arguments passed to the tool
    tool_output: Any = None             # value returned by the tool
    llm_output: str | None = None       # text output from the LLM
    error: str | None = None            # error string if the step failed
    timestamp: float                    # unix timestamp (auto-set)
    state_hash: str | None = None       # optional hash for dedup
    metadata: dict[str, Any]           # arbitrary extra data
```

Record steps by calling the injected `record_step` callback:

```python
from triage.taxonomy import Step

record_step(Step(
    index=0,
    action="called search tool",
    tool_called="search",
    tool_input={"query": "latest AI news"},
    tool_output=["result 1", "result 2"],
))
```

## FailureContext

```python
from triage.taxonomy import FailureContext
```

Passed to every strategy callable. Contains everything needed to make a recovery decision.

```python
@dataclass
class FailureContext:
    failure_type: FailureType
    trajectory: list[Step]
    critical_step_index: int
    original_task: str
    last_checkpoint_id: str | None
    loop_steps: list[int] | None
    violated_constraint: str | None
    expected_schema: dict | None
    raw_error: Exception | None
    metadata: dict[str, Any]            # includes "attempt_number"
    attempt_history: list[tuple[FailureType, str]]
```

### Properties

```python
ctx.failed_step          # Step at critical_step_index, or None
ctx.steps_after_failure  # list[Step] after the critical step
```

### attempt_history

A list of `(FailureType, action_kind)` tuples from all prior recovery attempts in the current `run()` call. Use it to detect repeated failures and escalate intelligently:

```python
prior_retries = sum(1 for _, kind in ctx.attempt_history if kind == "retry")
if prior_retries >= 2:
    return RecoveryAction.ESCALATE("Too many retries.")
```
