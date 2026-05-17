# FailureType & Step

Core data types used throughout the triage API.

## FailureType

```python
from triage.taxonomy import FailureType
```

An `Enum` with 9 members. The classifier assigns one value per failure; the policy maps each to a strategy.

| Member | String value | Default recovery |
|---|---|---|
| `WRONG_TOOL_CALLED` | `wrong_tool_called` | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | `constraint_ignored` | Replan with constraint reminder |
| `LOOP_DETECTED` | `loop_detected` | Replan or rollback |
| `PLAN_INCOMPLETE` | `plan_incomplete` | Resume from subgoal |
| `SCHEMA_MISMATCH` | `schema_mismatch` | Retry with schema hint |
| `CONTEXT_OVERFLOW` | `context_overflow` | Replan with compressed context |
| `EXTERNAL_FAULT` | `external_fault` | Backoff and retry |
| `TIMEOUT` | `timeout` | Backoff and retry |
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
    idempotent: bool = False            # set True only for read-only / safe-to-retry steps
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
    idempotent=True,   # safe to replay — read-only
))
```

`idempotent` defaults to `False`. Mark a step `idempotent=True` only when it is genuinely safe to replay without side effects (read-only tool calls, pure computations). Steps that send emails, write to databases, or charge payment methods must remain `False`.

When `Agent(strict_idempotency=True)` is set, any `RETRY` action is blocked if the trajectory contains a step with `idempotent=False`, and `TriageEscalationError` is raised instead.

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

---

## TriageContext

```python
from triage.taxonomy import TriageContext
```

Structured recovery context injected into agents as `_triage_context`. Replaces the scattered `_triage_hint`, `_triage_subgoal`, and `_triage_state` kwargs with a single typed object. The individual kwargs are still injected for backward compatibility.

```python
@dataclass
class TriageContext:
    failure_type: FailureType   # what was classified
    attempt_number: int         # 0-based attempt index from this run()
    hint: str | None            # mirrors _triage_hint
    subgoal: str | None         # mirrors _triage_subgoal
    state: dict[str, Any]       # mirrors _triage_state (empty dict if no state)
```

Usage inside a wrapped agent:

```python
async def my_agent(task: str, *, record_step, update_state, **kwargs) -> Any:
    tc: TriageContext | None = kwargs.get("_triage_context")
    if tc:
        print(f"Recovering from {tc.failure_type.value}, attempt {tc.attempt_number}")
        if tc.hint:
            # pass hint into the LLM prompt
            ...
        if tc.state:
            # skip re-fetching — state was restored from checkpoint
            data = tc.state.get("data")
```
