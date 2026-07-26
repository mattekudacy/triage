# FailureType & Step

Core data types used throughout the triage API.

## FailureType

::: triage.taxonomy.FailureType

The 9 members and their stable string values:

| Member | String value | Default recovery intent |
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

String values are the stable public identifiers used in logs and serialized state.

## Step

::: triage.taxonomy.Step

`idempotent` defaults to `False`. Mark `True` only for steps that are genuinely safe
to replay — read-only tool calls and pure computations. Steps that send email, write
to a database, or charge payment methods must stay `False`.

When `Agent(strict_idempotency=True)` is set, a `RETRY` action is blocked if the
trajectory contains any `idempotent=False` step.

## FailureContext

::: triage.taxonomy.FailureContext

### attempt_history

A list of `(FailureType, action_kind)` tuples from all prior recovery attempts in the
current `run()` call. Use it to detect repeated failures and escalate intelligently:

```python
prior_retries = sum(1 for _, kind in ctx.attempt_history if kind == "retry")
if prior_retries >= 2:
    return RecoveryAction.ESCALATE("Too many retries.")
```

## TriageContext

::: triage.taxonomy.TriageContext

Injected as `_triage_context` on every recovery attempt:

```python
async def my_agent(task: str, *, record_step, **kwargs) -> Any:
    tc: TriageContext | None = kwargs.get("_triage_context")
    if tc:
        print(f"Recovering from {tc.failure_type.value}, attempt {tc.attempt_number}")
        if tc.hint:
            ...  # pass hint into the LLM prompt
        if tc.state:
            data = tc.state.get("data")  # restored from checkpoint
```
