# Attempt History

`attempt_history` lets strategies see every prior `(FailureType, action_kind)` pair from the current `Agent.run()` call. Use it to escalate intelligently instead of retrying forever.

## Where it lives

`attempt_history` is a field on `FailureContext`:

```python
@dataclass
class FailureContext:
    failure_type: FailureType
    trajectory: list[Step]
    critical_step_index: int
    original_task: str
    last_checkpoint_id: str | None
    raw_error: Exception | None
    metadata: dict[str, Any]
    attempt_history: list[tuple[FailureType, str]]  # (failure_type, action_kind)
```

Each entry is a `(FailureType, str)` tuple where the string is the lowercase action kind: `"retry"`, `"replan"`, `"rollback"`, `"resume"`, `"escalate"`, `"abort"`.

## Reading attempt history in a strategy

```python
from triage.taxonomy import FailureContext, FailureType
from triage.policy import RecoveryAction

async def smart_external_fault(ctx: FailureContext) -> RecoveryAction:
    # Count how many times EXTERNAL_FAULT has been seen this run
    external_faults = sum(
        1 for ft, _ in ctx.attempt_history
        if ft == FailureType.EXTERNAL_FAULT
    )
    if external_faults >= 3:
        return RecoveryAction.ESCALATE(
            message="External service unavailable after 3 retries."
        )
    delay = 2.0 ** len(ctx.attempt_history)   # exponential backoff
    return RecoveryAction.RETRY(delay=delay)
```

## Common patterns

### Escalate after N failures of any type

```python
async def escalate_after_3(ctx: FailureContext) -> RecoveryAction:
    if len(ctx.attempt_history) >= 3:
        return RecoveryAction.ESCALATE(message="Too many failures.")
    return RecoveryAction.RETRY()
```

### Detect oscillation between replan and loop

```python
async def anti_oscillation(ctx: FailureContext) -> RecoveryAction:
    recent = [kind for _, kind in ctx.attempt_history[-4:]]
    if recent.count("replan") >= 2 and recent.count("retry") >= 2:
        return RecoveryAction.ESCALATE(message="Agent is oscillating.")
    return RecoveryAction.REPLAN()
```

### Switch strategy after repeated same failure

```python
async def adaptive_strategy(ctx: FailureContext) -> RecoveryAction:
    same_type_count = sum(
        1 for ft, _ in ctx.attempt_history
        if ft == ctx.failure_type
    )
    if same_type_count == 0:
        return RecoveryAction.RETRY()
    elif same_type_count == 1:
        return RecoveryAction.REPLAN()
    else:
        return RecoveryAction.ROLLBACK()
```

## Scope

`attempt_history` accumulates within a single `Agent.run()` call. It resets when `run()` is called again. It is **not** persisted across process restarts.

## Relationship to max_recovery_attempts

`Agent` has a separate `max_recovery_attempts` guard (default `3`). When the attempt counter hits this limit, triage raises `TriageEscalationError` regardless of what the strategy returns. `attempt_history` is a lower-level tool for strategies that want finer-grained control over when to give up.

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    max_recovery_attempts=5,   # hard cap
)
```
