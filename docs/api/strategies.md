# Strategies

Built-in strategy factory functions. Each returns a `StrategyFn = Callable[[FailureContext], Awaitable[RecoveryAction]]`.

---

## retry strategies

```python
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
```

### retry_with_tool_manifest(max_attempts=3)

Returns `RecoveryAction.RETRY` with a hint instructing the agent to use only tools in the current manifest.

```python
policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=2),
)
```

### backoff_and_retry(max_attempts=5)

Returns `RecoveryAction.RETRY` with exponential backoff. Delay = `2^attempt_number` seconds.

```python
policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=5),
)
```

---

## replan strategies

```python
from triage.strategies.replan import replan, resume_from_subgoal
```

### replan(hint=None, max_replans=3)

Returns `RecoveryAction.REPLAN` with the given hint (defaults to `"Generate a new plan. The previous approach failed."`).

```python
policy = triage.FailurePolicy(
    LOOP_DETECTED=replan(hint="You're stuck. Try a completely different approach."),
    CONSTRAINT_IGNORED=replan(hint="Re-read the original task before continuing."),
)
```

### resume_from_subgoal()

Returns `RecoveryAction.RESUME` with `from_subgoal` set to `ctx.metadata.get("incomplete_subgoal")`.

```python
policy = triage.FailurePolicy(
    PLAN_INCOMPLETE=resume_from_subgoal(),
)
```

---

## rollback strategies

```python
from triage.strategies.rollback import rollback_to_checkpoint
```

### rollback_to_checkpoint(checkpoint_id=None)

Returns `RecoveryAction.ROLLBACK`. If `checkpoint_id` is not given, uses `ctx.last_checkpoint_id` (the most recently saved checkpoint for this run).

```python
policy = triage.FailurePolicy(
    LOOP_DETECTED=rollback_to_checkpoint(),
)
```

---

## Writing custom strategies

A strategy is any `async def` taking `FailureContext` and returning `RecoveryAction`:

```python
from triage.taxonomy import FailureContext, FailureType
from triage.policy import RecoveryAction

async def escalate_after_3_faults(ctx: FailureContext) -> RecoveryAction:
    faults = sum(1 for ft, _ in ctx.attempt_history if ft == FailureType.EXTERNAL_FAULT)
    if faults >= 3:
        return RecoveryAction.ESCALATE("Service unavailable after 3 retries.")
    return RecoveryAction.RETRY(delay=2.0 ** faults)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=escalate_after_3_faults,
    default=triage.FailurePolicy.escalate_by_default(),
)
```

Strategies must not call the agent, restore checkpoints, or sleep — `Agent` executes the action they return.
