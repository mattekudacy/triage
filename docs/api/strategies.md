# Strategies

Built-in strategy factory functions. Each returns a
`StrategyFn = Callable[[FailureContext], Awaitable[RecoveryAction]]`.

## retry_with_tool_manifest

::: triage.strategies.retry.retry_with_tool_manifest

## backoff_and_retry

::: triage.strategies.retry.backoff_and_retry

## replan

::: triage.strategies.replan.replan

## resume_from_subgoal

::: triage.strategies.replan.resume_from_subgoal

## rollback_to_checkpoint

::: triage.strategies.rollback.rollback_to_checkpoint

## circuit_breaker

::: triage.strategies.circuit_breaker.circuit_breaker

---

## Conceptual notes

### Strategies declare intent; Agent executes it

A strategy receives a `FailureContext` and returns a `RecoveryAction`. It must not
call the wrapped agent, restore checkpoints, or sleep. `Agent` executes the action.

### Custom strategies

```python
from triage.taxonomy import FailureContext, FailureType
from triage.policy import RecoveryAction

async def escalate_after_3_faults(ctx: FailureContext) -> RecoveryAction:
    faults = sum(1 for ft, _ in ctx.attempt_history if ft == FailureType.EXTERNAL_FAULT)
    if faults >= 3:
        return RecoveryAction.ESCALATE("Service unavailable after 3 retries.")
    return RecoveryAction.RETRY(delay=2.0**faults)
```

### Sequencing strategies

Use `FailurePolicy.sequence()` to step through strategies across successive failures
of the same type without managing state yourself:

```python
from triage.policy import FailurePolicy
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan

policy = FailurePolicy(
    EXTERNAL_FAULT=FailurePolicy.sequence(
        backoff_and_retry(max_attempts=2),
        replan(hint="External service may be down — try a different approach."),
    ),
)
```
