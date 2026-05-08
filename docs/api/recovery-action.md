# RecoveryAction

`RecoveryAction` is the return value of every strategy. It declares intent — `Agent` executes it.

## Constructors

All constructors are UPPERCASE classmethods. `None` parameters are excluded from `action.params`.

### RETRY

```python
RecoveryAction.RETRY(
    hint: str | None = None,
    inject: dict | None = None,
    delay: float = 0.0,
) -> RecoveryAction
```

Re-runs the agent. `hint` is injected as `_triage_hint`. `delay` seconds are awaited before the next attempt.

### REPLAN

```python
RecoveryAction.REPLAN(hint: str | None = None) -> RecoveryAction
```

Aborts the current plan branch and re-runs with a new planning instruction. `hint` defaults to `"Generate a new plan."` if not set by the strategy.

### ROLLBACK

```python
RecoveryAction.ROLLBACK(checkpoint_id: str | None = None) -> RecoveryAction
```

Restores trajectory and state from a checkpoint, then re-runs. `None` → uses the latest checkpoint. Injects `_triage_hint` and (if state is non-empty) `_triage_state`.

### RESUME

```python
RecoveryAction.RESUME(from_subgoal: str | None = None) -> RecoveryAction
```

Re-runs the agent with `_triage_subgoal` injected into kwargs.

### ESCALATE

```python
RecoveryAction.ESCALATE(message: str | None = None) -> RecoveryAction
```

Raises `TriageEscalationError` with the given message and the current `FailureContext`.

### ABORT

```python
RecoveryAction.ABORT(reason: str | None = None) -> RecoveryAction
```

Raises `TriageAbortError`. Hard stop — no further recovery.

## Attributes

```python
action.kind    # str: "retry" | "replan" | "rollback" | "resume" | "escalate" | "abort"
action.params  # dict: non-None kwargs passed to the constructor
```

`params` never contains `None` values — they are filtered out in `__init__`. Access payload with `action.params.get("key")`.

## Example

```python
from triage.policy import RecoveryAction
from triage.taxonomy import FailureContext, FailureType

async def my_strategy(ctx: FailureContext) -> RecoveryAction:
    if ctx.failure_type == FailureType.EXTERNAL_FAULT:
        if len(ctx.attempt_history) >= 3:
            return RecoveryAction.ESCALATE("Service down after 3 retries.")
        return RecoveryAction.RETRY(delay=2.0 ** len(ctx.attempt_history))
    return RecoveryAction.REPLAN()
```
