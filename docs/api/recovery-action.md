# RecoveryAction

`RecoveryAction` is the return value of every strategy. It declares intent — `Agent` executes it.

::: triage.policy.RecoveryAction
    options:
      members:
        - RETRY
        - REPLAN
        - ROLLBACK
        - RESUME
        - ESCALATE
        - ABORT
        - SUSPEND

---

## Conceptual notes

### Constructors are UPPERCASE classmethods

```python
RecoveryAction.RETRY(hint="...", inject={"key": "val"}, delay=1.0)
RecoveryAction.REPLAN(hint="...")
RecoveryAction.ROLLBACK(checkpoint_id=None)  # None → latest checkpoint in the run
RecoveryAction.RESUME(from_subgoal="Step 3: summarise results")
RecoveryAction.ESCALATE(message="Needs human review")
RecoveryAction.ABORT(reason="Unrecoverable state")
RecoveryAction.SUSPEND(message="Approve?", metadata={"channel": "#ops"})
```

### Accessing the payload

```python
action.kind    # str: "retry" | "replan" | "rollback" | "resume" | "escalate" | "abort" | "suspend"
action.params  # dict: non-None kwargs passed to the constructor
```

`None` kwargs are excluded from `params`. Access with `.get()`, never by direct key:

```python
hint = action.params.get("hint")
```

### Custom strategy example

```python
async def my_strategy(ctx: FailureContext) -> RecoveryAction:
    faults = sum(1 for ft, _ in ctx.attempt_history if ft == FailureType.EXTERNAL_FAULT)
    if faults >= 3:
        return RecoveryAction.ESCALATE("Service unavailable after 3 retries.")
    return RecoveryAction.RETRY(delay=2.0**faults)
```
