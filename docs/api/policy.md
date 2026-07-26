# FailurePolicy

`FailurePolicy` is a dataclass that maps each `FailureType` to a strategy callable.
It is the primary configuration object passed to `Agent`.

## FailurePolicy

::: triage.policy.FailurePolicy
    options:
      members:
        - __init__
        - escalate_by_default
        - abort_by_default
        - chain
        - sequence
        - from_yaml

---

## Conceptual notes

### Declaration

Pass strategy factories as keyword arguments matching `FailureType` member names.
`default` is a catch-all for any type without an explicit entry:

```python
from triage.policy import FailurePolicy
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.strategies.replan import replan, resume_from_subgoal

policy = FailurePolicy(
    WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=5),
    LOOP_DETECTED=replan(hint="Try a different approach."),
    CONSTRAINT_IGNORED=replan(hint="Re-read the constraints carefully."),
    PLAN_INCOMPLETE=resume_from_subgoal(),
    default=FailurePolicy.escalate_by_default(),
)
```

### Custom strategy callables

Any `async def` that takes a `FailureContext` and returns a `RecoveryAction` is a valid
strategy:

```python
async def smart_external_fault(ctx: FailureContext) -> RecoveryAction:
    external_faults = sum(1 for ft, _ in ctx.attempt_history if ft == FailureType.EXTERNAL_FAULT)
    if external_faults >= 3:
        return RecoveryAction.ESCALATE(message="Service unavailable after 3 retries.")
    return RecoveryAction.RETRY(delay=2.0 ** len(ctx.attempt_history))
```

### Default actions

```python
# Escalate to human on any unhandled failure
default = FailurePolicy.escalate_by_default()

# Hard stop on any unhandled failure
default = FailurePolicy.abort_by_default()
```

### YAML / TOML loading

Policies can be loaded from a `.yaml` or `.toml` file (requires `pyyaml` for YAML):

```python
policy = FailurePolicy.from_yaml("policy.yaml")
```

Built-in strategy names: `backoff_and_retry`, `retry_with_tool_manifest`, `replan`,
`resume_from_subgoal`, `rollback_to_checkpoint`, `escalate`, `abort`.
Pass `strategy_registry={"my_fn": my_fn}` for custom strategies.
