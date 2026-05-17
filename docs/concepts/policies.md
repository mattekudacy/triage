# Policies & Recovery Actions

A `FailurePolicy` maps each `FailureType` to a strategy — an async callable that decides what to do next. The strategy returns a `RecoveryAction`, and `Agent` executes it.

## Declaring a policy

```python
import triage
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan, resume_from_subgoal
from triage.strategies.rollback import rollback_to_checkpoint

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    SCHEMA_MISMATCH    = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    CONSTRAINT_IGNORED = replan(hint="Re-read the constraints carefully."),
    PLAN_INCOMPLETE    = resume_from_subgoal(),
    default            = triage.FailurePolicy.escalate_by_default(),
)
```

Any `FailureType` not listed falls through to `default`. If `default` is unset, triage escalates automatically.

---

## RecoveryAction

`RecoveryAction` is the return type of every strategy. It declares intent — `Agent` executes it.

### RETRY

Re-runs the agent. Optionally injects a hint or delays before retrying.

```python
RecoveryAction.RETRY()
RecoveryAction.RETRY(hint="The correct tool name is 'search_web'.")
RecoveryAction.RETRY(delay=2.0)
```

| Parameter | Type | Description |
|---|---|---|
| `hint` | `str \| None` | Injected as `_triage_hint` kwarg on the next call |
| `delay` | `float` | Seconds to wait before retrying (default `0.0`) |

### REPLAN

Aborts the current plan branch and restarts with a new planning instruction.

```python
RecoveryAction.REPLAN()
RecoveryAction.REPLAN(hint="The agent was drifting. Re-read the goal.")
```

| Parameter | Type | Description |
|---|---|---|
| `hint` | `str \| None` | Injected as `_triage_hint`; defaults to `"Generate a new plan."` |

### ROLLBACK

Restores trajectory and state from a checkpoint, then re-runs.

```python
RecoveryAction.ROLLBACK()                           # uses latest checkpoint
RecoveryAction.ROLLBACK(checkpoint_id="abc-123")    # specific checkpoint
```

On execution, `Agent` loads the checkpoint and injects:
- `_triage_hint` — `"Rolled back to checkpoint '<id>'."`
- `_triage_state` — the checkpoint's state dict (only if non-empty)

Requires a `checkpoint_store` to be configured. If no checkpoint is available, triage escalates.

### RESUME

Continues execution from a named sub-goal.

```python
RecoveryAction.RESUME(from_subgoal="Step 3: validate the output schema")
```

| Parameter | Type | Description |
|---|---|---|
| `from_subgoal` | `str \| None` | Injected as `_triage_subgoal` kwarg |

### ESCALATE

Halts autonomous execution and raises `TriageEscalationError`.

```python
RecoveryAction.ESCALATE(message="External service unavailable after 3 retries.")
```

The calling code can catch `TriageEscalationError` and route to a human review queue.

### ABORT

Hard stop. Raises `TriageAbortError` immediately with no further recovery.

```python
RecoveryAction.ABORT(reason="Attempted to delete production data.")
```

---

## Built-in strategies

Import from `triage.strategies.*`:

### retry strategies

```python
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
```

**`retry_with_tool_manifest(max_attempts=3)`**
Retries and injects a hint reminding the agent to use tools from the manifest. Escalates after `max_attempts`.

**`backoff_and_retry(max_attempts=5, base_delay=1.0)`**
Exponential backoff retry. Delay doubles with each attempt: `base_delay * 2^n`. Escalates after `max_attempts`.

### replan strategies

```python
from triage.strategies.replan import replan, resume_from_subgoal
```

**`replan(hint=None, max_replans=3)`**
Returns `REPLAN` with the given hint. Escalates if the same failure type has already been replanned `max_replans` times (checked via `attempt_history`).

**`resume_from_subgoal()`**
Returns `RESUME` with the last incomplete sub-goal from the trajectory.

### rollback strategies

```python
from triage.strategies.rollback import rollback_to_checkpoint
```

**`rollback_to_checkpoint()`**
Returns `ROLLBACK` pointing to the latest available checkpoint.

---

## Writing a custom strategy

A strategy is any `async def` that takes a `FailureContext` and returns a `RecoveryAction`:

```python
from triage.taxonomy import FailureContext, FailureType
from triage.policy import RecoveryAction

async def smart_external_fault(ctx: FailureContext) -> RecoveryAction:
    external_faults = sum(
        1 for ft, _ in ctx.attempt_history
        if ft == FailureType.EXTERNAL_FAULT
    )
    if external_faults >= 3:
        return RecoveryAction.ESCALATE(
            message="External service unavailable after 3 retries."
        )
    delay = 2.0 ** len(ctx.attempt_history)
    return RecoveryAction.RETRY(delay=delay)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=smart_external_fault,
    default=triage.FailurePolicy.escalate_by_default(),
)
```

---

## Policy defaults

`FailurePolicy` provides two convenience defaults:

```python
# Escalate to human on any unhandled failure
default=triage.FailurePolicy.escalate_by_default()

# Hard stop on any unhandled failure
default=triage.FailurePolicy.abort_by_default()
```

If `default` is `None` and a failure type has no strategy, triage escalates automatically.

---

## Kwargs injected by Agent

When `Agent` executes a recovery action, it injects context into the next call's kwargs:

| Key | Set by | Value |
|---|---|---|
| `_triage_hint` | RETRY, REPLAN, ROLLBACK | Instruction string for the agent |
| `_triage_subgoal` | RESUME | Sub-goal string to resume from |
| `_triage_state` | ROLLBACK (non-empty state) | Dict restored from the checkpoint |

Your agent should accept `**kwargs` and check for these:

```python
async def my_agent(task: str, *, record_step, update_state, **kwargs):
    hint = kwargs.get("_triage_hint")
    state = kwargs.get("_triage_state", {})

    if hint:
        print(f"Recovery hint: {hint}")
    if state:
        # skip re-fetching — triage restored the last known state
        data = state.get("data")
```
