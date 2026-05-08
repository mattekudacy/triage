# Agent

`triage.Agent` is the core wrapper. It runs your async callable in a retry loop, classifying failures and dispatching recovery actions.

## Constructor

```python
Agent(
    fn: Callable[..., Awaitable[Any]],
    policy: FailurePolicy,
    *,
    classifier: Classifier | None = None,
    checkpoint_store: CheckpointStore | None = None,
    max_recovery_attempts: int = 3,
    auto_checkpoint: bool = False,
)
```

| Parameter | Default | Description |
|---|---|---|
| `fn` | required | The async agent callable to wrap |
| `policy` | required | Maps `FailureType` values to recovery strategies |
| `classifier` | `RulesClassifier()` | Classifies failures from the trajectory |
| `checkpoint_store` | `InMemoryCheckpointStore()` | Stores and retrieves checkpoints |
| `max_recovery_attempts` | `3` | Hard cap on recovery attempts per `run()` call |
| `auto_checkpoint` | `False` | If `True`, saves a checkpoint after every `record_step()` call |

## run()

```python
async def run(self, task: str, **kwargs) -> Any
```

Runs the wrapped agent. On failure, classifies the trajectory, dispatches the policy, and re-runs with injected context. Returns the result on success.

Raises:
- `TriageEscalationError` — when a strategy returns `ESCALATE` or `max_recovery_attempts` is exceeded
- `TriageAbortError` — when a strategy returns `ABORT`

Extra `**kwargs` are passed through to `fn` on the first call and re-passed (with triage additions) on each recovery attempt.

## Wrapped function contract

```python
async def my_agent(
    task: str,
    *,
    record_step: Callable[[Step], None],
    update_state: Callable[[dict], None],
    _triage_hint: str | None = None,       # injected on RETRY / REPLAN / ROLLBACK
    _triage_subgoal: str | None = None,    # injected on RESUME
    _triage_state: dict | None = None,     # injected on ROLLBACK (non-empty state)
    **kwargs,
) -> Any:
    ...
```

- `record_step` — call once per observable action; drives trajectory classification
- `update_state` — call to persist data that will be restored on `ROLLBACK`
- Both are injected by triage; do not import them

## Decorator form

```python
@triage.agent(policy=my_policy, auto_checkpoint=True)
async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    ...
```

`@triage.agent(...)` is a factory decorator that returns a `triage.Agent` instance.

## TriageEscalationError

```python
class TriageEscalationError(Exception):
    context: FailureContext
```

Raised when a strategy returns `ESCALATE` or `max_recovery_attempts` is exceeded. The `context` attribute contains the full `FailureContext` at the time of escalation.

## TriageAbortError

```python
class TriageAbortError(Exception):
    context: FailureContext
```

Raised when a strategy returns `ABORT`. Hard stop — no further recovery is attempted.

## Example

```python
import triage
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    data = fetch_data(task)
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})
    return process(data)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(my_agent, policy=policy, auto_checkpoint=True)

try:
    result = await agent.run("analyse Q1 data")
except triage.TriageEscalationError as exc:
    print(f"Needs review: {exc.context.failure_type.value}")
```
