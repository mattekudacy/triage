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
    max_total_attempts: int | None = None,
    auto_checkpoint: bool = False,
    tracer: Any = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `fn` | required | The async agent callable to wrap |
| `policy` | required | Maps `FailureType` values to recovery strategies |
| `classifier` | `RulesClassifier()` | Classifies failures from the trajectory |
| `checkpoint_store` | `InMemoryCheckpointStore()` | Stores and retrieves checkpoints |
| `max_recovery_attempts` | `3` | Hard cap on recovery loop iterations per `run()` call |
| `max_total_attempts` | `None` | Global cap on `len(attempt_history)` across all failure types; `None` disables |
| `auto_checkpoint` | `False` | If `True`, saves a checkpoint after every `record_step()` call |
| `tracer` | `None` | OTel `Tracer` override; `None` = auto-detect (see [Observability](#observability)) |

### max_total_attempts vs max_recovery_attempts

`max_recovery_attempts` counts loop iterations. An agent that alternates between `EXTERNAL_FAULT` and `LOOP_DETECTED` can cycle up to `max_recovery_attempts` times regardless of which types appear.

`max_total_attempts` counts total entries in `attempt_history` — one per failure+dispatch pair across all types. It fires before `max_recovery_attempts` when the total accumulated attempts reaches the limit:

```python
agent = triage.Agent(
    my_agent, policy=policy,
    max_recovery_attempts=5,    # per-loop guard
    max_total_attempts=3,       # global guard — fires first if reached
)
```

## Concurrent run() calls

A single `Agent` instance can safely handle multiple `run()` calls in flight at once — each call's trajectory, state, and checkpoint bookkeeping is isolated per-task via `contextvars`:

```python
import anyio

agent = triage.Agent(my_agent, policy=policy)

async def run_all(tasks: list[str]) -> dict[str, Any]:
    results = {}
    async def go(t: str) -> None:
        results[t] = await agent.run(t)
    async with anyio.create_task_group() as tg:
        for t in tasks:
            tg.start_soon(go, t)
    return results
```

## clone()

```python
def clone(self) -> Agent
```

Returns a new `Agent` sharing the same `fn`, `policy`, `classifier`, and `checkpoint_store` but with fresh per-run state, independent lifecycle hooks, and its own `_last_ctx`. Concurrency is no longer the reason to reach for `clone()` — use it when you want a task to have its own hooks or a dedicated checkpoint store:

```python
import asyncio

agents = [agent.clone() for _ in tasks]
results = await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks)])
```

## run()

```python
async def run(self, task: str, **kwargs) -> Any
```

Runs the wrapped agent. On failure, classifies the trajectory, dispatches the policy, and re-runs with injected context. Returns the result on success.

Raises:
- `TriageEscalationError` — strategy returns `ESCALATE`, `max_recovery_attempts` exceeded, or `max_total_attempts` exceeded
- `TriageAbortError` — strategy returns `ABORT`

## Wrapped function contract

```python
async def my_agent(
    task: str,
    *,
    record_step: Callable[[Step], None],
    update_state: Callable[[dict], None],
    _triage_context: TriageContext | None = None,  # injected after any failure
    _triage_hint: str | None = None,               # backward-compat scalar hint
    _triage_subgoal: str | None = None,            # injected on RESUME
    _triage_state: dict | None = None,             # injected on ROLLBACK
    **kwargs,
) -> Any:
    ...
```

- `record_step` — call once per observable action; drives trajectory classification
- `update_state` — persist data to be restored on `ROLLBACK`
- `_triage_context` — typed `TriageContext` with `failure_type`, `hint`, `subgoal`, `state`, `attempt_number`; available on every recovery attempt

## contextvars — zero-signature-change injection

If you cannot or do not want to change a function's signature, use `triage.get_recorder()` and `triage.get_state_updater()` inside the agent body:

```python
from triage.agent import get_recorder, get_state_updater

async def my_agent(task: str, **kwargs) -> str:
    record = get_recorder()       # works inside Agent.run() context
    update = get_state_updater()
    record(Step(index=0, action="fetch", tool_output=data))
    update({"data": data})
    return result
```

Both raise `RuntimeError` if called outside a `run()` context.

## Decorator form

```python
@triage.agent(policy=my_policy, auto_checkpoint=True)
async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    ...
```

## TriageEscalationError

```python
class TriageEscalationError(Exception):
    context: FailureContext
```

Raised when a strategy returns `ESCALATE` or either attempt cap is exceeded.

## TriageAbortError

```python
class TriageAbortError(Exception):
    context: FailureContext
```

Raised when a strategy returns `ABORT`. Hard stop — no further recovery.

## Observability

Install the optional extra to enable OpenTelemetry spans:

```bash
pip install triage-agent[otel]
```

### Auto-detection

When `opentelemetry-sdk` is installed and you've configured a real tracer provider, triage automatically emits spans — no code change required:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace

trace.set_tracer_provider(TracerProvider())  # configure your exporter here
agent = triage.Agent(my_agent, policy=policy)  # no tracer= arg needed
```

### Explicit tracer

Pass `tracer=` to override auto-detection:

```python
import opentelemetry.trace as otel_trace

tracer = otel_trace.get_tracer("my-app")
agent = triage.Agent(my_agent, policy=policy, tracer=tracer)
```

### Spans emitted

| Span name | When | Key attributes |
|-----------|------|----------------|
| `triage.run` | Wraps the entire `run()` call, including retries | `triage.run_id`, `triage.task` |
| `triage.classify` | Wraps each failure classification | `triage.run_id`, `triage.failure_type` |
| `triage.dispatch` | Wraps each strategy dispatch | `triage.run_id`, `triage.attempt`, `triage.action_kind`, `triage.failure_type` |

All spans from one `run()` call share the same `trace_id` and `triage.run_id` (the UUID assigned per run, also used to scope rollback checkpoints). Escalate and abort actions set the `triage.dispatch` span status to `ERROR`.

Spans are **additive** — the six existing structured log events (`failure_classified`, `action_dispatched`, etc.) continue to emit regardless of whether OTel is configured.

### Aggregating into a failure distribution

The `triage.classify` and `triage.run` span attributes are all you need to produce a per-type failure frequency and recovery-rate report across a run population. See the [Failure Distribution example](../examples/failure-distribution.md) for a working implementation using an `InMemorySpanExporter` and guidance on adapting it to a production backend.

## Example

```python
import triage
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step, TriageContext

async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    tc: TriageContext | None = kwargs.get("_triage_context")
    if tc:
        print(f"Recovery attempt {tc.attempt_number}: {tc.hint}")

    data = fetch_data(task)
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})
    return process(data)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(
    my_agent, policy=policy,
    max_total_attempts=5,
    auto_checkpoint=True,
)

try:
    result = await agent.run("analyse Q1 data")
except triage.TriageEscalationError as exc:
    print(f"Needs review: {exc.context.failure_type.value}")
```
