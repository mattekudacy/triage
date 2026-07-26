# Agent

`triage.Agent` is the core wrapper. It runs your async callable in a retry loop,
classifying failures and dispatching recovery actions.

## Agent class

::: triage.agent.Agent
    options:
      members:
        - __init__
        - run
        - stream
        - resume
        - clone
        - report_misclassification

## Decorator form

::: triage.agent.agent

## Exceptions

::: triage.agent.TriageEscalationError

::: triage.agent.TriageAbortError

## ContextVar helpers

Use these inside an agent body to avoid changing its signature:

::: triage.agent.get_recorder

::: triage.agent.get_state_updater

::: triage.agent.get_usage_recorder

---

## Conceptual notes

### max_total_attempts vs max_recovery_attempts

`max_recovery_attempts` counts loop iterations within one `run()` call.
`max_total_attempts` counts the total `len(attempt_history)` across all failure types and
fires first when it is lower. Use both together to bound cross-type accumulation:

```python
agent = triage.Agent(
    my_agent, policy=policy,
    max_recovery_attempts=5,  # per-loop guard
    max_total_attempts=3,     # global guard — fires first if reached
)
```

### Concurrent run() calls

A single `Agent` instance is safe for concurrent `run()` calls. Each call's trajectory,
state, and checkpoint bookkeeping is isolated per-task via `ContextVar`:

```python
import anyio

async def run_all(tasks: list[str]) -> dict[str, Any]:
    results = {}

    async def go(t: str) -> None:
        results[t] = await agent.run(t)

    async with anyio.create_task_group() as tg:
        for t in tasks:
            tg.start_soon(go, t)
    return results
```

### Wrapped function contract

```python
async def my_agent(
    task: str,
    *,
    record_step: Callable[[Step], None],
    update_state: Callable[[dict], None],
    record_usage: Callable[[Usage], None],
    _triage_context: TriageContext | None = None,
    _triage_hint: str | None = None,        # backward-compat
    _triage_subgoal: str | None = None,     # backward-compat
    _triage_state: dict | None = None,      # backward-compat
    **kwargs,
) -> Any: ...
```

`_triage_context` is the canonical form — a typed object with `failure_type`,
`attempt_number`, `hint`, `subgoal`, and `state`. The individual `_triage_*` kwargs
remain for backward compatibility.

### Observability

Install the optional extra:

```bash
pip install triage-agent[otel]
```

When `opentelemetry-sdk` is installed and a real `TracerProvider` is configured, triage
emits three span types per `run()` call with no code change required:

| Span | When | Key attributes |
|------|------|----------------|
| `triage.run` | Wraps the entire call including retries | `triage.run_id`, `triage.task` |
| `triage.classify` | Wraps each failure classification | `triage.failure_type` |
| `triage.dispatch` | Wraps each strategy dispatch | `triage.action_kind`, `triage.attempt` |

All spans share the same `trace_id` and `triage.run_id`. Span status is `OK` on success,
`ERROR` on escalate/abort, and `UNSET` (incomplete) on cancellation. The six structured
log events (`failure_classified`, `action_dispatched`, etc.) emit regardless of whether
OTel is configured.
