# Known Limitations

This page documents the current limitations of triage honestly. Some are design tradeoffs, some are on the roadmap, and some are fundamental constraints of the approach.

---

## Classifier

### RulesClassifier cannot detect semantic failures

`RulesClassifier` is pattern-based and makes zero API calls. It reliably detects structural failures — wrong tool name, bad JSON, HTTP errors, loops — but it physically cannot detect:

- `PLAN_INCOMPLETE` — requires knowing what sub-goals were supposed to be completed
- `CONTEXT_OVERFLOW` — requires detecting that the agent lost track of earlier context

These types return `UNKNOWN` from `RulesClassifier`. If your agent produces them, either:

- Map `UNKNOWN` to an appropriate recovery strategy as a catch-all
- Use `HybridClassifier(llm=LLMClassifier())` to get semantic detection at low cost (LLM is only called when rules return `UNKNOWN`)

### LLMClassifier blocks the event loop ~100–400ms — unless you use aclassify()

`LLMClassifier.classify()` is synchronous — required by the `Classifier` protocol (see `core.md` Rule 5) — so triage runs it via `anyio.to_thread.run_sync()` on the failure path, adding ~100–400ms of thread-hop latency.

As of v0.10, `LLMClassifier` (and `HybridClassifier`, when wrapping one) also defines `async def aclassify(trajectory, task)`, using the native async Anthropic/OpenAI client. `agent.py` detects `aclassify` via `getattr` and awaits it directly instead of dispatching to a thread — no protocol change required, since most classifiers (e.g. `RulesClassifier`) have no I/O and don't need it. This is automatic: pass `LLMClassifier()` or `HybridClassifier(llm=LLMClassifier())` to `Agent(classifier=...)` and the async path is used whenever available.

### No published benchmarks

There are no published false-positive/false-negative rates for `RulesClassifier` or `LLMClassifier`. Accuracy depends heavily on the frameworks, models, and error messages your agents produce. The `examples/benchmark.py` script runs a synthetic suite against both classifiers so you can measure accuracy for your own trajectories. Published numbers are on the roadmap for v0.14.

### Error messages are framework- and locale-dependent

`RulesClassifier` patterns are written for English-language error messages from major Python SDKs (OpenAI, Anthropic, LangGraph, CrewAI). If your framework surfaces errors in a different language or format, pattern coverage will be lower. In that case, supply a custom classifier or use `LLMClassifier`.

---

## Recovery

### Rollback does not undo side effects

`ROLLBACK` restores the trajectory snapshot and any state saved via `update_state()`. It does **not** undo:

- HTTP requests already sent
- Database writes already committed
- Emails or notifications already dispatched
- Files already written to disk

If your agent must be rollback-safe, design tools to be idempotent — re-running them after rollback should produce the same result, not a duplicate. Consider using database transactions, idempotency keys on HTTP calls, or staging areas for file writes.

### record_step is an honor system

triage has no way to intercept what your agent does internally. If your agent raises an exception before calling `record_step()`, the trajectory will be empty. triage handles this by synthesizing a sentinel step from the raw exception (`action="<no steps recorded>", error=str(exc)`), so the classifier still runs — but the trajectory context will be minimal.

The implication: the more faithfully your agent calls `record_step()` for each observable action, the more accurate the classifier will be. A trajectory with one sentinel step will almost always classify as `UNKNOWN`.

### Global attempt cap

`max_recovery_attempts` (default 3) counts total attempts per `run()` call across all failure types. For a hard cross-type cap, pass `max_total_attempts=N` to `Agent.__init__` (shipped v0.4). For custom logic, inspect `ctx.attempt_history` in a strategy:

```python
async def bounded_recovery(ctx: FailureContext) -> RecoveryAction:
    if len(ctx.attempt_history) >= 3:
        return RecoveryAction.ESCALATE("Too many failures of any type.")
    return RecoveryAction.RETRY()
```

### Strategy composition

Two built-in factories cover most cases:

- `FailurePolicy.sequence(s1, s2, s3)` (v0.13) — steps through strategies in order across successive failures of the same type. Escalates once all are exhausted.
- `FailurePolicy.chain(primary, fallback, after_kinds)` (v0.4) — falls through to `fallback` when `primary` returns an action whose `kind` is in `after_kinds` (default `"escalate"`).

For logic that doesn't fit either, inspect `ctx.attempt_history` in a custom strategy:

```python
async def replan_then_rollback(ctx: FailureContext) -> RecoveryAction:
    already_replanned = any(kind == "replan" for _, kind in ctx.attempt_history)
    if already_replanned:
        return RecoveryAction.ROLLBACK()
    return RecoveryAction.REPLAN(hint="Previous plan failed. Try a different approach.")
```

---

## API

### Only async agents are supported

`Agent` wraps `async def` callables only. Synchronous agent functions must be wrapped:

```python
import asyncio
from functools import partial

def my_sync_agent(task: str, *, record_step, **kwargs) -> str:
    ...

async def async_wrapper(task: str, *, record_step, **kwargs) -> str:
    loop = asyncio.get_event_loop()
    fn = partial(my_sync_agent, task, record_step=record_step, **kwargs)
    return await loop.run_in_executor(None, fn)

agent = triage.Agent(async_wrapper, policy=policy)
```

### Streaming agents require discrete step boundaries

The step-recording model assumes your agent produces observable, discrete actions. Streaming token-by-token output has no natural step boundary. triage works with streaming agents if you call `record_step()` at meaningful boundaries — tool call starts/ends, message completions, or plan transitions — rather than per token.

### _triage_hint is a plain string

Recovery hints injected as `_triage_hint` are unstructured strings designed to be passed directly into an LLM prompt. For programmatic use, prefer `_triage_context` (a typed `TriageContext` object injected alongside `_triage_hint` on every recovery attempt since v0.4) or `triage.get_recorder()` / `triage.get_state_updater()` to avoid signature changes entirely.

---

## Concurrency

### Concurrent run() calls on a single Agent instance

As of v0.10, `Agent` isolates per-run state (`_trajectory`, `_current_state`,
`_pending_checkpoints`, `_last_checkpoint_id`, `_last_ctx`) behind a `ContextVar`
rather than plain instance attributes. Because `asyncio`/`anyio` copy the current
`contextvars.Context` when spawning a new `Task`, two concurrent `run()` calls on
the *same* `Agent` instance — each in its own task — no longer see or corrupt
each other's trajectory or state:

```python
import anyio
import triage

agent = triage.Agent(my_agent, policy=policy)

async def run_parallel(tasks: list[str]) -> list:
    results = {}
    async def go(t):
        results[t] = await agent.run(t)
    async with anyio.create_task_group() as tg:
        for t in tasks:
            tg.start_soon(go, t)
    return results
```

`Agent.clone()` still exists and remains the right tool when you want fully
independent lifecycle hooks or per-task classifier/checkpoint-store instances —
but it is no longer required just to make concurrent `run()` calls safe.

Shared `CheckpointStore` instances are safe to share across agents. As of v0.11,
`InMemoryCheckpointStore` guards its storage dict with an `anyio.Lock`, so
concurrent `save()`/`load()`/`latest()` calls no longer race on the same dict.
`SQLiteCheckpointStore` and `RedisCheckpointStore` use atomic operations.

As of v0.13, checkpoints are tagged with a `run_id` generated once per `Agent.run()`
call. The rollback path calls `latest(run_id=...)` so each run rolls back to its own
most-recent checkpoint rather than the global newest. `CheckpointStore.latest()` still
accepts no argument and returns the global latest for callers that don't need scoping.

---

## Comparison with framework-native error handling

### vs. LangGraph

LangGraph's built-in error handling retries the full graph from the start. triage classifies the failure first and routes to a typed strategy — retry, replan, rollback, resume, escalate, or abort — with trajectory and state context available to the strategy. The two are composable: `wrap_langgraph()` adds triage's classification layer on top of a compiled LangGraph graph without replacing LangGraph's own logic.

### vs. try/except

`try/except` on exception type works well for synchronous, deterministic errors. Agent failures often carry no discriminating exception type — the same `RuntimeError` can mean a loop, a hallucination, or a network error depending on what the agent was doing before it raised. triage classifies on the trajectory, not the exception string.
