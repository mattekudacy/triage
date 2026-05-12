# Known Limitations

This page documents the current limitations of triage honestly. Some are design tradeoffs, some are planned for v0.4, and some are fundamental constraints of the approach.

---

## Classifier

### RulesClassifier cannot detect semantic failures

`RulesClassifier` is pattern-based and makes zero API calls. It reliably detects structural failures — wrong tool name, bad JSON, HTTP errors, loops — but it physically cannot detect:

- `HALLUCINATED_STATE` — requires comparing LLM assertions against tool outputs
- `GOAL_DRIFT` — requires understanding intent vs. trajectory
- `PLAN_INCOMPLETE` — requires knowing what sub-goals were supposed to be completed
- `CONTEXT_OVERFLOW` — requires detecting that the agent lost track of earlier context

These types return `UNKNOWN` from `RulesClassifier`. If your agent produces them, either:

- Map `UNKNOWN` to an appropriate recovery strategy as a catch-all
- Use `HybridClassifier(llm=LLMClassifier())` to get semantic detection at low cost (LLM is only called when rules return `UNKNOWN`)

### LLMClassifier blocks the event loop ~100–400ms

`LLMClassifier.classify()` is synchronous. triage runs it via `anyio.to_thread.run_sync()` to avoid freezing the event loop, but the classification still adds ~100–400ms latency on the failure path. This is acceptable for most agents since classification only happens after a failure, not on every step. A fully async classifier is planned for v0.4.

### No benchmarks yet

There are no published false-positive/false-negative rates for `RulesClassifier` or `LLMClassifier`. Accuracy depends heavily on the frameworks, models, and error messages your agents produce. The `examples/benchmark.py` script runs a synthetic suite against both classifiers so you can measure accuracy for your own trajectories.

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

### No global attempt cap across failure types

`max_recovery_attempts` (default 3) counts total attempts per `run()` call. However, each strategy also has its own `max_attempts` parameter. An agent that alternates between two failure types could cycle up to `max_recovery_attempts` times regardless of per-strategy limits.

Workaround: use `attempt_history` in a custom strategy to count total attempts across all types:

```python
async def bounded_recovery(ctx: FailureContext) -> RecoveryAction:
    if len(ctx.attempt_history) >= 3:
        return RecoveryAction.ESCALATE("Too many failures of any type.")
    return RecoveryAction.RETRY()
```

### Strategies are not composable (yet)

A `FailurePolicy` maps each `FailureType` to exactly one strategy. There is no built-in way to say "try replan first, then rollback if replan fails." The workaround is a custom strategy that checks `attempt_history`:

```python
async def replan_then_rollback(ctx: FailureContext) -> RecoveryAction:
    already_replanned = any(kind == "replan" for _, kind in ctx.attempt_history)
    if already_replanned:
        return RecoveryAction.ROLLBACK()
    return RecoveryAction.REPLAN(hint="Previous plan failed. Try a different approach.")
```

Strategy chaining (`FailurePolicy.chain(primary, fallback)`) is planned for v0.4.

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

### record_step requires a signature change

The wrapped agent function must accept `record_step` (and optionally `update_state`) as keyword arguments. Wrapping an existing agent function without modifying its signature requires either an adapter closure or — planned for v0.4 — a `contextvars`-based approach where `record_step` is accessible without being in the signature.

### _triage_hint is a plain string

Recovery hints injected as `_triage_hint` are unstructured strings. They are designed to be passed directly into an LLM prompt. There is no type-safe schema for what a hint means, which limits programmatic interpretation. A structured `_triage_context` object is planned for v0.4.

---

## Concurrency

### A single Agent instance is not safe for concurrent run() calls

`Agent` holds mutable run-state (`_trajectory`, `_current_state`, `_pending_checkpoints`) that is reset at the start of each `run()` call. Calling `run()` concurrently on the same instance will corrupt this state.

For parallel task dispatch, create one `Agent` instance per concurrent task:

```python
import asyncio
import triage

async def run_parallel(tasks: list[str]) -> list:
    agents = [triage.Agent(my_agent, policy=policy) for _ in tasks]
    return await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks)])
```

Or use a factory function:

```python
def make_agent() -> triage.Agent:
    return triage.Agent(my_agent, policy=policy, checkpoint_store=shared_store)

results = await asyncio.gather(*[make_agent().run(t) for t in tasks])
```

Shared `CheckpointStore` instances are safe to share across agents — `InMemoryCheckpointStore` has no concurrency protection (last-write-wins), but `SQLiteCheckpointStore` and `RedisCheckpointStore` use atomic operations.

---

## Comparison with framework-native error handling

### vs. LangGraph

LangGraph's built-in error handling retries the full graph from the start. triage classifies the failure first and routes to a typed strategy — retry, replan, rollback, resume, escalate, or abort — with trajectory and state context available to the strategy. The two are composable: `wrap_langgraph()` adds triage's classification layer on top of a compiled LangGraph graph without replacing LangGraph's own logic.

### vs. try/except

`try/except` on exception type works well for synchronous, deterministic errors. Agent failures often carry no discriminating exception type — the same `RuntimeError` can mean a loop, a hallucination, or a network error depending on what the agent was doing before it raised. triage classifies on the trajectory, not the exception string.
