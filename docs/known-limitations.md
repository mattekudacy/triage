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

### Accuracy is corpus-dependent

`RulesClassifier` scores 100% on the in-corpus synthetic suite in `examples/benchmark.py` (see `docs/concepts/classifiers.md` for the full table) — but that suite is training data, so the number says nothing about generalization.

The honest figure is the held-out one. On corpus C (27 entries from azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI, and LlamaIndex, scored once without editing `rules.py`), `RulesClassifier` gets **52% recall at 100% precision** — all 13 misses returned `UNKNOWN`, zero misroutes. Expect that shape on a stack the patterns have never seen: unrecognized errors fall through to your default policy rather than being routed to the wrong strategy.

Real-world accuracy depends on the frameworks, models, and error message formats your agents produce — particularly SDK version and language. Reproduce both measurements with:

```bash
PYTHONPATH=. python scripts/classifier_accuracy.py   # four-block corpus measurement
python examples/benchmark.py                         # synthetic suite
```

Add your own cases to `examples/benchmark.py`'s `CASES` list to measure coverage for your specific stack.

### Error messages are framework- and locale-dependent

`RulesClassifier` patterns are written for English-language error messages from major Python SDKs (OpenAI, Anthropic, LangGraph, botocore, and other common providers). If your framework surfaces errors in a different language or format, pattern coverage will be lower. In that case, supply a custom classifier or use `LLMClassifier`.

Note that `RulesClassifier(framework=...)` accepts only `"openai"`, `"anthropic"`, and `"langgraph"` for supplemental per-SDK patterns. Unknown values are silently ignored (generic patterns still apply), so a typo degrades coverage without raising.

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

## Type-checking escape hatches

`pyproject.toml` has three `[[tool.mypy.overrides]]` blocks. Each was added for a specific reason; this section keeps them from accumulating silently.

### `ignore_missing_imports = true` (all optional/third-party stubs)

Applies to: `anthropic`, `openai`, `aiosqlite`, `redis`, `langgraph`, `langchain`, `langchain_core`, `opentelemetry`, `tomllib`.

Standard override — none of those libraries ship typed stubs usable by mypy in strict mode. Removing it would require either vendoring stub packages or switching to a non-strict mypy config.

### `warn_unused_ignores = false` on three adapter/classifier modules

Applies to: `triage.policy`, `triage.classifier.llm`, `triage.adapters.langchain`.

These modules contain `# type: ignore[misc]` comments that suppress errors which only fire when the optional dependency (`langchain`, `anthropic`) is installed. When the dep is absent, mypy resolves the type to `Any` and the suppress becomes unused — triggering `[unused-ignore]`. Making the suppress conditional on the install state would require a per-file override for every possible install combination; `warn_unused_ignores = false` is the practical solution.

### `disable_error_code = ["assignment", "misc"]` on `triage.observability.*`

The conditional-import fallback pattern (`Tracer = Any`, `_otel_trace = None`) deliberately assigns incompatible types in the `except ImportError` branch. Suppressing only `[assignment]` and `[misc]` lets all other error codes (including real bugs) remain visible. Blanket `ignore_errors = true` was rejected because it would permanently uncheck the record helpers and the `id(meter)` cache logic.

---

## Comparison with framework-native error handling

### vs. LangGraph

LangGraph's built-in error handling retries the full graph from the start. triage classifies the failure first and routes to a typed strategy — retry, replan, rollback, resume, escalate, or abort — with trajectory and state context available to the strategy. The two are composable: `wrap_langgraph()` adds triage's classification layer on top of a compiled LangGraph graph without replacing LangGraph's own logic.

### vs. try/except

`try/except` on exception type works well for synchronous, deterministic errors. Agent failures often carry no discriminating exception type — the same `RuntimeError` can mean a loop, a hallucination, or a network error depending on what the agent was doing before it raised. triage classifies on the trajectory, not the exception string.
