# triage — project memory

## What this project is

`triage` is a framework-agnostic Python library (PyPI: `triage-agent`, import: `triage`)
that wraps any async agent callable, classifies failures by type, and routes each type to
a recovery strategy. It does not replace agent frameworks — it wraps them.

Core deps: `anyio>=4.0`, `pydantic>=2.0`, stdlib only inside `triage/`.
No framework imports anywhere in `triage/` core — adapters live in `triage/adapters/`.

## Repo layout

```
triage/                   — importable package
  taxonomy.py             — FailureType enum (9 members), Step (with idempotent field), FailureContext
  trajectory.py           — Trajectory class (append / replay_from / last_n_steps)
  checkpoint/             — Checkpoint package
    __init__.py           — re-exports Checkpoint, CheckpointStore, InMemoryCheckpointStore, make_checkpoint
    base.py               — Checkpoint dataclass, CheckpointStore protocol, make_checkpoint, serialization helpers
    memory.py             — InMemoryCheckpointStore (default, anyio.Lock-guarded since v0.11)
    sqlite.py             — SQLiteCheckpointStore (requires aiosqlite)
    redis.py              — RedisCheckpointStore (requires redis[asyncio])
  policy.py               — RecoveryAction (7 constructors), FailurePolicy dataclass
  agent.py                — Agent class (run/stream/resume/clone), Triage*Error, @agent decorator
  classifier/
    base.py               — Classifier protocol (runtime_checkable)
    rules.py              — RulesClassifier — 6 rules in priority order, sync, zero API calls
    llm.py                — LLMClassifier — semantic classifier; Anthropic or OpenAI-compatible backend
    hybrid.py             — HybridClassifier — rules first, LLMClassifier fallback on UNKNOWN
  strategies/
    retry.py              — retry_with_tool_manifest(), backoff_and_retry()
    replan.py             — replan(), resume_from_subgoal()
    rollback.py           — rollback_to_checkpoint()
    circuit_breaker.py    — circuit_breaker() — wraps any strategy, short-circuits while OPEN
    saga.py               — compensating_rollback() — runs undo callables before restore
  suspension.py           — SuspendedRun, SuspensionStore protocol, InMemorySuspensionStore
  suspension_redis.py     — RedisSuspensionStore (requires redis)
  breaker.py              — CircuitBreaker, BreakerState (CLOSED/OPEN/HALF_OPEN)
  breaker_store.py        — BreakerStore protocol, RedisBreakerStore (cross-worker state)
  usage.py                — Usage, UsageMeter — token/cost accounting for max_tokens/max_cost_usd
  pricing.py              — PRICE_TABLE, lookup_cost() — Anthropic per-token rates
  streaming.py            — StreamRetryEvent — yielded by Agent.stream() at retry boundaries
  adapters/
    langgraph.py          — wrap_langgraph() — wraps a compiled LangGraph StateGraph
    langchain.py          — wrap_langchain() — wraps a LangChain AgentExecutor
  observability/
    otel.py               — resolve_tracer(), run_span/classify_span/dispatch_span context managers (lazy OTel import)
    metrics.py            — resolve_meter() + five instruments (lazy OTel import)
  bench.py                — run_benchmark(), BenchReport, BenchResult — eval harness with baseline comparison
  feedback.py             — Correction, record_correction(), coverage_report() — misclassification feedback loop
  testing.py              — make_step(), RecordingAgent, assert_classifies_as()
  scorer/
    base.py               — RiskScore dataclass, StepRiskScorer protocol (sync, no API calls)
    rules.py              — RulesRiskScorer — destructive pattern detection, zero API calls
tests/                    — pytest-asyncio, asyncio_mode=auto, 709 tests (zero skips with `.[dev]`)
  data/                   — error_corpus_{a,b,c}.json — classifier accuracy corpora
examples/                 — runnable demos (raw_openai.py needs OPENAI_API_KEY)
scripts/                  — bench_synthetic.py, classifier_accuracy.py, gen_error_corpus*.py
```

## Running tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -x --tb=short
```

The venv uses Python 3.13 (`/opt/homebrew/bin/python3.13`). Install every dependency the
suite touches with the single `dev` extra — the full suite then runs with zero skips:
```bash
.venv/bin/pip install -e ".[dev]"
```

`PYTHONPATH=.` still works and remains the faster path for quick iteration without
reinstalling. Before marking work done, also run:
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy triage/ --strict
```

## Key design rules

- **All public APIs are `async def`**, even if the body has no awaits.
- **Strategies declare intent; `agent.py` executes it.** A `RecoveryAction.ROLLBACK`
  does not restore state — it tells `agent.py` which checkpoint to load.
- **`RulesClassifier.classify()` is synchronous** and must make zero API calls.
  `LLMClassifier` uses the Anthropic sync client — blocks ~100-400ms, acceptable on
  the failure path only.
- **No framework imports in core.** Adapter code for LangGraph, CrewAI, etc. lives in
  `triage/adapters/`. The only allowed imports inside `triage/` core are:
  `stdlib`, `anyio`, `pydantic`, and sibling `triage.*` modules.
- **Wrapped callables receive `record_step` by keyword injection**, not positional arg:
  ```python
  async def my_agent(task: str, *, record_step, **kwargs) -> Any: ...
  ```
- **`auto_checkpoint` uses `_pending_checkpoints` drain.** `_record_step` queues
  coroutines; `run()` drains them via `_drain_checkpoints()` before returning or
  re-raising. Guarantees checkpoints are awaited before any policy action runs.
- **`classify()` runs in a thread.** `Agent.run()` calls `classify` via
  `anyio.to_thread.run_sync()` so `LLMClassifier`'s blocking HTTP never freezes
  the event loop. `RulesClassifier` is fast but the thread overhead is negligible
  on the failure path.
- **Agent state is real.** Wrapped callables receive `update_state` as a second
  injected keyword argument. Calling `update_state({"key": value})` stores state
  in `_current_state`, which is written into every checkpoint. On ROLLBACK, the
  checkpoint state is restored to `_current_state` and injected into kwargs as
  `_triage_state`.

## FailureType taxonomy (stable — do not reorder)

9 members. HALLUCINATED_STATE and GOAL_DRIFT were removed in v0.7 — the LLMClassifier
had no disambiguation logic for them and they were indistinguishable from CONSTRAINT_IGNORED
without a labeled corpus.

| Member | String value | Recovery intent |
|--------|-------------|-----------------|
| WRONG_TOOL_CALLED | wrong_tool_called | retry with correct manifest |
| CONSTRAINT_IGNORED | constraint_ignored | replan with constraint reminder |
| LOOP_DETECTED | loop_detected | replan or rollback |
| PLAN_INCOMPLETE | plan_incomplete | resume from subgoal |
| SCHEMA_MISMATCH | schema_mismatch | retry with schema hint |
| CONTEXT_OVERFLOW | context_overflow | replan with compressed context |
| EXTERNAL_FAULT | external_fault | backoff_and_retry |
| TIMEOUT | timeout | backoff_and_retry or replan |
| UNKNOWN | unknown | escalate |

## RulesClassifier rule priority

1. LOOP_DETECTED — last `loop_window` steps (default 3, configurable): identical `tool_called` + canonical `tool_input`
2. WRONG_TOOL_CALLED — error matches tool-not-found patterns across OpenAI/Anthropic/generic SDKs
3. SCHEMA_MISMATCH — error matches `validation error|json.*parse|jsondecodeerror|invalid json|unexpected token`
4. EXTERNAL_FAULT — error contains `\b(429|500|502|503)\b` (word-boundary, avoids false positives)
5. TIMEOUT — error matches `timeout|timed out|deadline exceeded|time limit`
6. CONSTRAINT_IGNORED — `llm_output` contains any string from `self.constraints`
7. UNKNOWN — default

**RulesClassifier scope:** Detects only structural/syntactic failures. PLAN_INCOMPLETE and
CONTEXT_OVERFLOW require semantic understanding and always return UNKNOWN from RulesClassifier
— use LLMClassifier or HybridClassifier for those.

**`StepRiskScorer` contract:** `StepRiskScorer.__call__()` is synchronous and must not make
API calls. It is invoked on the hot path inside `_record_step` on every recorded step; keep
it fast (< 1ms for typical inputs). This matches the same zero-API-call contract as
`RulesClassifier.classify()`.

**Per-framework patterns:** `RulesClassifier(framework="openai"|"anthropic"|"langgraph")` adds
SDK-specific patterns for rules 2–4 (WRONG_TOOL, SCHEMA, EXTERNAL_FAULT). Generic patterns
always apply; framework patterns are ORed in. Unknown framework values are silently ignored.

**Zero-trajectory fallback:** If the agent raises before calling `record_step()`, Agent
synthesizes a sentinel `Step(action="<no steps recorded>", error=str(exc))` so the
classifier always has at least one step to inspect.

## RecoveryAction constructors (stable API)

```python
RecoveryAction.RETRY(hint, inject, delay)
RecoveryAction.REPLAN(hint)
RecoveryAction.ROLLBACK(checkpoint_id)   # None → agent uses store.latest()
RecoveryAction.RESUME(from_subgoal)
RecoveryAction.ESCALATE(message)         # raises TriageEscalationError in agent.py
RecoveryAction.SUSPEND(message, metadata) # raises TriageSuspendedError; resume via Agent.resume(token)
RecoveryAction.ABORT(reason)             # raises TriageAbortError in agent.py
```

`None` kwargs are excluded from `action.params`. Access payload as `action.params["key"]`.

## Agent kwargs convention

`agent.py` injects four callbacks and recovery context into `**kwargs`:

| Key | Type | Set by |
|-----|------|--------|
| `record_step` | `(Step) -> None` | Always — injected on every call |
| `update_state` | `(dict) -> None` | Always — injected on every call |
| `record_usage` | `(Usage) -> None` | Always — feeds `max_tokens`/`max_cost_usd` |
| `record_compensator` | `(int, Callable) -> None` | Always — saga undo callables |
| `_triage_context` | `TriageContext` | All recovery actions (v0.4+) |
| `_triage_hint` | `str` | RETRY, REPLAN, ROLLBACK (backward compat) |
| `_triage_subgoal` | `str` | RESUME (backward compat) |
| `_triage_state` | `dict` | ROLLBACK (only when checkpoint state is non-empty; backward compat) |

`_triage_context` is the canonical form — a typed `TriageContext(failure_type, attempt_number, hint, subgoal, state)`. The individual `_triage_*` kwargs remain for backward compatibility and will not be removed.

Alternatively, use `triage.get_recorder()`, `triage.get_state_updater()`,
`triage.get_usage_recorder()`, or `triage.get_compensator_recorder()` (all backed by
`contextvars`) inside the agent body to avoid signature changes.

Wrapped functions should accept `**kwargs` and check for these keys.

## Adapter pattern

Each adapter in `triage/adapters/` exposes one public function:

```python
wrap_<name>(agent_obj, policy, **kwargs) -> triage.Agent
```

`**kwargs` are passed through to `Agent.__init__` (classifier, checkpoint_store,
max_recovery_attempts, auto_checkpoint). Framework imports are lazy — inside a
`try/except ImportError` — so the adapter module only raises at import time if
the optional dep is missing.

## Optional extras

| Extra | Installs | Enables |
|-------|----------|---------|
| `triage-agent[anthropic]` | `anthropic>=0.25` | `LLMClassifier` |
| `triage-agent[sqlite]` | `aiosqlite>=0.19` | `SQLiteCheckpointStore` |
| `triage-agent[redis]` | `redis[asyncio]>=5.0` | `RedisCheckpointStore` |
| `triage-agent[langgraph]` | `langgraph>=0.2` | `wrap_langgraph` |
| `triage-agent[langchain]` | `langchain-core>=0.1`, `langchain>=0.1` | `wrap_langchain` |
| `triage-agent[yaml]` | `pyyaml>=6.0` | `FailurePolicy.from_yaml()` with `.yaml`/`.yml` files |

## Public API stability (v0.2)

Stable: `FailureType` members + values, `Step`/`FailureContext` field names,
`RecoveryAction` constructor names + kwarg names, `FailurePolicy` field names,
`Classifier.classify()` signature, `Agent.__init__` arg names,
`CheckpointStore` method signatures, adapter `wrap_*` function signatures.

Internal (may change): `FailurePolicy._FIELD_MAP`, `RecoveryAction.params` layout,
`Agent._record_step`, `Agent._trajectory`, `Agent._pending_checkpoints`,
`InMemoryCheckpointStore` internals, checkpoint serialization format.

## Design decisions worth not re-litigating

Per-release notes live in `CHANGELOG.md` — don't duplicate them here. This section keeps
only the decisions whose *rationale* isn't recoverable from the code or the changelog,
because each one looks like a bug or an arbitrary choice until you know why.

**Concurrency state uses `ContextVar`; shared counters deliberately do not.**
Per-run state (`_trajectory`, `_current_state`, `_pending_checkpoints`, `_last_checkpoint_id`,
`_last_ctx`, `_run_id`) lives on a `_RunState` dataclass behind a per-instance `ContextVar`,
exposed via property getters so call sites read as plain attributes. `asyncio`/`anyio` copy
the context on task spawn, so concurrent `run()` calls on one `Agent` get independent
bookkeeping.

The inverse applies to anything that must be visible *across* a thread hop:
`HybridClassifier`'s LLM call counter and `CircuitBreaker`'s state are `threading.Lock`-guarded
plain attributes, **not** `ContextVar`s. A `ContextVar` mutated inside
`anyio.to_thread.run_sync()` — the dispatch path for the sync `classify()` — is invisible to
the caller once the thread call returns, which would silently make the call cap a no-op.
Regression guard: `tests/test_classifier_hybrid.py::test_cap_persists_across_calls_dispatched_via_to_thread`.

**`aclassify()` is duck-typed, not part of the `Classifier` protocol.**
`Classifier.classify()` stays required and synchronous per core.md Rule 5. `LLMClassifier` and
`HybridClassifier` additionally define `async def aclassify(trajectory, task)`;
`Agent._run_loop` checks `getattr(self._classifier, "aclassify", None)` and awaits it when
present, skipping the thread hop. Adding it to the protocol would force every custom
classifier to implement an async method it doesn't need.

**Transient-error detection avoids `isinstance`.** `_is_retryable()` in
`triage/classifier/llm.py` checks `exc.status_code` and `type(exc).__name__` against
`RateLimitError`/`APITimeoutError`/`APIConnectionError`/`InternalServerError` rather than
importing the real anthropic/openai exception classes — core.md Rule 1 forbids those imports
in core, and the string check works across both SDKs and their version churn.

**Caps enforce at the failure boundary, not preemptively.** `max_tokens`, `max_cost_usd`, and
`max_recovery_seconds` are checked *after* a failure, before the next recovery attempt. An
agent that burns budget but never raises runs to completion regardless. These are
"stop retrying once I'm over budget" guards, not hard ceilings — triage only has control at
the failure boundary. Documented in the `Agent.__init__` docstring; don't "fix" it.

**Compensation and hooks are best-effort by design.** Saga compensators run in reverse
step-index order before checkpoint restore; a raising compensator is logged
(`compensator_error`) and optionally surfaced via `on_compensator_error`, but never aborts
recovery — the checkpoint restore always proceeds. Lifecycle hook exceptions are likewise
swallowed with `logger.warning`. A hook or an undo callable must not be able to break the
recovery path.

**Removed features stay removed.** `HALLUCINATED_STATE`/`GOAL_DRIFT` (v0.7) had no
LLMClassifier disambiguation logic and were indistinguishable from `CONSTRAINT_IGNORED`
without a labeled corpus. The `crewai` adapter patched an internal `step_callback` that broke
silently across versions; `openai_agents` tracked a pre-stable SDK. `docs/adapters/crewai.md`
and `openai-agents.md` remain as redirect pages showing the plain-callable pattern instead.

**Informational flags are not enforced.** `Step.idempotent`, `Step.partial`, and `Step.index`
are caller-supplied metadata for strategies and hooks to inspect — `agent.py` does not act on
them. The one exception is opt-in: `Agent(strict_idempotency=True)` escalates rather than
retrying when a non-idempotent step is in the trajectory. `Trajectory.append()` warns
(`non_monotonic_step_index`) on a non-increasing index but still appends.

## Classifier accuracy measurement

Corpora live in `tests/data/error_corpus_{a,b,c}.json`; `scripts/classifier_accuracy.py`
scores all blocks. As of v1.0: corpora A and B are **training data** (their misses guided the
v0.25/v0.26 pattern fixes), so their 100%/90% scores prove nothing about generalization.
Corpus C is the real number — scored once, `rules.py` untouched: **52% recall, 100% precision**,
all 13 misses returning `UNKNOWN` with zero misroutes.

Keep corpus C frozen. Tuning `rules.py` against C's misses turns it into training data and
the measurement disappears — generate corpus D for the next improvement cycle instead. When
quoting accuracy anywhere, quote the held-out number and label the training ones as training.
