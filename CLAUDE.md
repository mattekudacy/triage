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
  policy.py               — RecoveryAction (6 constructors), FailurePolicy dataclass
  agent.py                — Agent class, TriageEscalationError, TriageAbortError, @agent decorator
  classifier/
    base.py               — Classifier protocol (runtime_checkable)
    rules.py              — RulesClassifier — 6 rules in priority order, sync, zero API calls
    llm.py                — LLMClassifier — semantic classifier; Anthropic or OpenAI-compatible backend
    hybrid.py             — HybridClassifier — rules first, LLMClassifier fallback on UNKNOWN
  strategies/
    retry.py              — retry_with_tool_manifest(), backoff_and_retry()
    replan.py             — replan(), resume_from_subgoal()
    rollback.py           — rollback_to_checkpoint()
  adapters/
    langgraph.py          — wrap_langgraph() — wraps a compiled LangGraph StateGraph
    langchain.py          — wrap_langchain() — wraps a LangChain AgentExecutor
  observability/
    otel.py               — resolve_tracer(), run_span/classify_span/dispatch_span context managers (lazy OTel import)
  bench.py                — run_benchmark(), BenchReport, BenchResult — eval harness with baseline comparison
  feedback.py             — Correction, record_correction(), load_corrections() — misclassification feedback loop
  scorer/
    base.py             — RiskScore dataclass, StepRiskScorer protocol (sync, no API calls)
    rules.py            — RulesRiskScorer — destructive pattern detection, zero API calls
tests/                    — pytest-asyncio, asyncio_mode=auto, 283+ tests (2 skipped without optional deps)
examples/
  raw_openai.py           — end-to-end demo, needs OPENAI_API_KEY
```

## Running tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -x --tb=short
```

The venv uses Python 3.13 (`/opt/homebrew/bin/python3.13`). Install deps with:
```bash
.venv/bin/pip install anyio pydantic pytest pytest-asyncio anthropic aiosqlite fakeredis
```

`pip install -e .` now works cleanly (verified 2026-07-11 with pip 26.0.1) — the
hatchling shadow issue that previously blocked it is resolved. `PYTHONPATH=.` still
works too and remains the faster path for quick iteration without reinstalling.

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
RecoveryAction.ABORT(reason)             # raises TriageAbortError in agent.py
```

`None` kwargs are excluded from `action.params`. Access payload as `action.params["key"]`.

## Agent kwargs convention

`agent.py` injects two callbacks and recovery context into `**kwargs`:

| Key | Type | Set by |
|-----|------|--------|
| `record_step` | `(Step) -> None` | Always — injected on every call |
| `update_state` | `(dict) -> None` | Always — injected on every call |
| `_triage_context` | `TriageContext` | All recovery actions (v0.4+) |
| `_triage_hint` | `str` | RETRY, REPLAN, ROLLBACK (backward compat) |
| `_triage_subgoal` | `str` | RESUME (backward compat) |
| `_triage_state` | `dict` | ROLLBACK (only when checkpoint state is non-empty; backward compat) |

`_triage_context` is the canonical form — a typed `TriageContext(failure_type, attempt_number, hint, subgoal, state)`. The individual `_triage_*` kwargs remain for backward compatibility and will not be removed.

Alternatively, use `triage.get_recorder()` and `triage.get_state_updater()` (backed by `contextvars`) inside the agent body to avoid signature changes.

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

## v0.3 changes (shipped)

- **Async `LLMClassifier`** — `classify()` runs via `anyio.to_thread.run_sync`; event loop never blocked.
- **Real agent state in checkpoints** — `update_state(dict)` injected alongside `record_step`; state persisted in every checkpoint; restored as `_triage_state` on ROLLBACK.
- **`HybridClassifier`** — rules first, LLM only on UNKNOWN. See `triage/classifier/hybrid.py`.
- **BYOK env vars** — `TRIAGE_LLM_BASE_URL`, `TRIAGE_LLM_MODEL`, `TRIAGE_LLM_API_KEY` read by `LLMClassifier.__init__` when no explicit arg supplied.
- **`attempt_history` on `FailureContext`** — `list[tuple[FailureType, str]]` of `(failure_type, action_kind)` from all prior attempts in the current `run()` call.

## v0.4 changes (shipped)

- **`max_total_attempts`** — global cross-type attempt cap on `Agent.__init__`; fires before `max_recovery_attempts` when `len(attempt_history)` reaches the limit.
- **`Agent.clone()`** — returns a new `Agent` sharing policy/classifier/store but with fresh per-run state; required for concurrent `run()` calls across tasks.
- **`FailurePolicy.chain(primary, fallback, after_kinds)`** — static factory returning a strategy that falls through to `fallback` when `primary` returns an action whose `kind` is in `after_kinds` (default `"escalate"`).
- **`contextvars` injection** — `triage.get_recorder()` and `triage.get_state_updater()` read from `ContextVar`s set by `run()`; allows agent functions to avoid signature changes.
- **`TriageContext` dataclass** — replaces scattered `_triage_hint`/`_triage_subgoal`/`_triage_state` kwargs with a single typed `_triage_context` object; individual kwargs remain for backward compat.
- **RulesClassifier improvements** — configurable `loop_window`, expanded `_WRONG_TOOL_RE` (OpenAI/Anthropic/generic), word-boundary `_EXTERNAL_CODE_RE`, `invalid json`/`unexpected token` in `_SCHEMA_RE`.

## v0.5 changes (shipped)

- **`py.typed` marker** — PEP 561 marker file; mypy/pyright now type-check triage in strict codebases.
- **`CancelledError` propagation** — `_run_loop` now uses `except BaseException` with explicit re-raise for non-`Exception` subclasses; `_drain_checkpoints` uses swap-and-clear so `_pending_checkpoints` is atomically emptied before iteration.
- **`triage.testing` module** — public `make_step()`, `RecordingAgent`, and `assert_classifies_as()` utilities for testing triage-wrapped agents without hand-rolling fixtures.

## v0.6 changes (shipped)

- **`TIMEOUT` failure type** — `FailureType.TIMEOUT = "timeout"` added before `UNKNOWN`; `RulesClassifier` detects it via `_TIMEOUT_RE` (matches `timeout`, `timed out`, `deadline exceeded`, `time limit`); `FailurePolicy.TIMEOUT` field added.
- **Lifecycle hooks** — `on_step`, `on_failure`, `on_recovery` sync callbacks on `Agent.__init__`; exceptions from hooks are swallowed with `logger.warning`; hooks are copied by `clone()`; `on_recovery(ctx, action)` signature.

## v0.7 changes (shipped)

- **Taxonomy trimmed to 9 types** — `HALLUCINATED_STATE` and `GOAL_DRIFT` removed; LLMClassifier had no prompt logic to distinguish them from `CONSTRAINT_IGNORED` without a labeled corpus.
- **Adapters cut to 2** — `crewai` (patches internal `step_callback`) and `openai_agents` (SDK still pre-stable) removed; `langgraph` and `langchain` remain.
- **`Step.idempotent: bool = True`** — informational flag; strategies can inspect `ctx.trajectory` before recommending retry on steps that sent email/wrote DB; not auto-enforced by `agent.py`.
- **`triage.bench`** — `run_benchmark(agent_fn, tasks, policy)` eval harness; returns `BenchReport` with `success_rate`, `mean_latency_s`, `total_recoveries`, and `summary()`; uses `on_recovery` hook, no agent.py changes needed.

## v0.8 changes (shipped)

- **`strict_idempotency=True` on `Agent`** — escalates instead of retrying when any `step.idempotent=False` is in the trajectory; safe default for agents that send emails, charge cards, or write to external systems.
- **`max_recovery_seconds: float | None` on `Agent`** — wall-clock budget cap; timer starts on first failure; subsequent failures that exceed the cap raise `TriageEscalationError`; copied by `clone()`.
- **Structured event logs** — all five `logger` calls now pass `extra={"triage_event": ...}` dicts; keys: `failure_classified`, `action_dispatched`, `retry_backoff`, `attempt_start`, `hook_error`; no new deps.
- **`triage.feedback` module** — `Correction` dataclass + `record_correction(ctx, expected_type)` appends JSONL corrections; `load_corrections()` reads them back; `Agent.report_misclassification(expected_type)` convenience method uses `self._last_ctx`.
- **`RulesClassifier.fit(corrections_path)`** — reads `corrections.jsonl`, re-classifies each entry, logs structured `fit_misclassification` warnings where predicted ≠ expected; returns `{failure_type: {"correct": N, "wrong": N}}` coverage dict.
- **`FailurePolicy.from_yaml(path)`** — loads policy from `.toml` (stdlib `tomllib`) or `.yaml`/`.yml` (optional `pyyaml`); built-in strategy registry: `backoff_and_retry`, `retry_with_tool_manifest`, `replan`, `resume_from_subgoal`, `rollback_to_checkpoint`, `escalate`, `abort`; `strategy_registry` param for custom strategies.
- **Bench baseline comparison** — `run_benchmark(baseline_fn=...)` runs a raw (no-triage) callable alongside the triage-wrapped agent; `BenchReport.compare()` returns a side-by-side table; `baseline_results` list on `BenchReport`.
- **Multi-agent context propagation** — `_run_loop` checks `exc.__cause__`/`exc.__context__` for chained `TriageEscalationError`; reuses child's `failure_type` instead of re-classifying; outer policy decides whether to re-route or escalate.

## v0.9 changes (shipped)

- **Per-framework error pattern tables on `RulesClassifier`** — `RulesClassifier(framework="openai"|"anthropic"|"langgraph")` activates supplemental per-SDK patterns for WRONG_TOOL_CALLED, SCHEMA_MISMATCH, and EXTERNAL_FAULT; generic patterns unchanged; unknown framework values silently ignored (generic still applies); no protocol or agent.py changes.
- **Step risk scoring** — `RulesRiskScorer` scores each step as it's recorded; `Agent(risk_scorer=..., risk_threshold=0.9)` raises `TriageAbortError` before the agent executes a step scoring >= threshold; `RulesRiskScorer` detects destructive patterns (email, payment, DELETE, DROP TABLE) with zero API calls; `StepRiskScorer` protocol and `RiskScore` dataclass are the public extension points.

## v0.10 changes (shipped)

- **Concurrency-safe `Agent`** — per-run state (`_trajectory`, `_current_state`,
  `_pending_checkpoints`, `_last_checkpoint_id`, `_last_ctx`) moved off plain instance
  attributes onto a `_RunState` dataclass held behind a per-instance `ContextVar`
  (`self._run_state_var`), exposed via property getters/setters so every existing
  `self._trajectory = ...`-style call site in `agent.py` is unchanged. Because
  `asyncio`/`anyio` copy the current `contextvars.Context` on `Task` spawn, two
  concurrent `run()` calls on the *same* `Agent` instance now get independent
  trajectory/state/checkpoint bookkeeping instead of clobbering a shared attribute.
  `Agent.clone()` is unchanged and still recommended when you want independent
  lifecycle hooks or checkpoint stores — it's just no longer required purely for
  concurrency safety. See `tests/test_agent.py::test_concurrent_runs_*`.
- **`pip install -e .` confirmed fixed** — the hatchling shadow issue previously
  documented was already resolved upstream (pip 26.0.1); verified clean install +
  full test pass with no `PYTHONPATH` from an unrelated working directory. `CLAUDE.md`
  "Running tests" section updated to drop the workaround note (`PYTHONPATH=.` still
  works and remains the faster iteration path).
- **Native-async classification** — `LLMClassifier` and `HybridClassifier` gained an
  optional `async def aclassify(trajectory, task) -> FailureType` method using the
  native async Anthropic/OpenAI SDK client (`AsyncAnthropic`/`AsyncOpenAI`, built and
  cached separately from the sync client via `_build_async_client()`/`_get_async_client()`
  behind an `anyio.Lock`). `Classifier.classify()` stays required and synchronous per
  core.md Rule 5 — `aclassify` is purely additive and duck-typed, not part of the
  `Classifier` Protocol. `Agent._run_loop` checks `getattr(self._classifier, "aclassify", None)`
  and awaits it directly when present, skipping the `anyio.to_thread.run_sync()` hop;
  classifiers without `aclassify` (e.g. `RulesClassifier`) are unaffected and keep using
  the thread-based path. `HybridClassifier.aclassify()` runs rules first (unchanged) and,
  on UNKNOWN, calls `self._llm.aclassify()` if the wrapped LLM classifier defines it,
  else falls back to `self._llm.classify()`.

## v0.11 changes (shipped)

- **`InMemoryCheckpointStore` is concurrency-safe** — `save()`/`load()`/`latest()` now
  guard the internal dict with an `anyio.Lock`. Previously last-write-wins; now safe
  to share across the concurrent `Agent.run()` calls that v0.10 made possible. Note
  this only serializes *storage access* — it does not coordinate "which checkpoint is
  latest" semantics across concurrent writers with different rollback intentions.
- **`LLMClassifier`/`HybridClassifier` retry on transient errors** — `classify()` and
  `aclassify()` both gained `max_retries` (default `1`) and `retry_backoff_base`
  (default `0.5`s, doubles each attempt) constructor params. Retries only fire for
  errors that look transient — HTTP `429`/`500`/`502`/`503`/`529` (checked via
  `exc.status_code`) or an exception class named `RateLimitError`/`APITimeoutError`/
  `APIConnectionError`/`InternalServerError` (checked via `type(exc).__name__`, not
  `isinstance`, so this works without importing the real anthropic/openai exception
  types). Non-transient errors fall straight to `UNKNOWN` on the first attempt, same
  as before. See `_is_retryable()` in `triage/classifier/llm.py`.
- **`HybridClassifier(max_llm_calls_per_run=N)`** — caps LLM calls within one
  `Agent.run()` call; once reached, ambiguous (rules-`UNKNOWN`) failures return
  `UNKNOWN` without touching the LLM. The counter is a `threading.Lock`-guarded plain
  instance attribute, **not** a `ContextVar` — a `ContextVar` mutated inside
  `anyio.to_thread.run_sync()` (the dispatch path for `classify()`) is invisible to
  the caller once that thread call returns, which would silently make the cap a
  no-op. `Agent.run()` calls `classifier.reset_call_count()` (duck-typed via
  `getattr`) at the start of every run, scoping the budget per run. Sharing one
  `HybridClassifier` across concurrent runs makes the reset best-effort, not strictly
  isolated per task — construct a separate instance per concurrent task for a precise
  budget. See `tests/test_classifier_hybrid.py::test_cap_persists_across_calls_dispatched_via_to_thread`
  for the regression this guards against.
- **`Trajectory.append()` warns on non-monotonic `Step.index`** — logs a
  `non_monotonic_step_index` structured warning (does not raise, does not block the
  append) when an appended step's index is `<=` the previous step's index.
  `Step.index` remains caller-supplied and informational, not auto-assigned or
  enforced — this is a diagnostic aid for a common caller bug (e.g. reusing an index
  across a retry), not a new invariant. New `tests/test_trajectory.py` covers
  `Trajectory` end-to-end (previously untested as a standalone module).
- **`corrections.jsonl` rotation** — `record_correction(..., max_lines=10_000)` (new
  param, default 10,000) rotates the corrections file once it exceeds the threshold:
  the whole file is moved to `<path>.1` (overwriting any previous backup) and a fresh
  file starts. Pass `max_lines=None` to disable rotation and grow unbounded, matching
  pre-v0.11 behavior.

## v0.12 changes (shipped)

- **Fuzzy loop detection on `RulesClassifier`** — new `loop_similarity_threshold: float | None = None`
  constructor param (range `(0.0, 1.0]`, raises `ValueError` outside that range). Default
  `None` preserves exact-match-only behavior (no change on upgrade). When set, the
  `LOOP_DETECTED` rule additionally matches steps whose canonical `tool_input` JSON strings
  are similar — not just identical — via stdlib `difflib.SequenceMatcher.ratio()` (no new
  dependency, per core.md Rule 1). Comparison is **consecutive** (each step vs. the previous
  step in the window), not all-vs-first, so a query that drifts gradually across the window
  is still caught even if the first and last steps have drifted far apart from each other.
  `tool_called` still must match exactly across the whole window — only `tool_input` gets the
  fuzzy comparison. Implemented via a new `_is_loop_window()` helper in `triage/classifier/rules.py`
  shared by both the exact and fuzzy paths. See `tests/test_classifier_rules.py`'s
  "Fuzzy loop detection" section.

## v0.13 changes (shipped)

- **Run-scoped checkpoints** — `Checkpoint` gains a `run_id: str | None = None` field.
  `Agent.run()` generates a UUID once per call and stores it in `_RunState`; all
  `auto_checkpoint` saves carry that ID. The rollback path calls `latest(run_id=self._run_id)`
  so each run rolls back only to its own most-recent checkpoint rather than the global newest.
  `CheckpointStore.latest()` signature updated to `latest(run_id=None)` — `None` preserves
  existing global behavior. All three stores (`InMemoryCheckpointStore`, `SQLiteCheckpointStore`,
  `RedisCheckpointStore`) implement scoped filtering; SQLite adds a migration guard for existing
  tables. Closes the concurrency footgun documented in `known-limitations.md` since v0.10.
- **`FailurePolicy.sequence(*strategies)`** — steps through an ordered list of strategies across
  successive failures of the same type. Position is derived from `ctx.attempt_history` (count of
  prior attempts matching `ctx.failure_type`) so no external state is needed and the sequence is
  safe to share across concurrent runs. Escalates with an informative message once all strategies
  are exhausted. Requires at least one strategy (raises `ValueError` otherwise). Replaces the
  multi-step `attempt_history`-inspection boilerplate documented as a workaround in
  `known-limitations.md`. `FailurePolicy.chain()` (v0.4, two-strategy case) is unchanged.
- **`known-limitations.md` doc cleanup** — removed three stale "planned for v0.4" entries that
  had shipped in v0.4 (`get_recorder()`/`get_state_updater()`, `TriageContext`, `max_total_attempts`).
  Concurrency and chaining sections updated to reflect v0.13 state.

## v0.16 changes (shipped)

- **Circuit breaker** — new `triage/breaker.py`: `CircuitBreaker(failure_threshold, window_seconds,
  cooldown_seconds)` dataclass with CLOSED/OPEN/HALF_OPEN states and a sliding failure window.
  State is **shared across runs** via a plain `threading.Lock`-guarded object (deliberately *not* a
  `ContextVar` — a ContextVar mutated inside `anyio.to_thread.run_sync` is invisible to the caller).
  `record_failure()` / `record_success()` accept a `_now` injectable float for deterministic testing.
  Transitions: CLOSED → OPEN when `failure_count >= failure_threshold` within `window_seconds`;
  OPEN → HALF_OPEN after `cooldown_seconds`; HALF_OPEN → CLOSED on `record_success()`;
  HALF_OPEN → OPEN on `record_failure()`.
- **`circuit_breaker()` strategy** — `triage/strategies/circuit_breaker.py`:
  `circuit_breaker(breaker, inner, open_action="escalate")` wraps any existing strategy.
  While OPEN: returns `ESCALATE`/`ABORT` immediately (no inner call). While CLOSED/HALF_OPEN:
  calls `breaker.record_failure()` then delegates to `inner`. No new `RecoveryAction` kinds.
- **GitHub Release notes extracted from CHANGELOG.md** — `release.yml` now uses `awk` to extract the
  section for the tagged version and sets it as the GitHub Release body via `body_path`.
- `CircuitBreaker`, `BreakerState` exported from `triage.__all__`.

## v0.15 changes (shipped)

- **Cost/token budgets** — new `triage/usage.py` module: `Usage` dataclass
  (`input_tokens`, `output_tokens`, `cost_usd`, `calls`) and a thread-safe `UsageMeter`
  that accumulates usage across all LLM calls in a run.
  `Agent.__init__` gains `max_tokens: int | None` and `max_cost_usd: float | None` params
  (both default `None`; copied by `clone()`). When exceeded, triage raises
  `TriageEscalationError` before the next recovery attempt — same pattern as the existing
  `max_recovery_seconds` wall-clock cap. The meter resets at the start of every `run()`.
  **Injection:** `record_usage` is injected into the wrapped fn as a keyword argument
  (alongside `record_step` / `update_state`); `triage.get_usage_recorder()` provides the
  same callback via a `ContextVar` for agents that avoid signature changes.
  **`LLMClassifier` auto-reporting:** `_call_sync()` and `_call_async()` now duck-type
  `.usage` on the response object and push `Usage(input_tokens=..., output_tokens=...)`
  to the run meter automatically, covering both Anthropic
  (`response.usage.input_tokens / output_tokens`) and OpenAI-compatible
  (`response.usage.prompt_tokens / completion_tokens`) backends. Backends that don't expose
  `.usage` are silently skipped (best-effort, never breaks classification).
  **Exports:** `Usage`, `UsageMeter`, `get_usage_recorder` added to `triage.__all__`.

## v0.14 changes (shipped)

- **OpenTelemetry spans** — new `triage/observability/otel.py` module (new package,
  follows the adapter pattern). All OTel imports are lazy behind `try/except ImportError`
  so the module is importable regardless of whether `opentelemetry-sdk` is installed.
  `resolve_tracer(explicit)` implements priority: explicit `tracer=` arg → auto-detect
  real provider via `get_tracer_provider()` (returns `None` for `ProxyTracerProvider`/
  `NoOpTracerProvider`) → `None` (no-op). `Agent.__init__` gains `tracer=None` param;
  `clone()` copies it. Three scopes emitted per `run()` call:
  - `triage.run` — root span, attributes `triage.run_id` (UUID from v0.13) + `triage.task`
  - `triage.classify` — wraps each classification; attribute `triage.failure_type`
  - `triage.dispatch` — wraps each strategy dispatch; attributes `triage.action_kind`,
    `triage.failure_type`, `triage.attempt`; escalate/abort set span status = `ERROR`
  All spans from one `run()` share the same OTel `trace_id` and `triage.run_id`.
  Spans are **additive** — the 6 existing `triage_event` structured log calls are unchanged.
  `agent.py` does not import OTel directly — it imports from `triage.observability.otel`,
  keeping core.md Rule 1 intact (`triage/observability/` is the OTel boundary).
  Install via `pip install triage-agent[otel]` (extra was already declared in `pyproject.toml`).
  Tests in `tests/test_observability_otel.py` (5 tests skipped without `opentelemetry-sdk`,
  7 tests always run to verify the no-op path).

## v0.23 changes (shipped)

- **Cost model for `max_cost_usd`** — new `triage/pricing.py` module: `PRICE_TABLE` dict mapping
  model ID prefixes to `(input_per_token_usd, output_per_token_usd)` tuples (Anthropic models only;
  rates from 2026-06-24). `lookup_cost(model, input_tokens, output_tokens) -> float` does longest-prefix
  matching so both short IDs (`"claude-haiku-4-5"`) and dated variants
  (`"claude-haiku-4-5-20251001"`) resolve correctly; unknown models return `0.0`.
  `LLMClassifier._report_usage()` now calls `lookup_cost(self._model, ...)` and passes the
  computed value as `cost_usd` into `Usage`, so `max_cost_usd` enforcement actually fires
  for Anthropic-backend classifiers. Users can patch `PRICE_TABLE` at runtime to add custom
  model rates. `tests/test_pricing.py` — 16 tests covering known models, dated variants,
  unknown fallback, and override.

## v0.24 changes (shipped)

- **Saga / compensating rollback** — new `triage/strategies/saga.py` module with
  `compensating_rollback(checkpoint_id=None)` strategy. Agents register undo callables via
  `record_compensator(step_index, fn)` (injected kwarg alongside `record_step`) or via
  `triage.agent.get_compensator_recorder()` (contextvar accessor). Both sync and async
  compensators are supported. On any `ROLLBACK` action (whether via `compensating_rollback()`
  or plain `RecoveryAction.ROLLBACK`), `agent.py` runs all registered compensators in
  **reverse step-index order** before restoring the checkpoint. Compensator errors are logged
  (`triage_event: "compensator_error"`) but never abort recovery — compensation is best-effort.
  Compensators are in-memory only (callables can't be serialized); they are cleared at the
  top of each loop iteration so retries start with an empty list.
  `get_compensator_recorder` exported from `triage.__all__`. `tests/test_saga.py` — 11 tests.

