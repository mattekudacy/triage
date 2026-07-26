# Changelog

All notable changes to `triage-agent` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.21.0] - 2026-07-26

### Added
- **`Agent.stream()`** — new method for async-generator callables (functions that `yield`).
  On failure, classifies the exception, dispatches a recovery action per the policy, yields
  a `StreamRetryEvent`, then re-starts the generator with updated kwargs. The caller discards
  accumulated output on receiving a `StreamRetryEvent`. All existing caps
  (`max_recovery_attempts`, `max_total_attempts`, `max_recovery_seconds`, `max_tokens`,
  `max_cost_usd`), lifecycle hooks (`on_failure`, `on_recovery`, `on_escalate`), OTel spans,
  metrics, and circuit-breaker success-signalling work unchanged.
- **`StreamRetryEvent` dataclass** — new `triage/streaming.py`: `StreamRetryEvent(attempt,
  failure_type, action_kind, hint)` yielded by `Agent.stream()` at each retry boundary.
  Exported from `triage.__all__`.
- **`Step.partial: bool = False`** — informational flag callers can set on steps that are
  still in-progress mid-stream. Not enforced by triage; available for strategies and hooks
  that inspect `ctx.trajectory`.
- **Type guard between `run()` and `stream()`** — calling `Agent.run()` on an async-generator
  callable raises `TypeError("use agent.stream()")`, and calling `Agent.stream()` on a plain
  coroutine callable raises `TypeError("use agent.run()")`. Detection uses
  `inspect.isasyncgenfunction()` on both the callable and its `__call__` method.

---

## [0.20.0] - 2026-07-25

### Added
- **Persistent circuit breaker state** — new `triage/breaker_store.py`: `BreakerStore` protocol
  (sync, thread-safe, 6 methods) and `RedisBreakerStore` implementation backed by a synchronous
  `redis.Redis` client. `CircuitBreaker` gains an optional `store: BreakerStore | None = None`
  constructor param. When set, all state reads/writes (failure window, OPEN/HALF_OPEN state,
  `opened_at`, probe-in-flight flag) go through the store so the breaker is shared across
  workers and survives process restarts. The in-memory path (no store) is completely unchanged.
- **Wall-clock timestamps on the store path** — `time.monotonic()` is process-local and cannot
  be compared across machines. When `store` is attached, `CircuitBreaker` automatically switches
  to `time.time()` (wall-clock UTC seconds) for all timestamps. The `_now` injectable on all
  public methods continues to work; just pass wall-clock floats when a store is present.
- **`RedisBreakerStore` uses a sorted set for the failure window** — scores are wall-clock
  timestamps; `evict_before()` issues `ZREMRANGEBYSCORE` and `add_failure()` issues `ZADD`
  with a UUID-suffixed member to avoid score collisions. `save()` pipelines the state, `opened_at`,
  and probe-in-flight key writes atomically. Optional `ttl_seconds` param auto-expires all keys
  after each write — useful in serverless environments.
- `BreakerStore`, `RedisBreakerStore` added to `triage.__all__` (lazy-imported via `__getattr__`).

---

## [0.19.0] - 2026-07-23

### Added
- **Native sync-agent support** — `Agent` now accepts plain `def` callables alongside `async def`. Sync functions are run via `anyio.to_thread.run_sync()` so the event loop is never blocked; the full policy loop, classification, checkpointing, lifecycle hooks, and ContextVar injection (`get_recorder()`, `get_state_updater()`, `get_usage_recorder()`) work unchanged. Detection uses `inspect.iscoroutinefunction()` on both the callable and its `__call__` method, so callable class instances with `async __call__` are correctly identified as async. The `@agent` decorator accepts sync functions too.

---

## [0.18.0] - 2026-07-23

### Added
- **Failure distribution example** — `examples/failure_distribution.py` demonstrates aggregating triage's OTel spans into a per-type frequency table across a run population. Reads `triage.classify` span attributes (`triage.failure_type`) and `triage.run` span status codes to compute input counts, failure-type mix, and recovery rate. No new library code — uses only the span attributes shipped in v0.14.
- **Failure distribution docs** — `docs/examples/failure-distribution.md` explains the output columns, recovery outcome detection logic, and how to adapt the pattern to a production OTel backend (Jaeger, Tempo, Honeycomb, etc.). Cross-reference added to the Observability section of `docs/api/agent.md`.

### Fixed
- **OTel SDK compatibility** — `resolve_meter()` now recognises `_ProxyMeterProvider` (the internal name used by newer `opentelemetry-sdk` releases) as a no-op provider, matching the existing `ProxyMeterProvider` guard. OTel observability tests updated to pass `tracer=` / `meter=` explicitly to `Agent()` instead of relying on `set_tracer_provider` / `set_meter_provider`, which newer SDK versions treat as one-shot and warn on subsequent calls.

---

## [0.17.0] - 2026-07-22

### Added
- **Human-in-the-loop pause/resume** — `RecoveryAction.SUSPEND(message, metadata)` pauses a run instead of raising. The agent serializes a `SuspendedRun` to the `SuspensionStore` and raises `TriageSuspendedError(token, run)`. Call `agent.resume(token, action=RecoveryAction.RETRY())` (or `REPLAN`, `ABORT`, etc.) to restart from the exact suspension point. The token is single-use; the store deletes it on load.
- **`SuspensionStore` protocol** — `async save(run)` / `async load(token)` / `async delete(token)`. `InMemorySuspensionStore` is the default (not durable across restarts). Swap for a Redis-backed implementation in production.
- **`suspension_store=` on `Agent`** — accepts any `SuspensionStore` implementation; defaults to `InMemorySuspensionStore`. Copied by `clone()`.
- **`Agent.resume(token, *, action)`** — resumes a suspended run. Loads the `SuspendedRun`, deletes the token, executes the supplied action, then continues the recovery loop. Shares the same `circuit_breakers` auto-signaling and OTel span/metrics instrumentation as `run()`.
- `TriageSuspendedError`, `SuspendedRun`, `SuspensionStore`, `InMemorySuspensionStore` exported from `triage.__all__`.

### Fixed
- **HALF_OPEN single-probe guarantee** — `CircuitBreaker.allow_request()` now sets `_probe_in_flight = True` atomically in HALF_OPEN. Concurrent callers receive `False` until the probe records its outcome via `record_failure()` or `record_success()`. All three clearing paths (`record_failure`, `record_success`, `reset`) clear the flag.
- **`circuit_breaker()` HALF_OPEN semantics** — `record_failure()` is now called *after* the inner strategy returns, not before. This preserves correct probe semantics: the failure is counted, but the inner strategy runs its full logic before the breaker re-opens.
- **`on_escalate`, `circuit_breakers`, OTel metrics** — added in this release cycle (see below).

### Added (same release cycle, shipping together)
- **`on_escalate` hook** — `Agent(on_escalate=async def(ctx) -> RecoveryAction | None)`: called just before `TriageEscalationError` is raised; return an action to override, `None` to proceed with escalation.
- **`circuit_breakers=` on `Agent`** — list of `CircuitBreaker` instances; `record_success()` is called on all of them after a clean `run()` return, closing HALF_OPEN breakers automatically.
- **OTel metrics** — `triage/observability/metrics.py`: five instruments (`triage.runs`, `triage.failures`, `triage.recoveries`, `triage.run.duration`, `triage.recovery.attempts`). Same lazy-import and auto-detect pattern as OTel spans. Pass `Agent(meter=...)` to override.
- **ROADMAP.md** — published at the repo root.

---

## [0.16.0] - 2026-07-21

### Added
- **Circuit breaker** — new `triage/breaker.py`: `CircuitBreaker` with CLOSED → OPEN → HALF_OPEN states, a sliding failure-count window (`window_seconds`), and a cooldown period (`cooldown_seconds`). State is shared across runs (plain `threading.Lock`-guarded, not a `ContextVar` — deliberate, since cross-run state is the point). `record_failure()` and `record_success()` are injectable with a `_now` parameter for deterministic testing without real-clock dependency.
- **`circuit_breaker()` strategy** — new `triage/strategies/circuit_breaker.py`: `circuit_breaker(breaker, inner, open_action="escalate")` factory. When the breaker is OPEN, returns `ESCALATE` (or `ABORT` if `open_action="abort"`) immediately without calling the inner strategy. When CLOSED or HALF_OPEN, records the failure and delegates to the inner strategy. Composes with all existing strategies — no new `RecoveryAction` kinds needed.
- `CircuitBreaker`, `BreakerState` added to `triage.__all__`.

### Fixed
- **GitHub Release notes** — `release.yml` now extracts the relevant `CHANGELOG.md` section for the tagged version and uses it as the release body, instead of a static "See CHANGELOG.md" placeholder.

---

## [0.15.1] - 2026-07-21

### Fixed
- Patch release to validate the automated release pipeline (autotag → release → PyPI publish). No functional changes from 0.15.0.

---

## [0.15.0] - 2026-07-21

### Added
- **Cost/token budgets** — new `triage/usage.py`: `Usage` dataclass (`input_tokens`, `output_tokens`, `cost_usd`, `calls`) and a thread-safe `UsageMeter`. `Agent.__init__` gains `max_tokens: int | None` and `max_cost_usd: float | None`; when exceeded, triage raises `TriageEscalationError` before the next recovery attempt (same pattern as `max_recovery_seconds`). Meter resets at the start of every `run()` and is isolated per concurrent run via `_RunState`.
- **`record_usage` injection** — injected into the wrapped fn alongside `record_step`/`update_state`; `triage.get_usage_recorder()` provides the same callback via `ContextVar` for agents that avoid signature changes.
- **`LLMClassifier` auto-reports usage** — `_call_sync()` and `_call_async()` duck-type `.usage` on the API response and push token counts to the run meter automatically. Covers Anthropic (`input_tokens`/`output_tokens`) and OpenAI-compatible (`prompt_tokens`/`completion_tokens`) backends. Silently skipped for backends that don't expose `.usage`.
- **HybridClassifier accuracy benchmarks** — `examples/benchmark.py` gains `SEMANTIC_CASES` (3× `PLAN_INCOMPLETE` + 2× `CONTEXT_OVERFLOW`) and `--hybrid` flag. `print_summary` bug fixed (previously always re-ran `RulesClassifier` regardless of which classifier was under test). `docs/concepts/classifiers.md` updated with accuracy tables for `LLMClassifier` and `HybridClassifier`.
- `Usage`, `UsageMeter`, `get_usage_recorder` added to `triage.__all__`.

---

## [0.14.0] - 2026-07-16

### Added
- **OpenTelemetry spans** — new `triage/observability/otel.py` package. All OTel imports are lazy so the module is safe to import without `opentelemetry-sdk` installed. Three spans per `run()`: `triage.run` (root, carries `triage.run_id` + `triage.task`), `triage.classify` (per classification, carries `triage.failure_type`), `triage.dispatch` (per strategy dispatch, carries `triage.action_kind`, `triage.failure_type`, `triage.attempt`; escalate/abort set status = ERROR). `Agent.__init__` gains `tracer=None` — pass an explicit `Tracer` or let triage auto-detect a configured global provider. `clone()` copies the tracer. Spans are additive; the six existing structured log events are unchanged. Install with `pip install triage-agent[otel]`.
- `examples/policy_sequence.py` — demonstrates `FailurePolicy.sequence()` with two scenarios: EXTERNAL_FAULT (retry → replan) and LOOP_DETECTED (replan → rollback).
- `examples/otel_tracing.py` — demonstrates OTel auto-detect and explicit-tracer modes with an in-memory span exporter.

### Fixed
- **CI now installs the actual package** — `pip install -e ".[dev,langgraph,langchain,sqlite,redis]"` replaces the hand-listed dependency list. Adapter test files (`test_adapter_langgraph.py`, `test_adapter_langchain.py`) now run in CI instead of silently skipping.
- **Single version source of truth** — `pyproject.toml` now uses `dynamic = ["version"]`; hatchling reads the version exclusively from `triage/__init__.py`. The previously redundant static `version =` field is removed.
- **Ruff violations fixed** — 50 violations resolved: `B023` loop-variable capture in `bench.py` (real bug: `on_recovery` closure now binds `failure_types` via default arg), import modernisation (`UP035`, `UP037`), import ordering (`I001`), line-length (`E501`). `ruff check triage/` now passes with zero findings and is enforced in CI.
- CI gains a `lint` job (`ruff check triage/`) and a `package` job (`python -m build && twine check dist/*`).

---

## [0.13.0]

### Added
- **Run-scoped checkpoints** — `Checkpoint` gains `run_id: str | None = None`. `Agent.run()` generates a UUID per call; rollback calls `latest(run_id=...)` so each concurrent run rolls back only to its own checkpoints. All three stores implement scoped filtering; SQLite adds a migration guard. Closes the concurrent-rollback footgun documented since v0.10.
- **`FailurePolicy.sequence(*strategies)`** — steps through strategies in order across successive failures of the same type, derived from `ctx.attempt_history` (no external state needed, safe for concurrent runs). Escalates once all strategies are exhausted. Raises `ValueError` if called with no strategies.

### Changed
- `CheckpointStore.latest()` signature updated to `latest(run_id=None)` — `None` preserves the existing global-latest behavior.
- `known-limitations.md` cleaned up: removed three stale "planned for v0.4" entries that had shipped in v0.4.

---

## [0.12.0]

### Added
- **Fuzzy loop detection** — `RulesClassifier(loop_similarity_threshold=0.85)` matches steps whose `tool_input` JSON strings are *similar*, not just identical, via `difflib.SequenceMatcher` (no new dependency). Comparison is consecutive within the window. `tool_called` still must match exactly.

---

## [0.11.0]

### Added
- **`InMemoryCheckpointStore` concurrency-safe** — `save()`/`load()`/`latest()` now guard the internal dict with an `anyio.Lock`.
- **`LLMClassifier`/`HybridClassifier` retry on transient errors** — `max_retries` (default 1) and `retry_backoff_base` (default 0.5s) constructor params. Retries fire for HTTP 429/500/502/503/529 or exception class names `RateLimitError`/`APITimeoutError`/`APIConnectionError`/`InternalServerError`.
- **`HybridClassifier(max_llm_calls_per_run=N)`** — caps LLM calls within one `Agent.run()` call; reset at the start of each run via `reset_call_count()`.
- **`Trajectory.append()` warns on non-monotonic `Step.index`** — structured `non_monotonic_step_index` warning; does not raise.
- **`corrections.jsonl` rotation** — `record_correction(..., max_lines=10_000)` rotates the file once it exceeds the threshold.

---

## [0.10.0]

### Added
- **Concurrency-safe `Agent`** — per-run state moved onto `_RunState` held behind a `ContextVar`. Two concurrent `run()` calls on the same `Agent` instance get independent trajectory/state/checkpoint bookkeeping. `Agent.clone()` still exists for independent hooks or stores.
- **Native-async classification** — `LLMClassifier` and `HybridClassifier` gain optional `async def aclassify(trajectory, task)` using native async SDK clients. `Agent._run_loop` detects and awaits `aclassify` directly, skipping the `anyio.to_thread.run_sync()` hop.

---

## [0.9.0]

### Added
- **Per-framework error patterns** — `RulesClassifier(framework="openai"|"anthropic"|"langgraph")` activates supplemental SDK-specific patterns for WRONG_TOOL_CALLED, SCHEMA_MISMATCH, and EXTERNAL_FAULT. Unknown framework values are silently ignored.
- **Step risk scoring** — `RulesRiskScorer` scores each recorded step; `Agent(risk_scorer=..., risk_threshold=0.9)` raises `TriageAbortError` before executing a step at or above the threshold. `StepRiskScorer` protocol and `RiskScore` dataclass are the public extension points.

---

## [0.8.0]

### Added
- **`strict_idempotency=True`** — escalates instead of retrying when any `step.idempotent=False` is in the trajectory.
- **`max_recovery_seconds`** — wall-clock budget cap for the recovery process.
- **Structured event logs** — all `logger` calls now pass `extra={"triage_event": ...}` dicts for structured log routing.
- **`triage.feedback` module** — `Correction`, `record_correction()`, `load_corrections()`, `Agent.report_misclassification()`.
- **`RulesClassifier.fit(corrections_path)`** — reviews misclassification coverage from a corrections file.
- **`FailurePolicy.from_yaml(path)`** — loads policy from `.toml` (stdlib) or `.yaml`/`.yml` (optional pyyaml).
- **Bench baseline comparison** — `run_benchmark(baseline_fn=...)` runs a raw agent alongside the triage-wrapped one; `BenchReport.compare()` returns a side-by-side table.
- **Multi-agent context propagation** — `_run_loop` reuses a chained child's `failure_type` instead of re-classifying.

---

## [0.7.0]

### Changed
- **Taxonomy trimmed to 9 types** — `HALLUCINATED_STATE` and `GOAL_DRIFT` removed (indistinguishable from `CONSTRAINT_IGNORED` without a labeled corpus).
- **Adapters cut to 2** — `crewai` and `openai_agents` adapters removed; `langgraph` and `langchain` remain.

### Added
- **`Step.idempotent: bool = True`** — informational flag for strategies that need to reason about side effects.
- **`triage.bench`** — `run_benchmark()` eval harness with `BenchReport`.

---

## [0.6.0]

### Added
- **`TIMEOUT` failure type** — `FailureType.TIMEOUT = "timeout"`, detected by `RulesClassifier` via `_TIMEOUT_RE`.
- **Lifecycle hooks** — `on_step`, `on_failure`, `on_recovery` sync callbacks on `Agent.__init__`; hook exceptions are swallowed; hooks copied by `clone()`.

---

## [0.5.0]

### Added
- **`py.typed` marker** — PEP 561 compliance; mypy/pyright now type-check triage in strict mode.
- **`CancelledError` propagation** — `_run_loop` uses `except BaseException` with re-raise for non-`Exception` subclasses.
- **`triage.testing` module** — `make_step()`, `RecordingAgent`, `assert_classifies_as()` test utilities.

---

## [0.4.0]

### Added
- **`max_total_attempts`** — global cross-type attempt cap on `Agent.__init__`.
- **`Agent.clone()`** — fresh per-run state, independent hooks, shared policy/classifier/store.
- **`FailurePolicy.chain(primary, fallback, after_kinds)`** — two-strategy fall-through factory.
- **`contextvars` injection** — `triage.get_recorder()` and `triage.get_state_updater()`.
- **`TriageContext` dataclass** — typed recovery context injected as `_triage_context` alongside legacy scalar kwargs.
- **`RulesClassifier` improvements** — configurable `loop_window`, expanded tool/schema/fault patterns.

---

## [0.3.0]

### Added
- **Async `LLMClassifier`** — `classify()` runs via `anyio.to_thread.run_sync`; event loop never blocked.
- **Real agent state in checkpoints** — `update_state(dict)` injected alongside `record_step`; state persisted and restored as `_triage_state` on ROLLBACK.
- **`HybridClassifier`** — rules first, LLM only on UNKNOWN.
- **BYOK env vars** — `TRIAGE_LLM_BASE_URL`, `TRIAGE_LLM_MODEL`, `TRIAGE_LLM_API_KEY`.
- **`attempt_history` on `FailureContext`** — `list[tuple[FailureType, str]]` of prior attempts.
