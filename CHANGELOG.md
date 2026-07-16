# Changelog

All notable changes to `triage-agent` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
