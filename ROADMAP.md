# triage-agent Roadmap

`triage-agent` wraps async agent callables, classifies failures by type, and routes each
type to a recovery strategy. This document tracks what has shipped and what comes next.

---

## Done (v0.1–v0.16)

| Version | Feature | Description |
|---------|---------|-------------|
| v0.1 | Initial release | Core taxonomy (9 failure types), trajectory, checkpoint, RulesClassifier, policy, and agent |
| v0.2 | Public API stability | Stable FailureType / RecoveryAction / FailurePolicy / Agent / CheckpointStore signatures |
| v0.3 | HybridClassifier + state | HybridClassifier (rules → LLM fallback); agent state persisted in checkpoints; async LLMClassifier |
| v0.4 | Recovery control | `max_total_attempts`, `Agent.clone()`, `FailurePolicy.chain()`, `TriageContext`, `contextvars` injection |
| v0.5 | Type safety + testing | PEP 561 `py.typed` marker; `CancelledError` propagation; `triage.testing` utilities |
| v0.6 | TIMEOUT type + hooks | `TIMEOUT` failure type with `RulesClassifier` detection; `on_step` / `on_failure` / `on_recovery` lifecycle hooks |
| v0.7 | Taxonomy cleanup + bench | Removed two ambiguous failure types; `triage.bench` eval harness; `Step.idempotent` flag |
| v0.8 | Safety + feedback | `strict_idempotency`, `max_recovery_seconds`, structured event logs, misclassification feedback loop, YAML/TOML policy loading |
| v0.9 | Framework patterns + risk scoring | Per-SDK error patterns (`openai` / `anthropic` / `langgraph`); step-level risk scoring with abort threshold |
| v0.10 | Concurrent runs | Per-run state via `ContextVar`; two concurrent `run()` calls on the same `Agent` instance are now safe |
| v0.11 | Resilience improvements | Concurrency-safe `InMemoryCheckpointStore`; LLM retry on transient errors; `HybridClassifier` LLM call cap per run |
| v0.12 | Fuzzy loop detection | Loop detection via `difflib.SequenceMatcher`; catches gradual query drift without new dependencies |
| v0.13 | Run-scoped checkpoints + sequence | Checkpoints carry `run_id`; rollback stays within a run; `FailurePolicy.sequence()` for ordered strategy escalation |
| v0.14 | OpenTelemetry spans | `triage.run` / `triage.classify` / `triage.dispatch` spans; lazy OTel import; `tracer=` on `Agent` |
| v0.15 | Cost/token budgets | `max_tokens` / `max_cost_usd` caps; `UsageMeter`; `LLMClassifier` auto-reports token usage |
| v0.16 | Circuit breaker | `CircuitBreaker` with CLOSED / OPEN / HALF_OPEN states; `circuit_breaker()` strategy wrapper |
| v0.17 | Human-in-the-loop pause/resume | `RecoveryAction.SUSPEND`; `SuspensionStore` protocol; `Agent.resume(token, action=...)`; `InMemorySuspensionStore` default |
| v0.18 | Failure distribution example | `examples/failure_distribution.py` + `docs/examples/failure-distribution.md`; aggregates OTel span attributes into per-type frequency and recovery-rate table; no new library code |
| v0.19 | Native sync-agent support | Plain `def` callables accepted by `Agent`; run via `anyio.to_thread.run_sync()`; all policy, checkpointing, hooks, and ContextVar injection unchanged |

---

## Later (no version assigned)

- **OpenAI Agents SDK adapter** — `wrap_openai_agents()`; deprioritised until the SDK stabilises.
- **Persistent circuit breaker state** — serialize `CircuitBreaker` state to the checkpoint store (Redis-backed) so the open/half-open state survives process restarts in multi-worker and serverless deployments.
- **Streaming agent support** — needs a design doc covering the partial `Step` model and mid-stream retry semantics before implementation begins.
- **Saga / compensating rollback** — reverse-order compensate callables that undo side effects on rollback; deliberately minimal and only if there is concrete user demand.
