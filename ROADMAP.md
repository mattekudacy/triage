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

---

## Near-term (v0.17–v0.20)

### v0.17 — Human-in-the-loop pause/resume

When a failure warrants human judgment, `ESCALATE` today raises immediately and discards
run state. This feature makes `ESCALATE` optionally suspend instead: the agent serializes
a `SuspendedRun` token to the checkpoint store and returns it to the caller without raising.
`Agent.resume(token, action=...)` then restarts the run from that exact point once a human
(or an approval webhook) has supplied a decision. The core persists and reloads state;
routing that decision from Slack, a CLI, or an HTTP callback is userland.

### v0.18 — Failure distribution example

The OTel integration (v0.14) emits per-classification spans, but there is no reference
showing how to aggregate them into a breakdown of failure types across a run population.
`examples/failure_distribution.py` will demonstrate reading the span attributes and producing
a frequency table — input counts, output failure-type mix, recovery rate — using only the
existing OTel metrics counters. No new library code; purely an example and accompanying
documentation section.

### v0.19 — OpenAI Agents SDK adapter

`wrap_openai_agents()` following the same pattern as `wrap_langgraph()` and `wrap_langchain()`.
The OpenAI Agents SDK exception hierarchy (`AgentError`, `MaxTurnsExceeded`, tool-not-found
errors) maps cleanly onto the `FailureType` taxonomy, making automatic classification
straightforward. Framework import is lazy, behind `try/except ImportError`, so the module
is safe to import without the SDK installed.

### v0.20 — Native sync-agent support

Synchronous agent functions — legacy code, compute-bound tasks, libraries that predate
async — cannot be wrapped today without converting them to `async def`. This feature allows
passing a plain `def` callable to `Agent`; triage runs it via `anyio.to_thread.run_sync()`
so the full policy loop, classification, checkpointing, and lifecycle hooks apply unchanged.
No public API changes for users who already wrap async callables.

---

## Later (no version assigned)

- **Persistent circuit breaker state** — serialize `CircuitBreaker` state to the checkpoint store (Redis-backed) so the open/half-open state survives process restarts in multi-worker and serverless deployments.
- **Streaming agent support** — needs a design doc covering the partial `Step` model and mid-stream retry semantics before implementation begins.
- **Saga / compensating rollback** — reverse-order compensate callables that undo side effects on rollback; deliberately minimal and only if there is concrete user demand.
