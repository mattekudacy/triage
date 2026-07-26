# triage-agent Roadmap

`triage-agent` wraps async agent callables, classifies failure by type, and routes each
type to a recovery strategy. This document tracks what has shipped and what comes next.

---

## Done (v0.1–v0.21)

| Version | Feature                           | Description                                                                                                                                                                                                                   |
| ------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v0.1    | Initial release                   | Core taxonomy (9 failure types), trajectory, checkpoint, RulesClassifier, policy, and agent                                                                                                                                   |
| v0.2    | Public API stability              | Stable FailureType / RecoveryAction / FailurePolicy / Agent / CheckpointStore signatures                                                                                                                                      |
| v0.3    | HybridClassifier + state          | HybridClassifier (rules → LLM fallback); agent state persisted in checkpoints; async LLMClassifier                                                                                                                            |
| v0.4    | Recovery control                  | `max_total_attempts`, `Agent.clone()`, `FailurePolicy.chain()`, `TriageContext`, `contextvars` injection                                                                                                                      |
| v0.5    | Type safety + testing             | PEP 561 `py.typed` marker; `CancelledError` propagation; `triage.testing` utilities                                                                                                                                           |
| v0.6    | TIMEOUT type + hooks              | `TIMEOUT` failure type with `RulesClassifier` detection; `on_step` / `on_failure` / `on_recovery` lifecycle hooks                                                                                                             |
| v0.7    | Taxonomy cleanup + bench          | Removed two ambiguous failure types; `triage.bench` eval harness; `Step.idempotent` flag                                                                                                                                      |
| v0.8    | Safety + feedback                 | `strict_idempotency`, `max_recovery_seconds`, structured event logs, misclassification feedback loop, YAML/TOML policy loading                                                                                                |
| v0.9    | Framework patterns + risk scoring | Per-SDK error patterns (`openai` / `anthropic` / `langgraph`); step-level risk scoring with abort threshold                                                                                                                   |
| v0.10   | Concurrent runs                   | Per-run state via `ContextVar`; two concurrent `run()` calls on the same `Agent` instance are now safe                                                                                                                        |
| v0.11   | Resilience improvements           | Concurrency-safe `InMemoryCheckpointStore`; LLM retry on transient errors; `HybridClassifier` LLM call cap per run                                                                                                            |
| v0.12   | Fuzzy loop detection              | Loop detection via `difflib.SequenceMatcher`; catches gradual query drift without new dependencies                                                                                                                            |
| v0.13   | Run-scoped checkpoints + sequence | Checkpoints carry `run_id`; rollback stays within a run; `FailurePolicy.sequence()` for ordered strategy escalation                                                                                                           |
| v0.14   | OpenTelemetry spans               | `triage.run` / `triage.classify` / `triage.dispatch` spans; lazy OTel import; `tracer=` on `Agent`                                                                                                                            |
| v0.15   | Cost/token budgets                | `max_tokens` / `max_cost_usd` caps; `UsageMeter`; `LLMClassifier` auto-reports token usage                                                                                                                                    |
| v0.16   | Circuit breaker                   | `CircuitBreaker` with CLOSED / OPEN / HALF_OPEN states; `circuit_breaker()` strategy wrapper                                                                                                                                  |
| v0.17   | Human-in-the-loop pause/resume    | `RecoveryAction.SUSPEND`; `SuspensionStore` protocol; `Agent.resume(token, action=...)`; `InMemorySuspensionStore` default                                                                                                    |
| v0.18   | Failure distribution example      | `examples/failure_distribution.py` + `docs/examples/failure-distribution.md`; aggregates OTel span attributes into per-type frequency and recovery-rate table; no new library code                                            |
| v0.19   | Native sync-agent support         | Plain `def` callables accepted by `Agent`; run via `anyio.to_thread.run_sync()`; all policy, checkpointing, hooks, and ContextVar injection unchanged                                                                         |
| v0.20   | Persistent circuit breaker state  | `BreakerStore` protocol + `RedisBreakerStore`; `CircuitBreaker(store=...)` shares OPEN/HALF_OPEN state across workers and survives process restarts; switches to wall-clock timestamps automatically when a store is attached |
| v0.21   | Streaming agent support           | `Agent.stream()` for async-generator callables; `StreamRetryEvent` yielded at retry boundaries; `Step.partial` flag; type guard between `run()` and `stream()`                                                                |

---

## Next (no version assigned yet)

Items are grouped by urgency. Within each group, order is rough priority.

### Feature completeness

- **`RedisSuspensionStore`** — the only real gap in the current feature set.
  `serialize_run` / `deserialize_run` exist and are tested specifically to enable this,
  so it is mostly written. Until it lands, `InMemorySuspensionStore` loses pending
  human approvals on process restart — which is wrong for the one feature whose
  correctness most depends on durability.

- **Cost model for `max_cost_usd`** — `Usage.cost_usd` is entirely caller-supplied; there
  is no price table anywhere in the library. Most users will leave it at `0.0` and get a
  dollar cap that never fires. A small per-model price table (Anthropic, OpenAI, common
  OSS checkpoints) with an override hook would make the feature work as its name implies.

- **Preemptive budget enforcement** — budgets are checked only at failure points, so
  overage is detected after the fact rather than prevented. `agent.py` documents this
  honestly (the "not a hard token ceiling" note explains failure-point enforcement is
  deliberate). Noted here as a design choice that is disclosed, not a defect; promote if
  concrete demand emerges.

- **OpenAI Agents SDK adapter** — `wrap_openai_agents()`; deprioritised until the SDK
  stabilises. Largest user pool currently unreachable; the adapter mapping is mostly
  mechanical once the SDK settles.

- **Saga / compensating rollback** — reverse-order compensate callables that undo side
  effects on rollback. Highest complexity on the roadmap. Build only when there is
  concrete user demand.

### Evidence and positioning

- **Cut a 1.0 with an API stability commitment** — twenty-one minor versions in a short
  window reads as churn, and some bumps were not library changes at all (v0.18 was an
  example file). The public API has been stable since v0.2; a 1.0 with an explicit
  stability promise signals that to potential adopters.

### Longer term

- **Multi-agent failure taxonomy** — current failure types are single-agent. The failure
  modes people hit hardest now are handoff loss, inter-agent misalignment, and output
  poisoning between agents. Aligning with the published MAST taxonomy would provide
  citable rigor and is a stronger differentiator than another adapter.

