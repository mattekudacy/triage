# triage-agent Roadmap

`triage-agent` wraps async agent callables, classifies failure by type, and routes each
type to a recovery strategy. This document tracks what has shipped and what comes next.

---

## Done (v0.1–v0.23)

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
| v0.22   | Redis suspension store            | `RedisSuspensionStore` — durable human-in-the-loop pause/resume backed by Redis; `key_prefix` and `ttl_seconds` options; reuses `serialize_run`/`deserialize_run`                                                            |
| v0.23   | Cost model for `max_cost_usd`     | `triage.pricing` module with per-model price table (Anthropic models); `LLMClassifier._report_usage()` auto-populates `cost_usd` from token counts; `lookup_cost()` public override hook                                     |
| v0.24   | Saga / compensating rollback      | `record_compensator(step_index, fn)` injected kwarg + `get_compensator_recorder()` contextvar; on ROLLBACK, compensators run in reverse step-index order before checkpoint restore; errors swallowed and logged; `triage/strategies/saga.py` |

---

## Next (no version assigned yet)

Items are grouped by urgency. Within each group, order is rough priority.

### Feature completeness

- **OpenAI Agents SDK adapter** — `wrap_openai_agents()`; deprioritised until the SDK
  stabilises. Largest user pool currently unreachable; the adapter mapping is mostly
  mechanical once the SDK settles.
- ~~**Run `scripts/mast_mode_pilot_accuracy.py` against a real model**~~ — done, scored
  once against `gpt-oss:120b`: 4/5 recall across the 3 piloted MAST modes (2.5: 2/2,
  2.4: 1/1, 1.4: 1/2). See `docs/concepts/multi-agent-failures.md`'s "Pilot measurement
  scored once". One run, n=5, no negative-example set yet — a first data point, not a
  validated floor. Re-running for stability and building the negative half of the
  corpus are the natural next steps, not yet started.
- ~~**Run `scripts/hybrid_ambiguity_accuracy.py` against a real model**~~ — done, scored
  once against `gpt-oss:120b`. Result is worse than expected: **100% override rate**
  (12/12) on the `unknown_labeled` group — every genuinely-unknown entry `RulesClassifier`
  correctly left as `UNKNOWN` got turned into a confident wrong guess by the LLM fallback,
  clustering hard on `external_fault` as a catch-all. `tricky_but_classifiable` recall was
  100% (4/4) — the recall-gap side of the story holds up. See `docs/known-limitations.md`'s
  updated "close the recall gap, but not the precision gap" section.
- **Fix `HybridClassifier` overriding a correct rules-`UNKNOWN` with a confident wrong
  guess** — raised in priority by the measurement above: 100% override rate on this run
  is not an edge case, it's the default behavior. Needs a confidence signal the fallback
  can decline on; see the SystemOneClassifier section below for the most direct path to
  one. Until that ships, `known-limitations.md` now says explicitly: treat any
  `HybridClassifier` answer as guilty until proven innocent, not a safe default.
- **MCP JSON-RPC error-code extraction helper** — corpus E scoping step 3 (MCP half
  only): a small opt-in helper that reads `McpError.error.code` and populates
  `Step.metadata["json_rpc_code"]`, so `RulesClassifier`'s structured-code matching
  (already shipped) actually fires without every caller writing the extraction
  themselves. Independent of everything else in this list.

### SystemOneClassifier (Jev / Laya) — proposed, not started

TypeSafe AI's Jev (released 2026-09-15, still waitlisted as of this writing) is a
"System One" decision model: you send it program state plus typed questions and get
back typed answers with calibrated-confidence probabilities, instead of free text to
parse. Laya (Apache-2.0, ~421M params, runs locally via `laya-serve`) speaks the same
API shape and needs no waitlist or key — same relationship Ollama has to Anthropic in
`LLMClassifier` today.

- **`SystemOneClassifier`** — new classifier in `triage/classifier/systemone.py`, one
  pick-one question over the 9 `FailureType` values instead of parsing free text.
  Defaults to hosted Jev; `base_url=` points it at local `laya-serve` instead. Lazy
  import, new optional extra. Blocked on Jev's waitlist for the *default* path — the
  `base_url=` (Laya) path can be built and tested now.
- **Confidence-gated `HybridClassifier` fallback** — direct fix for the item above, and
  the highest-priority item in this section now that it's backed by data: a measured
  100% override rate (see "Feature completeness" above), not the `n=1` this section was
  originally scoped against. When `SystemOneClassifier`'s confidence for its top answer
  is below a threshold, return `UNKNOWN` instead of guessing.
- **Calibrate the threshold against our own corpora** — set it using corpora A/B/C +
  `error_corpus_ambiguous.json` (training data), then score held-out D and E exactly
  once. Scoring against D/E to *pick* the threshold burns them as held-out data — don't.
- **A CI-enforceable accuracy floor for the semantic classifier** — not possible today
  because LLM answers vary run to run. If Laya's answers are stable enough, this
  becomes possible for the first time, pinned to a specific Laya model version.
- **Feed into the MAST phase 3 pilot measurement** — a confidence score lets the pilot
  scorer report "ambiguous between 2.4/2.5" instead of forcing one label, matching what
  the corpus sourcing already found by hand.
- **Smaller/optional:** Jev's rate in `triage/pricing.py` (so `max_cost_usd` caps work
  with it); expose confidence via an optional duck-typed method rather than changing
  `classify()`'s stable return type; feed `record_correction()` misclassifications back
  into threshold recalibration.
- **Explicitly not planned:** Jev as a `StepRiskScorer` backend (contract is
  sync/no-API-calls/<1ms; no System One model meets that); Jev choosing the
  `RecoveryAction` (policy must stay deterministic); Jev as the `Agent()` *default*
  classifier (breaks offline-by-default, reopens the vendor-lock-in issue PR #18 just
  fixed, needs a major version bump under the v1.0 stability promise).

### Evidence and positioning

- ~~**Cut a 1.0 with an API stability commitment**~~ — shipped in v1.0.0. Public API
  frozen; breaking changes require a major version bump.

### Longer term

- **Multi-agent failure taxonomy (MAST)** — phases 1 and 2 done (see
  `docs/concepts/multi-agent-failures.md`); phase 3 (extending `LLMClassifier`/a
  semantic classifier to recognize MAST's 12 semantic-only modes) is scoped but not
  measured — see the pilot-measurement item above. Once that pilot has a real number,
  decide whether to source the remaining 9 modes and whether any earn a stable
  `FailureType` member.
- **Corpus F (gRPC status codes)** — optional. Tests whether the "protocol spec
  guarantees a code's meaning" structural signal that worked for JSON-RPC (corpus E)
  generalizes to another spec-backed protocol, or confirms corpus E's HTTP-code finding
  isn't an artifact of that corpus's particular vendor mix.
- **Widening `_HTTP_EXTERNAL_STATUS_CODES` to include 404/400/422** — considered and
  rejected for now. Would recover some of corpus E's routing-sensitive gap, at the
  direct cost of `RulesClassifier`'s 100%-precision-by-construction guarantee. Not a
  free improvement; would need its own corpus to justify.
- **Auto-capturing OTel spans inside `Agent.run()`'s lifecycle** — `otel_ingest.py`'s
  `trajectory_from_spans()` is a pure function today; the caller still loops over
  `trajectory.steps` and calls `record_step()` themselves. Wiring this directly into
  `Agent.run()` (auto-capturing spans around the wrapped callable, merging with manual
  `record_step()` calls without duplicating them) was deliberately deferred — needs its
  own design pass on span-capture timing and de-duplication before `Agent.__init__`'s
  stable arg list grows for it.

### Watching for adoption signal (no action planned)

- **`RedisSuspensionStore`, `RedisBreakerStore`, `compensating_rollback()`** — shipped
  and tested, marked experimental as of the last release: no known production users yet.
  Not scheduled for further work absent someone reporting a real use case.

