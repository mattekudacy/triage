# Multi-Agent Pipeline Example

Demo of a two-level agent pipeline — an orchestrator delegating to a researcher and a writer, each wrapped in its own `triage.Agent` — and how a child agent's escalation propagates to the parent's recovery policy instead of crashing the whole pipeline.

**Source:** [`examples/multi_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/multi_agent.py)

## Requirements

None — fully synthetic, no API key needed.

## What it demonstrates

```
orchestrator
  └─ researcher  (triage-wrapped, its own policy and recovery budget)
  └─ writer      (triage-wrapped)
```

1. **Per-agent policies with independent recovery budgets.** The researcher's policy retries `EXTERNAL_FAULT` once before escalating; the orchestrator's policy has its own, separate `max_recovery_attempts`.
2. **Escalation propagation via exception chaining.** The researcher hits a simulated rate limit twice, exhausts its budget, and raises `TriageEscalationError`. The orchestrator's agent body catches that and re-raises a plain `RuntimeError` `from exc` — Python's `raise ... from` sets `exc.__cause__`, which is how the failure context reaches the outer layer instead of being swallowed.
3. **`Agent.clone()` for safe concurrent reuse.** The orchestrator calls `researcher_agent.clone()` (and `writer_agent.clone()`) rather than reusing the module-level `Agent` instance directly, so each pipeline run gets independent per-run state — see [`CLAUDE.md`'s design-decisions section](https://github.com/mattekudacy/triage/blob/main/CLAUDE.md#design-decisions-worth-not-re-litigating) on why per-run state lives behind a `ContextVar` rather than a shared instance attribute.
4. **Replanning around a child failure.** The orchestrator's own policy maps `EXTERNAL_FAULT` (received via the chained exception) to `replan()` with a fallback hint, which switches the researcher to a cached-results query path on the next attempt — recovering the pipeline without surfacing the failure to the caller.

## Run

```bash
python examples/multi_agent.py
```
