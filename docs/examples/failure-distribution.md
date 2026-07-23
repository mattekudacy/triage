# Failure Distribution

Aggregate triage's OTel spans into a failure-type frequency table across a run population — showing input counts, failure-type mix, and recovery rate per type.

**Source:** [`examples/failure_distribution.py`](https://github.com/mattekudacy/triage/blob/main/examples/failure_distribution.py)

## Requirements

```bash
pip install triage-agent[otel]
```

No API key needed — all agents in this example are synthetic.

## What it demonstrates

triage emits a `triage.classify` span for every classified failure (attribute: `triage.failure_type`) and a `triage.run` span for every `run()` call (status `ERROR` when the run did not recover). Reading these two span types from any span exporter is enough to produce a complete failure-distribution report.

The example:

1. Runs 20 synthetic agent calls across three injected failure types plus clean (no-failure) runs.
2. Reads the finished spans from an `InMemorySpanExporter`.
3. Groups runs by failure type using `triage.classify` span attributes.
4. Determines recovery outcome from the `triage.run` span's status code (`OK` = recovered, `ERROR` = escalated or aborted).
5. Prints a frequency table with input counts, percentage share, and per-type recovery rate.

## Reading the output

```
── Failure distribution ─────────────────────────────────────────────────────

  Total runs   : 20
  Successful   : 15  (75%)
  With failures: 5

  FAILURE TYPE                COUNT       %    RECOVERED    RECOVERY RATE
  -------------------------------------------------------------------------
  schema_mismatch                 5      25%          0/5               0%  ░░░░░
  wrong_tool_called               5      25%          5/5             100%  █████
  external_fault                  5      25%          5/5             100%  █████
  (no failure)                    5      25%            —                —
  -------------------------------------------------------------------------

── Span inventory ───────────────────────────────────────────────────────────

  triage.classify             20 spans
  triage.dispatch             15 spans
  triage.run                  20 spans
```

- **FAILURE TYPE** — the `triage.failure_type` value from the last `triage.classify` span in that run. Runs with no classification span appear as `(no failure)`.
- **RECOVERED** — runs where the `triage.run` span ended with status `OK` (strategy succeeded). Runs that escalated, aborted, or exceeded `max_recovery_attempts` show as `ERROR`.
- The block bar (`█░`) gives an at-a-glance recovery rate per type.

## Adapting to production

Swap `InMemorySpanExporter` for any OTel-compatible backend (Jaeger, Tempo, Honeycomb, etc.) and query the `triage.classify` and `triage.run` spans directly. The attribute names are stable:

| Attribute | Span | Value |
|-----------|------|-------|
| `triage.failure_type` | `triage.classify` | `FailureType` string value, e.g. `"external_fault"` |
| `triage.run_id` | all | UUID shared across all spans from one `run()` call |
| `triage.action_kind` | `triage.dispatch` | `"retry"`, `"replan"`, `"rollback"`, `"escalate"`, etc. |
| span status | `triage.run` | `OK` = recovered, `ERROR` = not recovered |

A minimal OTEL query for a breakdown by failure type and outcome:

```
spans
| where name == "triage.classify"
| summarize count() by attributes["triage.failure_type"]
```

To correlate with recovery outcome, join on `triage.run_id` against the `triage.run` span's status.

## No new library code

This pattern uses only the span attributes that triage has emitted since v0.14. There is no new API surface — the report is purely a function of reading `triage.classify` and `triage.run` spans from whatever exporter your application already uses.
