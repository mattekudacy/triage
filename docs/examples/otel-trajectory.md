# OTel Trajectory

Build a `Trajectory` from OpenTelemetry spans a framework already emits, instead of hand-writing `Step` objects — `record_step()` becomes a thin replay loop over spans, not a Step construction per tool call.

**Source:** [`examples/otel_trajectory.py`](https://github.com/mattekudacy/triage/blob/main/examples/otel_trajectory.py)

## Requirements

```bash
pip install triage-agent[otel]
```

No API key needed — the "already-instrumented" tool call in this example is synthetic, but it's a real OTel span with real `gen_ai.*` attributes and a real recorded exception, not a mock.

## Why this exists

Every other example in this directory has the wrapped agent construct `Step` objects itself. That's a real upfront cost — you have to instrument your agent before triage can tell you anything, and the payoff is probabilistic. If your framework already emits OpenTelemetry spans for its tool calls and errors — openllmetry/traceloop-style auto-instrumentation, or the OTel GenAI semantic conventions directly (`execute_tool` spans, `gen_ai.tool.name`) — triage can read those spans instead of asking you to duplicate that work by hand.

## What it demonstrates

1. `already_instrumented_search()` stands in for code you do not own — a LangChain run, an OpenAI-compatible client with openllmetry attached, anything that emits its own OTel spans. It is not triage-aware: no `Step` import, no `record_step` call. It raises on the first attempt (a "wrong tool" failure) with a real span carrying `gen_ai.tool.name`, a recorded exception, and an `ERROR` status.
2. The triage-wrapped agent runs that function unmodified, then converts the spans it just emitted into `Step`s with `trajectory_from_spans()` and replays them via `record_step()` — the entire integration is a `try/finally` and a loop, not per-tool-call instrumentation.
3. A normal `triage.Agent.run()` with `WRONG_TOOL_CALLED: retry_with_tool_manifest()` classifies and recovers correctly, entirely from OTel-derived `Step`s.

## What gets extracted, and how reliably

| `Step` field | Source | Reliability |
|---|---|---|
| `error`, `exception_type` | The span's `exception` event (`exception.message`/`exception.type`) | High — OTel's core exception-recording convention, stable for years |
| `tool_called`, `tool_input`, `tool_output` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`/`gen_ai.tool.input`, `gen_ai.tool.call.result`/`gen_ai.tool.output` | Best-effort — the GenAI semantic conventions are still Development-stability; both key spellings are tried |
| `metadata["http_status"]` | `http.response.status_code` (or the older `http.status_code`) | High — stable HTTP semantic convention |

That last row connects to `RulesClassifier`'s structured-error-code matching (see `docs/concepts/classifiers.md`'s "Structured error codes"): a real HTTP client span carrying a `429`/`500`/`502`/`503` or `408`/`504` status is enough to fire `EXTERNAL_FAULT`/`TIMEOUT` with zero extra code from the caller — the span already had the signal, `trajectory_from_spans()` just carries it through.

## Filtering

By default, only spans that look like meaningful work — any `gen_ai.*` attribute, an HTTP status attribute, or an `ERROR` status — become a `Step`. A real trace tree also contains internal/infrastructure spans (DB calls, framework bookkeeping) that would just add noise; pass `include_all=True` to convert every span given, e.g. if you've already filtered the list yourself.

## Current scope

`trajectory_from_spans()` is a pure conversion function — you still call it and loop over the result yourself, as the example shows. There's no `Agent`-level auto-capture yet (no new `Agent.__init__` parameter): that would need to hook the exact window around the wrapped callable's execution without duplicating any manual `record_step()` calls the callable itself makes, which needs its own design pass. Worth watching if you'd use it — [open an issue](https://github.com/mattekudacy/triage/issues).
