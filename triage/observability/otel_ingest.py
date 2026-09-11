"""
triage.observability.otel_ingest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build a Trajectory from OpenTelemetry spans an agent framework already
emits — the inverse of triage.observability.otel (which emits triage's own
spans). Where that module answers "how do I watch triage," this one answers
"how do I get triage without hand-writing record_step() calls."

Why this exists: every triage integration today requires the wrapped
callable to construct Step objects itself, one per tool call or error. That
is a real, upfront cost, and it buys nothing if your framework (LangChain,
a custom agent loop, an OpenAI-compatible client with auto-instrumentation)
already emits OpenTelemetry spans for the same tool calls and errors —
openllmetry/traceloop-style instrumentation and the OTel GenAI semantic
conventions (execute_tool, gen_ai.tool.name, ...) already exist for exactly
this. trajectory_from_spans() converts those spans into a Trajectory
directly, so record_step() becomes a thin loop over already-captured spans
instead of hand-instrumentation.

IMPORTANT — the OTel GenAI semantic conventions are NOT stable as of this
writing (Development stability, no mandatory requirement level, attribute
names have already changed between spec revisions — e.g.
gen_ai.tool.call.arguments vs. gen_ai.tool.input for the same concept).
This module is defensive about it: it tries multiple known key spellings
per field and never raises on an unrecognized shape, but a future spec
revision may still require an update here. The one part of this that IS
built on stable ground is error extraction: OTel's core "exception" event
convention (exception.type/exception.message, populated by every SDK's
record_exception()) has been stable for years, so Step.error/exception_type
are the most reliable fields this produces. Step.tool_called/tool_input are
best-effort by comparison.

A bonus this enables for free: spans that carry the stable
http.response.status_code (or the older http.status_code) attribute get it
copied into Step.metadata["http_status"] — the same field
RulesClassifier's structured-error-code matching reads (see
triage/classifier/rules.py). A real HTTP client span with a 429/500/502/503
or 408/504 status is enough to fire EXTERNAL_FAULT/TIMEOUT without any
message-text or exception-type support at all, with zero extra code from
the caller.

Usage::

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from triage.observability.otel_ingest import trajectory_from_spans

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    async def my_agent(task: str, *, record_step, **kwargs):
        # your already-OTel-instrumented framework runs here, unmodified —
        # no manual Step construction
        try:
            result = await already_instrumented_framework.run(task)
        finally:
            trajectory = trajectory_from_spans(exporter.get_finished_spans())
            for step in trajectory.steps:
                record_step(step)
        return result

See examples/otel_trajectory.py for a complete runnable version.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from triage.taxonomy import Step
from triage.trajectory import Trajectory

logger = logging.getLogger("triage")

try:
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    StatusCode = None

# Known attribute key spellings per concept, tried in order — the GenAI
# semantic conventions are Development-stability and have already renamed
# some of these between spec revisions (see module docstring).
_TOOL_NAME_KEYS = ("gen_ai.tool.name",)
_TOOL_INPUT_KEYS = ("gen_ai.tool.call.arguments", "gen_ai.tool.input")
_TOOL_OUTPUT_KEYS = ("gen_ai.tool.call.result", "gen_ai.tool.output")
_OPERATION_NAME_KEYS = ("gen_ai.operation.name",)
_HTTP_STATUS_KEYS = ("http.response.status_code", "http.status_code")
_ERROR_TYPE_FALLBACK_KEYS = ("error.type",)

_GEN_AI_PREFIX = "gen_ai."


def trajectory_from_spans(spans: Sequence[Any], *, include_all: bool = False) -> Trajectory:
    """Convert finished OTel spans into a Trajectory.

    Parameters
    ----------
    spans:
        Finished spans, e.g. from ``InMemorySpanExporter.get_finished_spans()``
        or any ``SpanExporter.export()`` batch. Duck-typed — anything with
        ``.name``, ``.attributes``, ``.status``, ``.events``,
        ``.start_time``/``.end_time`` works (this is exactly the shape of
        ``opentelemetry.sdk.trace.ReadableSpan``).

    include_all:
        By default (``False``), only spans that look like meaningful work —
        any ``gen_ai.*`` attribute, an HTTP status attribute, or an ERROR
        status — become a Step. A trace tree from a real framework typically
        also contains internal/infra spans (DB calls, framework-internal
        bookkeeping) that would just add noise. Pass ``True`` to convert
        every span given, e.g. if you already filtered the list yourself.

    Spans are ordered by ``end_time`` (falling back to ``start_time``, then
    input order) before conversion, since callers may hand these in
    arbitrary order (e.g. from a dict or an unordered export batch).

    Raises ``RuntimeError`` if ``opentelemetry`` is not installed — unlike
    ``triage.observability.otel``'s emit-side helpers, which no-op when OTel
    is absent, this function is meaningless without it: there is no
    sensible empty-Trajectory fallback for "you handed me spans but I can't
    read span status without the SDK."
    """
    if not _OTEL_AVAILABLE:
        raise RuntimeError(
            "triage.observability.otel_ingest.trajectory_from_spans() requires "
            "opentelemetry-api to be installed: pip install triage-agent[otel]"
        )

    ordered = sorted(
        spans,
        key=lambda s: (getattr(s, "end_time", None) or getattr(s, "start_time", None) or 0,),
    )

    steps: list[Step] = []
    for i, span in enumerate(ordered):
        attrs: Mapping[str, Any] = getattr(span, "attributes", None) or {}
        is_error = _is_error_span(span)

        if not include_all and not _looks_relevant(attrs, is_error):
            continue

        error, exception_type = _extract_error(span, attrs, is_error)
        step = Step(
            index=i,
            action=_first_present(attrs, _OPERATION_NAME_KEYS) or getattr(span, "name", "") or "",
            tool_called=_first_present(attrs, _TOOL_NAME_KEYS),
            tool_input=_decode_json_field(_first_present(attrs, _TOOL_INPUT_KEYS)),
            tool_output=_decode_json_field(_first_present(attrs, _TOOL_OUTPUT_KEYS)),
            error=error,
            exception_type=exception_type,
            metadata=_extract_metadata(attrs),
        )
        timestamp = getattr(span, "end_time", None) or getattr(span, "start_time", None)
        if timestamp is not None:
            step.timestamp = timestamp / 1e9  # OTel times are ns since epoch
        steps.append(step)

    return Trajectory(steps=steps)


def _is_error_span(span: Any) -> bool:
    status = getattr(span, "status", None)
    if status is None:
        return False
    return getattr(status, "status_code", None) == StatusCode.ERROR


def _looks_relevant(attrs: Mapping[str, Any], is_error: bool) -> bool:
    if is_error:
        return True
    if any(k.startswith(_GEN_AI_PREFIX) for k in attrs):
        return True
    return any(k in attrs for k in _HTTP_STATUS_KEYS)


def _first_present(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in attrs and attrs[key] is not None:
            return attrs[key]
    return None


def _decode_json_field(value: Any) -> Any:
    """OTel span attribute values are primitives (or arrays of one primitive
    type) per the stable attribute-value spec — a structured tool call
    argument/result is realistically carried as a JSON-encoded string, not
    a native dict/list. Decode it if so; pass through unchanged otherwise
    (already-structured data, or a plain string that isn't JSON)."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _extract_error(
    span: Any, attrs: Mapping[str, Any], is_error: bool
) -> tuple[str | None, str | None]:
    if not is_error:
        return None, None

    for event in getattr(span, "events", None) or []:
        if getattr(event, "name", None) != "exception":
            continue
        event_attrs: Mapping[str, Any] = getattr(event, "attributes", None) or {}
        message = event_attrs.get("exception.message")
        exc_type = event_attrs.get("exception.type")
        if message or exc_type:
            return (
                str(message) if message else None,
                str(exc_type) if exc_type else None,
            )

    # No exception event — fall back to the span status description, and
    # error.type if the instrumentation set it directly on the span.
    status = getattr(span, "status", None)
    description = getattr(status, "description", None) if status is not None else None
    exc_type = _first_present(attrs, _ERROR_TYPE_FALLBACK_KEYS)
    return (
        str(description) if description else None,
        str(exc_type) if exc_type else None,
    )


def _extract_metadata(attrs: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    http_status = _first_present(attrs, _HTTP_STATUS_KEYS)
    if http_status is not None:
        try:
            metadata["http_status"] = int(http_status)
        except (TypeError, ValueError):
            logger.warning(
                "[triage] otel_ingest: non-integer HTTP status attribute %r, skipping",
                http_status,
                extra={"triage_event": "otel_ingest_bad_http_status"},
            )
    return metadata
