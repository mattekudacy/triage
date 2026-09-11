"""Tests for triage.observability.otel_ingest — building a Trajectory from
OTel spans instead of hand-written record_step() calls.

All tests here use real opentelemetry-sdk span objects (TracerProvider +
InMemorySpanExporter), not hand-mocked dicts — the same discipline as
tests/test_observability_otel.py's span-tree tests. Skipped entirely when
opentelemetry-sdk is not installed, same convention as that file.
"""

from __future__ import annotations

import json

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _otel_available() -> bool:
    try:
        import opentelemetry  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark_otel = pytest.mark.skipif(
    not _otel_available(),
    reason="opentelemetry-sdk not installed",
)


def _make_exporter():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("triage-otel-ingest-test")
    return tracer, exporter


# ── requires OTel installed ─────────────────────────────────────────────────


def test_raises_without_otel_installed(monkeypatch):
    import triage.observability.otel_ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "_OTEL_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="opentelemetry-api"):
        ingest_mod.trajectory_from_spans([])


# ── error extraction (the stable, reliable part) ────────────────────────────


@pytestmark_otel
def test_error_span_becomes_step_with_error_and_exception_type():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("execute_tool search") as span:
            try:
                raise RuntimeError("tool 'search' not found")
            except RuntimeError as e:
                span.record_exception(e)
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    step = traj[0]
    assert step.error == "tool 'search' not found"
    assert step.exception_type == "RuntimeError"


@pytestmark_otel
def test_error_span_without_exception_event_falls_back_to_status_description():
    from opentelemetry.trace import Status, StatusCode

    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("execute_tool search") as span:
        span.set_status(Status(StatusCode.ERROR, "upstream returned 503"))

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    assert traj[0].error == "upstream returned 503"


@pytestmark_otel
def test_ok_span_has_no_error_fields():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search", attributes={"gen_ai.tool.name": "search"}
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    assert traj[0].error is None
    assert traj[0].exception_type is None


# ── tool name / input / output (best-effort, spec is Development-stability) ──


@pytestmark_otel
def test_tool_call_span_extracts_tool_name_and_input():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "search",
            "gen_ai.tool.call.arguments": json.dumps({"q": "revenue Q1"}),
        },
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    step = traj[0]
    assert step.tool_called == "search"
    assert step.tool_input == {"q": "revenue Q1"}
    assert step.action == "execute_tool"


@pytestmark_otel
def test_tool_input_fallback_key_spelling():
    """gen_ai.tool.input is an alternate spelling seen across spec revisions
    and instrumentation library versions — must also be tried."""
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={
            "gen_ai.tool.name": "search",
            "gen_ai.tool.input": json.dumps({"q": "x"}),
        },
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].tool_input == {"q": "x"}


@pytestmark_otel
def test_tool_output_extracted():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={
            "gen_ai.tool.name": "search",
            "gen_ai.tool.call.result": json.dumps({"hits": 3}),
        },
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].tool_output == {"hits": 3}


@pytestmark_otel
def test_non_json_tool_input_passed_through_unchanged():
    """Not every real tool call argument is a JSON object — a plain string
    must not be dropped, just passed through as-is."""
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={"gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": "revenue Q1"},
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].tool_input == "revenue Q1"


# ── HTTP status -> Step.metadata["http_status"] (feeds RulesClassifier) ──────


@pytestmark_otel
def test_http_status_code_extracted_into_metadata():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("HTTP POST", attributes={"http.response.status_code": 503}):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    assert traj[0].metadata == {"http_status": 503}


@pytestmark_otel
def test_older_http_status_code_key_also_works():
    """http.status_code (pre-1.0 HTTP semconv name) must also be read —
    still in wide use across older instrumentation library versions."""
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("HTTP POST", attributes={"http.status_code": 429}):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].metadata == {"http_status": 429}


@pytestmark_otel
def test_non_integer_http_status_skipped_gracefully():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "HTTP POST", attributes={"http.response.status_code": "not-a-number"}
    ):
        pass

    # Must not raise — the span is still relevant (key present) but the
    # unparseable value is dropped rather than corrupting metadata.
    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 1
    assert traj[0].metadata == {}


# ── relevance filtering ──────────────────────────────────────────────────────


@pytestmark_otel
def test_irrelevant_span_filtered_out_by_default():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("internal.bookkeeping"):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert len(traj) == 0


@pytestmark_otel
def test_include_all_bypasses_relevance_filter():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("internal.bookkeeping"):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans(), include_all=True)
    assert len(traj) == 1
    assert traj[0].action == "internal.bookkeeping"


# ── ordering ──────────────────────────────────────────────────────────────────


@pytestmark_otel
def test_spans_sorted_by_end_time_regardless_of_input_order():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool first", attributes={"gen_ai.tool.name": "first"}
    ):
        pass
    with tracer.start_as_current_span(
        "execute_tool second", attributes={"gen_ai.tool.name": "second"}
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    reversed_spans = list(reversed(spans))  # deliberately out of order

    traj = trajectory_from_spans(reversed_spans)
    assert [s.tool_called for s in traj.steps] == ["first", "second"]


# ── action field ──────────────────────────────────────────────────────────────


@pytestmark_otel
def test_action_prefers_operation_name_over_span_name():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span(
        "execute_tool search",
        attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search"},
    ):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].action == "execute_tool"


@pytestmark_otel
def test_action_falls_back_to_span_name_without_operation_name():
    from triage.observability.otel_ingest import trajectory_from_spans

    tracer, exporter = _make_exporter()
    with tracer.start_as_current_span("HTTP POST", attributes={"http.response.status_code": 500}):
        pass

    traj = trajectory_from_spans(exporter.get_finished_spans())
    assert traj[0].action == "HTTP POST"


# ── empty input ──────────────────────────────────────────────────────────────


def test_empty_span_list_returns_empty_trajectory():
    from triage.observability.otel_ingest import trajectory_from_spans

    if not _otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    traj = trajectory_from_spans([])
    assert len(traj) == 0
