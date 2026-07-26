"""Tests for triage.classifier.rules — RulesClassifier."""

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
) -> Step:
    return Step(
        index=index,
        action="test step",
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
    )


def traj(*steps: Step) -> Trajectory:
    t = Trajectory()
    for s in steps:
        t.append(s)
    return t


# ── LOOP_DETECTED ──────────────────────────────────────────────────────────────

def test_loop_detected():
    step = make_step(tool_called="search", tool_input={"q": "hello"})
    t = traj(step, make_step(1, tool_called="search", tool_input={"q": "hello"}),
             make_step(2, tool_called="search", tool_input={"q": "hello"}))
    assert RulesClassifier().classify(t, "task") == FailureType.LOOP_DETECTED


def test_loop_not_detected_two_steps():
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "hello"}),
        make_step(1, tool_called="search", tool_input={"q": "hello"}),
    )
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_loop_window_configurable_detects_at_4():
    clf = RulesClassifier(loop_window=4)
    step = make_step(tool_called="search", tool_input={"q": "x"})
    # 3 identical steps — below the window, must NOT trigger
    t3 = traj(step,
              make_step(1, tool_called="search", tool_input={"q": "x"}),
              make_step(2, tool_called="search", tool_input={"q": "x"}))
    assert clf.classify(t3, "task") == FailureType.UNKNOWN
    # 4 identical steps — at window, must trigger
    t4 = traj(step,
              make_step(1, tool_called="search", tool_input={"q": "x"}),
              make_step(2, tool_called="search", tool_input={"q": "x"}),
              make_step(3, tool_called="search", tool_input={"q": "x"}))
    assert clf.classify(t4, "task") == FailureType.LOOP_DETECTED


def test_loop_window_below_2_raises():
    import pytest
    with pytest.raises(ValueError, match="loop_window"):
        RulesClassifier(loop_window=1)


# ── Fuzzy loop detection (loop_similarity_threshold) ────────────────────────────

def test_fuzzy_loop_detected_on_reworded_query():
    clf = RulesClassifier(loop_similarity_threshold=0.9)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "revenue Q1 report"}),
        make_step(1, tool_called="search", tool_input={"q": "revenue Q1 reports"}),
        make_step(2, tool_called="search", tool_input={"q": "revenue Q1 reports."}),
    )
    assert clf.classify(t, "task") == FailureType.LOOP_DETECTED


def test_fuzzy_loop_not_detected_below_threshold():
    """Queries about genuinely different topics must not trigger a loop."""
    clf = RulesClassifier(loop_similarity_threshold=0.9)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "revenue Q1"}),
        make_step(1, tool_called="search", tool_input={"q": "completely different topic"}),
        make_step(2, tool_called="search", tool_input={"q": "another unrelated subject"}),
    )
    assert clf.classify(t, "task") == FailureType.UNKNOWN


def test_fuzzy_loop_still_requires_matching_tool_called():
    """Similar tool_input across different tools must not trigger a loop."""
    clf = RulesClassifier(loop_similarity_threshold=0.9)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "revenue Q1"}),
        make_step(1, tool_called="lookup", tool_input={"q": "revenue Q1"}),
        make_step(2, tool_called="search", tool_input={"q": "revenue Q1"}),
    )
    assert clf.classify(t, "task") == FailureType.UNKNOWN


def test_fuzzy_loop_default_none_preserves_exact_match_only():
    """Without loop_similarity_threshold, a reworded query must NOT trigger a
    loop — this is the pre-v0.12 behavior and must not change by default."""
    clf = RulesClassifier()  # loop_similarity_threshold=None (default)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "revenue Q1"}),
        make_step(1, tool_called="search", tool_input={"q": "revenue for Q1"}),
        make_step(2, tool_called="search", tool_input={"q": "revenue in Q1"}),
    )
    assert clf.classify(t, "task") == FailureType.UNKNOWN


def test_fuzzy_loop_exact_match_still_detected_with_threshold_set():
    """Setting a threshold must not break exact-match loop detection."""
    clf = RulesClassifier(loop_similarity_threshold=0.9)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "same query"}),
        make_step(1, tool_called="search", tool_input={"q": "same query"}),
        make_step(2, tool_called="search", tool_input={"q": "same query"}),
    )
    assert clf.classify(t, "task") == FailureType.LOOP_DETECTED


def test_fuzzy_loop_catches_gradual_drift_consecutively():
    """A loop where the query drifts a little each step is still caught, even
    if the first and last steps have drifted far apart from each other —
    comparison is consecutive (step vs. previous step), not all-vs-first."""
    clf = RulesClassifier(loop_similarity_threshold=0.85)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "find sales report Q1 2024"}),
        make_step(1, tool_called="search", tool_input={"q": "find sales report Q1 2025"}),
        make_step(2, tool_called="search", tool_input={"q": "find sales reports Q1 2025"}),
    )
    assert clf.classify(t, "task") == FailureType.LOOP_DETECTED


def test_fuzzy_loop_none_tool_input_not_falsely_matched():
    clf = RulesClassifier(loop_similarity_threshold=0.9)
    t = traj(
        make_step(0, tool_called="search", tool_input=None),
        make_step(1, tool_called="search", tool_input=None),
        make_step(2, tool_called="search", tool_input=None),
    )
    assert clf.classify(t, "task") == FailureType.LOOP_DETECTED  # identical (both "None")


def test_loop_similarity_threshold_zero_raises():
    import pytest
    with pytest.raises(ValueError, match="loop_similarity_threshold"):
        RulesClassifier(loop_similarity_threshold=0.0)


def test_loop_similarity_threshold_above_one_raises():
    import pytest
    with pytest.raises(ValueError, match="loop_similarity_threshold"):
        RulesClassifier(loop_similarity_threshold=1.5)


def test_loop_similarity_threshold_negative_raises():
    import pytest
    with pytest.raises(ValueError, match="loop_similarity_threshold"):
        RulesClassifier(loop_similarity_threshold=-0.1)


def test_loop_similarity_threshold_one_is_valid():
    """Upper bound 1.0 is inclusive — equivalent to requiring exact match."""
    clf = RulesClassifier(loop_similarity_threshold=1.0)
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "same"}),
        make_step(1, tool_called="search", tool_input={"q": "same"}),
        make_step(2, tool_called="search", tool_input={"q": "same"}),
    )
    assert clf.classify(t, "task") == FailureType.LOOP_DETECTED


def test_fuzzy_loop_respects_loop_window():
    """Fuzzy matching still only looks at the last loop_window steps."""
    clf = RulesClassifier(loop_window=4, loop_similarity_threshold=0.9)
    # Only 3 similar steps — below the configured window of 4
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "revenue Q1"}),
        make_step(1, tool_called="search", tool_input={"q": "revenue for Q1"}),
        make_step(2, tool_called="search", tool_input={"q": "revenue in Q1"}),
    )
    assert clf.classify(t, "task") == FailureType.UNKNOWN


def test_loop_not_detected_different_inputs():
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "hello"}),
        make_step(1, tool_called="search", tool_input={"q": "world"}),
        make_step(2, tool_called="search", tool_input={"q": "hello"}),
    )
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_loop_not_detected_none_tool():
    # Steps with tool_called=None should not match
    t = traj(make_step(0), make_step(1), make_step(2))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


# ── WRONG_TOOL_CALLED ──────────────────────────────────────────────────────────

def test_wrong_tool_called_no_tool_named():
    t = traj(make_step(error="no tool named calculator"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_called_tool_not_found():
    t = traj(make_step(error="Tool 'bar' not found"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_called_case_insensitive():
    t = traj(make_step(error="NO TOOL NAMED foo"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_called_openai_structured_code():
    t = traj(make_step(error="tool_not_found: the requested tool does not exist"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_called_function_does_not_exist():
    # Anthropic-style message
    t = traj(make_step(error="function 'send_email' does not exist"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_not_triggered_by_unrelated_error():
    t = traj(make_step(error="connection refused"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


# ── SCHEMA_MISMATCH ────────────────────────────────────────────────────────────

def test_schema_mismatch_validation_error():
    t = traj(make_step(error="validation error: field required"))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_schema_mismatch_json_decode():
    t = traj(make_step(error="JSONDecodeError: Expecting value at line 1"))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_schema_mismatch_json_parse():
    t = traj(make_step(error="json parse failed"))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_schema_mismatch_not_triggered_by_unrelated_error():
    t = traj(make_step(error="index out of range"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


# ── EXTERNAL_FAULT ─────────────────────────────────────────────────────────────

def test_external_fault_429():
    t = traj(make_step(error="HTTP 429 Too Many Requests"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_external_fault_500():
    t = traj(make_step(error="500 Internal Server Error"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_external_fault_502():
    t = traj(make_step(error="502 Bad Gateway"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_external_fault_503():
    t = traj(make_step(error="503 Service Unavailable"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_external_fault_not_triggered_by_unrelated_number():
    t = traj(make_step(error="expected 200 items but got 42"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_external_fault_word_boundary_no_false_positive_200():
    # "200" is a success code, must not trigger EXTERNAL_FAULT
    t = traj(make_step(error="expected 200 records"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_external_fault_word_boundary_429_standalone():
    # "429" as a bare number inside a sentence must still trigger
    t = traj(make_step(error="rate limited, status 429"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


# ── CONSTRAINT_IGNORED ────────────────────────────────────────────────────────

def test_constraint_ignored():
    classifier = RulesClassifier(constraints=["do not use markdown"])
    t = traj(make_step(llm_output="Here is the answer. Do not use markdown formatting."))
    assert classifier.classify(t, "task") == FailureType.CONSTRAINT_IGNORED


def test_constraint_ignored_case_insensitive():
    classifier = RulesClassifier(constraints=["DO NOT USE MARKDOWN"])
    t = traj(make_step(llm_output="do not use markdown in your reply"))
    assert classifier.classify(t, "task") == FailureType.CONSTRAINT_IGNORED


def test_constraint_ignored_no_constraints():
    classifier = RulesClassifier(constraints=[])
    t = traj(make_step(llm_output="some output"))
    assert classifier.classify(t, "task") == FailureType.UNKNOWN


def test_constraint_not_violated():
    classifier = RulesClassifier(constraints=["forbidden phrase"])
    t = traj(make_step(llm_output="totally clean output"))
    assert classifier.classify(t, "task") == FailureType.UNKNOWN


# ── UNKNOWN fallback ──────────────────────────────────────────────────────────

def test_unknown_fallback():
    t = traj(make_step(error="something completely unrelated"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_empty_trajectory():
    t = Trajectory()
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


# ── TIMEOUT ───────────────────────────────────────────────────────────────────

def test_timeout_detected_from_asyncio_error():
    t = traj(make_step(error="asyncio.TimeoutError: timeout"))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


def test_timeout_detected_timed_out():
    t = traj(make_step(error="request timed out after 30s"))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


def test_timeout_detected_deadline_exceeded():
    t = traj(make_step(error="deadline exceeded"))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


def test_timeout_detected_time_limit():
    t = traj(make_step(error="time limit reached"))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


def test_timeout_not_detected_on_unrelated_error():
    t = traj(make_step(error="connection refused"))
    assert RulesClassifier().classify(t, "task") != FailureType.TIMEOUT


def test_priority_external_over_timeout():
    # A step with both an HTTP code and a timeout string — EXTERNAL_FAULT wins (rule 4).
    # This documents that HTTP codes take priority over timeout patterns.
    t = traj(make_step(error="503 service timeout"))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


# ── Priority: LOOP_DETECTED wins over EXTERNAL_FAULT ─────────────────────────

def test_priority_loop_over_external():
    # Trajectory that triggers both LOOP_DETECTED and EXTERNAL_FAULT;
    # LOOP_DETECTED has higher priority and must win.
    step = make_step(tool_called="search", tool_input={"q": "q"}, error="503 error")
    t = traj(
        step,
        make_step(1, tool_called="search", tool_input={"q": "q"}, error="503 error"),
        make_step(2, tool_called="search", tool_input={"q": "q"}, error="503 error"),
    )
    assert RulesClassifier().classify(t, "task") == FailureType.LOOP_DETECTED


# ── per-framework patterns ────────────────────────────────────────────────────

def test_openai_wrong_tool_pattern():
    t = traj(make_step(error="Tool 'search' does not exist"))
    assert RulesClassifier(framework="openai").classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_anthropic_wrong_tool_pattern():
    t = traj(make_step(error="Invalid tool use: foo does not exist in tools list"))
    result = RulesClassifier(framework="anthropic").classify(t, "task")
    assert result == FailureType.WRONG_TOOL_CALLED


def test_langgraph_wrong_tool_pattern():
    t = traj(make_step(error="search not found in tool map"))
    result = RulesClassifier(framework="langgraph").classify(t, "task")
    assert result == FailureType.WRONG_TOOL_CALLED


def test_openai_schema_pattern():
    t = traj(make_step(error="Failed to parse tool arguments: unexpected end of JSON"))
    assert RulesClassifier(framework="openai").classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_anthropic_schema_pattern():
    t = traj(make_step(error="Tool input schema must be an object: calculator"))
    assert RulesClassifier(framework="anthropic").classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_openai_rate_limit_pattern():
    t = traj(make_step(error="You exceeded your current quota, please check your billing"))
    assert RulesClassifier(framework="openai").classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_framework_none_misses_framework_errors():
    # Framework-specific error string with no framework= set → falls through to UNKNOWN
    t = traj(make_step(error="Tool 'search' does not exist"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_unknown_framework_falls_back_to_generic():
    # Unrecognised framework value — generic patterns still fire
    t = traj(make_step(error="tool foo not found"))
    assert RulesClassifier(framework="crewai").classify(t, "task") == FailureType.WRONG_TOOL_CALLED
