"""Tests for triage.classifier.rules — RulesClassifier."""

import pytest

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
    exception_type: str | None = None,
    metadata: dict | None = None,
    agent_id: str | None = None,
) -> Step:
    return Step(
        index=index,
        action="test step",
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
        exception_type=exception_type,
        metadata=metadata or {},
        agent_id=agent_id,
    )


def traj(*steps: Step) -> Trajectory:
    t = Trajectory()
    for s in steps:
        t.append(s)
    return t


# ── LOOP_DETECTED ──────────────────────────────────────────────────────────────


def test_loop_detected():
    step = make_step(tool_called="search", tool_input={"q": "hello"})
    t = traj(
        step,
        make_step(1, tool_called="search", tool_input={"q": "hello"}),
        make_step(2, tool_called="search", tool_input={"q": "hello"}),
    )
    assert RulesClassifier().classify(t, "task") == FailureType.LOOP_DETECTED


def test_loop_detected_across_different_agent_ids():
    """MAST's "Step Repetition" (see docs/concepts/multi-agent-failures.md):
    a handoff causes a second agent to unnecessarily redo work a first agent
    already completed. Loop matching is deliberately agent_id-agnostic —
    tool_called/tool_input equality alone is enough, regardless of who made
    the call — so this fires with zero code change beyond Step.agent_id
    existing as a field. Pins that as a verified fact, not an assumption."""
    t = traj(
        make_step(0, tool_called="search", tool_input={"q": "hello"}, agent_id="agent-a"),
        make_step(1, tool_called="search", tool_input={"q": "hello"}, agent_id="agent-b"),
        make_step(2, tool_called="search", tool_input={"q": "hello"}, agent_id="agent-a"),
    )
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
    t3 = traj(
        step,
        make_step(1, tool_called="search", tool_input={"q": "x"}),
        make_step(2, tool_called="search", tool_input={"q": "x"}),
    )
    assert clf.classify(t3, "task") == FailureType.UNKNOWN
    # 4 identical steps — at window, must trigger
    t4 = traj(
        step,
        make_step(1, tool_called="search", tool_input={"q": "x"}),
        make_step(2, tool_called="search", tool_input={"q": "x"}),
        make_step(3, tool_called="search", tool_input={"q": "x"}),
    )
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


# ── adversarial near-miss corpus ──────────────────────────────────────────────
# These parametrized tables guard against false positives and false negatives
# in the regex-heavy rules. Line coverage on this module is 100%, but that
# only tells you every branch ran — not that the guards hold against inputs
# you didn't think of.


@pytest.mark.parametrize(
    "msg",
    [
        "expected 500 items but got 42",
        "processed 503 records successfully",
        "returned 429 results",
        "502 bytes written",
        "step 500 completed",
        "line 503: syntax error",
        "error in row 429",
    ],
)
def test_external_fault_false_positive_corpus(msg: str) -> None:
    """Numbers resembling HTTP status codes in non-error contexts must not fire."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") != FailureType.EXTERNAL_FAULT


@pytest.mark.parametrize(
    "msg",
    [
        "HTTP 429: rate limited",
        "status code 500",
        "received 503 from upstream",
        "server returned 502 bad gateway",
        "upstream error 429",
        "got 500 from remote",
    ],
)
def test_external_fault_true_positive_corpus(msg: str) -> None:
    """Genuine HTTP error strings must still fire EXTERNAL_FAULT."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


@pytest.mark.parametrize(
    "msg",
    [
        "tooltip not found in DOM",
        "found 3 tools available",
        "initialize tool chain",
        "toolbox is empty",
        "retool configuration loaded",
    ],
)
def test_wrong_tool_false_positive_corpus(msg: str) -> None:
    """'tool' in non-error contexts must not trigger WRONG_TOOL_CALLED."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") != FailureType.WRONG_TOOL_CALLED


@pytest.mark.parametrize(
    "msg",
    [
        "validation error: expected string at field 'name'",
        "jsondecodeerror at line 1",
        "invalid json in response body",
        "unexpected token '{' in json",
        "failed to json parse the response",
    ],
)
def test_schema_mismatch_true_positive_corpus(msg: str) -> None:
    """Schema-related error strings must fire SCHEMA_MISMATCH."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


@pytest.mark.parametrize(
    "msg",
    [
        "operation timed out after 30s",
        "deadline exceeded for request",
        "async time limit reached",
        "timed out waiting for response",
    ],
)
def test_timeout_true_positive_corpus(msg: str) -> None:
    """Timeout-related strings must fire TIMEOUT."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


# ── corpus C (v1.1) pattern additions: true positives + adversarial near-misses ──
# rules.py was tuned against corpus C's held-out misses for v1.1 (see CHANGELOG
# and tests/test_classifier_accuracy.py's routing-sensitive floor). These tables
# pin the true positives that motivated each new pattern and the near-misses
# found while narrowing them, so a future edit can't silently widen scope back
# into a false positive without failing a test.


@pytest.mark.parametrize(
    "msg",
    [
        "invalid request: tool with name 'web_scrape' was not found in the "
        "provided tool definitions",
        "Error: Could not find tool with name `get_current_weather`. Please use a valid tool.",
        "'summarize_pdf' is not a registered agent tool",
        "Tool lookup_weather is not registered. Available tools: search, calculator",
        "The model 'gpt-4-vision' does not exist or you do not have access to it.",
        "404 Endpoint projects/123/locations/us-central1/endpoints/456 is not found.",
        "The Resource 'Microsoft.CognitiveServices/accounts/my-account/deployments"
        "/gpt-4o' under resource group 'my-rg' was not found.",
    ],
)
def test_wrong_tool_corpus_c_true_positive(msg: str) -> None:
    """Held-out wrong_tool_called strings (corpus C) that guided v1.1 patterns."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


@pytest.mark.parametrize(
    "msg",
    [
        # "endpoint" near a plain-English noun, not a resource path — the
        # vertex aiplatform pattern requires a "/" in the identifier.
        "the API endpoint documentation is not found in the wiki",
        "endpoint reference is not found in the OpenAPI spec",
        # "model" in a non-SDK sense — the litellm/openai pattern still
        # requires the literal "does not exist" wording immediately after.
        "this pricing model does not exist in isolation from market conditions",
        # "deployment" outside the Azure resource-path shape — narrowed to
        # require "<provider>/deployments/<name>' under resource group"
        # specifically so CI/CD language doesn't misfire.
        "the new deployment pipeline was not found in the CI config",
    ],
)
def test_wrong_tool_corpus_c_false_positive(msg: str) -> None:
    """Near-misses found while narrowing the v1.1 wrong_tool patterns."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") != FailureType.WRONG_TOOL_CALLED


@pytest.mark.parametrize(
    "msg",
    [
        "Operation returned an invalid status 'Bad Request'. "
        "Error: Code: InvalidRequestBody Message: Request body is not valid JSON.",
        "Status 422: Unprocessable Entity: messages: value is not a valid list",
        "Error code: 400 - {'error': {'message': 'json_validate_failed: "
        "JSON schema validation failed', 'type': 'invalid_request_error'}}",
        "litellm.BadRequestError: OpenAIException - 'messages[0].content' "
        "is invalid. Expected a string but got an object.",
    ],
)
def test_schema_mismatch_corpus_c_true_positive(msg: str) -> None:
    """Held-out schema_mismatch strings (corpus C) that guided v1.1 patterns."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


@pytest.mark.parametrize(
    "msg",
    [
        "504 Deadline of 60.0s exceeded while calling aiplatform.googleapis.com:443",
        "Server disconnected after 30.0 seconds of inactivity",
    ],
)
def test_timeout_corpus_c_true_positive(msg: str) -> None:
    """Held-out timeout strings (corpus C / corpus B target) for v1.1 patterns."""
    t = traj(make_step(error=msg))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


def test_external_fault_corpus_c_true_positive_exception_type() -> None:
    """Cohere TooManyRequestsError carries no HTTP code or 'rate limit' text —
    detected via exception_type fallback, added for the v1.1 pattern pass."""
    t = traj(
        make_step(
            error="You are using a Trial key, which is limited to 5 API calls / minute.",
            exception_type="TooManyRequestsError",
        )
    )
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_schema_mismatch_corpus_c_true_positive_message_pattern() -> None:
    """LlamaIndex's distinctive phrasing, matched by message content — NOT by
    exception_type. An earlier v1.1 draft matched on exception_type
    "OutputParserError" instead; corpus D found CrewAI raises a
    same-named-but-unrelated exception (an unrecognized ReAct Action, not a
    schema problem), which that blanket match misrouted to SCHEMA_MISMATCH.
    See test_wrong_tool_corpus_d_false_positive."""
    t = traj(
        make_step(
            error="Got invalid output: Expected output to be formatted as a JSON instance "
            "that conforms to the JSON schema below.",
            exception_type="OutputParserError",
        )
    )
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_output_parser_error_exception_type_alone_does_not_fire_schema() -> None:
    """Same exception_type as above, but message content unrelated to schema —
    must NOT fire SCHEMA_MISMATCH via a blanket exception-type match. Pins the
    corpus D misroute fix: CrewAI's OutputParserError is not LlamaIndex's."""
    t = traj(
        make_step(
            error="Action 'search_the_web' don't exist, these are the only "
            "available Actions: web_search, calculator",
            exception_type="OutputParserError",
        )
    )
    assert RulesClassifier().classify(t, "task") != FailureType.SCHEMA_MISMATCH


# ── Structured error codes (Step.metadata) ──────────────────────────────────
# See docs/known-limitations.md's "Corpus E scoping" section and
# triage/classifier/rules.py's module docstring for the rationale: only codes
# with an unambiguous single-FailureType mapping are matched here, on purpose.
# No corpus dependency — these are synthetic Step objects, not transcribed
# error strings, because no existing corpus (A-D) carries a structured code.


def test_wrong_tool_json_rpc_method_not_found() -> None:
    """-32601 Method not found is JSON-RPC-spec-unambiguous: no wording at
    all is required for this to fire, unlike every message-text rule above."""
    t = traj(make_step(error="", metadata={"json_rpc_code": -32601}))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_wrong_tool_json_rpc_code_fires_even_with_unrelated_message() -> None:
    """The code alone is sufficient — message text is irrelevant to the code
    path, matching the "structural, not textual" design goal."""
    t = traj(
        make_step(
            error="the server said something went wrong",
            metadata={"json_rpc_code": -32601},
        )
    )
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_schema_mismatch_json_rpc_parse_error() -> None:
    """-32700 Parse error can only mean the request body failed to parse as
    JSON at all — unambiguous."""
    t = traj(make_step(error="", metadata={"json_rpc_code": -32700}))
    assert RulesClassifier().classify(t, "task") == FailureType.SCHEMA_MISMATCH


def test_json_rpc_invalid_request_does_not_fire_schema_mismatch() -> None:
    """-32600 Invalid Request is deliberately NOT mapped, despite the
    JSON-RPC spec describing it as unambiguous "malformed request". Corpus E
    found a real MCP server (langgenius/dify#22675) using -32600 for what its
    own bug-report analysis could not rule out as a session/auth lifecycle
    condition, not a malformed request — the same "generic code reused for
    an unrelated failure" pattern that made corpus D drop OutputParserError
    from _SCHEMA_EXCEPTION_TYPES (see
    test_output_parser_error_exception_type_alone_does_not_fire_schema).
    Falling through to UNKNOWN here is correct: never turn a code that real
    servers reuse loosely into a confident wrong guess."""
    t = traj(
        make_step(
            error="Failed to connect to MCP server: code=32600 message="
            "'Session terminated by server'",
            exception_type="McpError",
            metadata={"json_rpc_code": -32600},
        )
    )
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_external_fault_json_rpc_internal_error() -> None:
    """-32603 Internal error is JSON-RPC's server-fault code."""
    t = traj(make_step(error="", metadata={"json_rpc_code": -32603}))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


def test_json_rpc_invalid_params_does_not_fire_anything() -> None:
    """-32602 Invalid params is deliberately NOT in any code table — it's
    shared by both a bad tool name and a malformed argument shape, so the
    code alone can't resolve which FailureType applies. Falling through to
    UNKNOWN (given no matching message text either) is correct: never turn an
    ambiguous code into a confident wrong guess."""
    t = traj(make_step(error="", metadata={"json_rpc_code": -32602}))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


@pytest.mark.parametrize("code", [-32000, -32050, -32099])
def test_json_rpc_server_error_range_does_not_fire_anything(code: int) -> None:
    """The -32000..-32099 reserved range is implementation-defined per MCP
    server, not spec-guaranteed — excluded from every code table."""
    t = traj(make_step(error="", metadata={"json_rpc_code": code}))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_external_fault_http_status_metadata(status: int) -> None:
    """Same codes _EXTERNAL_CODE_RE already matches in message text, now also
    matched as a step.metadata attribute — covers SDKs whose exception
    message never echoes the status code at all (e.g. huggingface_hub's
    RepositoryNotFoundError, a corpus D miss)."""
    t = traj(make_step(error="does not exist", metadata={"http_status": status}))
    assert RulesClassifier().classify(t, "task") == FailureType.EXTERNAL_FAULT


@pytest.mark.parametrize("status", [408, 504])
def test_timeout_http_status_metadata(status: int) -> None:
    """408/504 have no message-text or exception-type equivalent anywhere
    else in this module — this is a net-new capability, not a redundant
    backstop like the 429/500/502/503 case above."""
    t = traj(make_step(error="", metadata={"http_status": status}))
    assert RulesClassifier().classify(t, "task") == FailureType.TIMEOUT


@pytest.mark.parametrize("status", [400, 404])
def test_http_status_404_and_400_do_not_fire_anything(status: int) -> None:
    """404/400 are deliberately excluded from every code table — too many
    unrelated causes share them (a missing tool, a missing unrelated
    resource, any malformed request) to map to one FailureType safely."""
    t = traj(make_step(error="", metadata={"http_status": status}))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_metadata_convention_is_opt_in_default_empty_dict_no_behavior_change() -> None:
    """Step.metadata defaults to {} — every pre-existing caller that never
    sets it must see zero behavior change from this feature."""
    t = traj(make_step(error="totally unrecognized error text"))
    assert RulesClassifier().classify(t, "task") == FailureType.UNKNOWN


def test_message_text_pattern_still_wins_when_metadata_absent() -> None:
    """Sanity check that the new metadata checks are additive, not a
    replacement — existing message-text matching is untouched."""
    t = traj(make_step(error="tool 'foo' not found"))
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED


def test_metadata_code_respects_stage_priority_order() -> None:
    """An unambiguous json_rpc_code in an early stage (WRONG_TOOL_CALLED, stage
    2) must win over a message pattern that would otherwise match a later
    stage (SCHEMA_MISMATCH, stage 3) on the same step — same first-match-wins
    priority order as every other signal type in this classifier."""
    t = traj(
        make_step(
            error="invalid json in response",  # would match _SCHEMA_RE (stage 3)
            metadata={"json_rpc_code": -32601},  # matches stage 2 first
        )
    )
    assert RulesClassifier().classify(t, "task") == FailureType.WRONG_TOOL_CALLED
