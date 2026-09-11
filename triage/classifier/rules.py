"""
triage.classifier.rules
~~~~~~~~~~~~~~~~~~~~~~~
Heuristic, synchronous, zero-API-calls classifier. MVP default.
Rules are evaluated in priority order; first match wins.

Scope: RulesClassifier reliably detects LOOP_DETECTED, WRONG_TOOL_CALLED,
SCHEMA_MISMATCH, EXTERNAL_FAULT, TIMEOUT, and CONSTRAINT_IGNORED. It returns
UNKNOWN for PLAN_INCOMPLETE and CONTEXT_OVERFLOW — those require semantic
understanding and are handled by LLMClassifier or HybridClassifier.

Structured error codes: in addition to message-text patterns, each rule also
checks a caller-supplied structured code in Step.metadata — "http_status" (int)
and/or "json_rpc_code" (int) — when present. This is opt-in: nothing populates
Step.metadata automatically, so existing callers see no behavior change. Only
codes with an UNAMBIGUOUS single-FailureType mapping are matched (see the
_JSON_RPC_*/_HTTP_* tables below) — the same 100%-precision-by-construction
guarantee every message-text rule in this module already keeps. See
docs/concepts/classifiers.md's "Structured error codes" section for how to
populate it and docs/known-limitations.md's "Corpus E scoping" for why this
exists.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

# Compiled patterns for speed (called on every failure).
# WRONG_TOOL_CALLED: covers major SDK error formats across OpenAI, Anthropic,
# LangGraph, CrewAI, and generic "not found" messages. Word-boundary anchors
# prevent false matches on e.g. "tooltip not found".
_WRONG_TOOL_RE = re.compile(
    r"(?:"
    r"tool[_\s]['\"]?\w+['\"]?\s+not\s+found"  # 'tool foo not found'
    r"|no\s+tool\s+named"  # 'no tool named X'
    r"|tool_not_found"  # OpenAI structured error code
    r"|function\s+['\"]?\w+['\"]?\s+does\s+not\s+exist"  # Anthropic-style (old)
    r"|no\s+function\s+named\s+['\"]?\w+"  # openai BadRequestError: no function named 'x'
    r"|['\"]?\w+['\"]?\s+does\s+not\s+exist\s+in\s+tools?"  # anthropic: 'x' does not exist in tools
    r"|unknown\s+tool\s+['\"]?\w+"  # generic
    r"|tool\s+['\"]?\w+['\"]?\s+not\s+found\s+in\s+the\s+provided"  # langchain ToolException
    # corpus C (v1.1): patterns below were added from held-out misses — see
    # CHANGELOG and tests/data/error_corpus_c.json for provenance.
    r"|\btool\s+with\s+name\s+[`'\"]?[\w.\-]+[`'\"]?\s+(?:was\s+)?not\s+found"  # cohere/llamaindex
    r"|could\s+not\s+find\s+tool\s+with\s+name"  # llamaindex ToolException
    r"|is\s+not\s+a\s+registered\s+(?:agent\s+)?tool"  # generic agent-registry KeyError
    r"|\bmodel\s+['\"]?\S+?['\"]?\s+does\s+not\s+exist"  # litellm/openai: model 'x' does not exist
    # Requires a "/" in the identifier so this matches a resource path
    # ("Endpoint projects/.../endpoints/456") and not plain-English phrasing
    # like "endpoint documentation is not found" — see
    # test_wrong_tool_false_positive_corpus.
    r"|\bendpoint\s+\S*/\S+\s+is\s+not\s+found"
    r"|\btool\s+[\w.\-]+\s+is\s+not\s+registered\b"  # corpus B target: Tool X is not registered
    # azure-core ResourceNotFoundError: message-only (not exception-type-based —
    # ResourceNotFoundError is reused across unrelated Azure resource kinds, so
    # matching on the type name alone would be too broad). Anchored to the
    # Azure resource-path shape ("<provider>/deployments/<name>' under
    # resource group") rather than a loose "deployment...not found" proximity
    # match, which would also fire on unrelated CI/CD "deployment pipeline"
    # language — see test_wrong_tool_false_positive_corpus.
    r"|\bdeployments?/[\w\-.]+['\"]?\s+under\s+resource\s+group"
    r")",
    re.IGNORECASE,
)
_SCHEMA_RE = re.compile(
    r"(?:"
    r"validation\s+error"  # pydantic ValidationError
    r"|json.*?parse"  # 'json parse failed', 'failed to parse json'
    r"|jsondecodeerror"  # explicit class name
    r"|invalid\s+json"  # 'invalid json in response'
    r"|unexpected\s+token"  # 'unexpected token {'
    # Python json.JSONDecodeError message phrasings
    r"|expecting\s+(?:property\s+name|value|','\s+delimiter)"
    r"|illegal\s+trailing\s+comma"
    r"|extra\s+data"
    # LangChain OutputParserException
    r"|invalid\s+json\s+output"
    # corpus C (v1.1): patterns below were added from held-out misses — see
    # CHANGELOG and tests/data/error_corpus_c.json for provenance.
    r"|request\s+body\s+is\s+not\s+valid\s+json"  # azure-core InvalidRequestBody
    r"|unprocessable\s+entity"  # HTTP 422 text form, used across many SDKs
    r"|json\s+schema\s+validation"  # groq: json_validate_failed / JSON schema validation failed
    r"|is\s+invalid\.\s+expected"  # litellm: 'x' is invalid. Expected a string, got object.
    r"|expected\s+output\s+to\s+be\s+formatted\s+as"  # llamaindex OutputParserError
    r")",
    re.IGNORECASE,
)
# HTTP status codes matched as whole tokens. The negative lookahead excludes
# "500 items", "503 records" etc. (number followed by a quantity noun).
# Python's re module requires fixed-width lookbehinds, so the reverse pattern
# ("step 500", "line 503") is handled by _external_code_match() below instead.
_EXTERNAL_CODE_RE = re.compile(
    r"\b(429|500|502|503)\b"
    r"(?!\s*(?:item|record|result|element|byte|char|step|file|line|row|doc)s?\b)",
    re.IGNORECASE,
)
# Words that, when immediately preceding a status code, indicate a quantity context
# rather than an HTTP error ("step 500", "line 503", "row 429").
_EXTERNAL_CODE_PRECEDING_RE = re.compile(
    r"\b(?:item|record|result|element|byte|char|step|file|line|row|doc)s?\s+"
    r"(429|500|502|503)\b",
    re.IGNORECASE,
)
# Text-form rate-limit and server-error phrases from SDK clients that don't embed
# HTTP codes in their message strings (openai.RateLimitError, anthropic.InternalServerError).
_EXTERNAL_TEXT_RE = re.compile(
    r"(?:"
    r"\brate[\s_]limit"  # 'rate limit reached', 'rate_limit_error'
    r"|\binternal\s+server\s+error"  # anthropic/openai InternalServerError
    r"|\bservice\s+unavailable"  # 503 text form
    r"|\bbad\s+gateway"  # 502 text form
    r"|\bquota\s+exceeded"  # openai quota
    r")",
    re.IGNORECASE,
)


def _external_code_match(text: str) -> bool:
    """Return True only if text contains a status code NOT in a quantity context."""
    if not _EXTERNAL_CODE_RE.search(text):
        return False
    # Exclude matches where the code is preceded by a quantity/ordinal word.
    return not _EXTERNAL_CODE_PRECEDING_RE.search(text)


_TIMEOUT_RE = re.compile(
    r"\btimeout\b|\btimed[\s_]?out\b|\bdeadline[\s_]?exceeded\b|\btime[\s_]?limit\b"
    # corpus C (v1.1): vertex aiplatform phrases the deadline with an
    # interposed duration — "Deadline of 60.0s exceeded" — the fixed-width
    # pattern above requires "deadline exceeded" adjacent.
    r"|\bdeadline\s+of\s+[\d.]+s?\s+exceeded\b"
    # corpus B target: aiohttp ServerConnectionError has no "timeout" keyword.
    r"|disconnected\s+after\s+[\d.]+\s*seconds?\s+of\s+inactivity\b",
    re.IGNORECASE,
)
# Exception type names that indicate timeout when str(exc) is empty or unhelpful.
_TIMEOUT_EXCEPTION_TYPES = frozenset(
    {
        "TimeoutError",
        "AsyncTimeoutError",
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "asyncio.TimeoutError",
        "trio.TooSlowError",
        "anyio.EndOfStream",
    }
)
# Exception type names that reliably indicate an external/upstream fault, used as a
# fallback when the error message doesn't contain an HTTP code or keyword.
_EXTERNAL_EXCEPTION_TYPES = frozenset(
    {
        "InternalServerError",  # openai / anthropic SDK
        "ServiceUnavailableError",
        "OverloadedError",
        "TooManyRequestsError",  # cohere: message text carries no HTTP code or "rate limit"
    }
)
# Exception type names that indicate schema/validation failure regardless of message
# wording — catches JSONDecodeError phrasings not explicitly enumerated in _SCHEMA_RE
# and frameworks that wrap parser failures under their own exception names.
_SCHEMA_EXCEPTION_TYPES = frozenset(
    {
        "JSONDecodeError",
        "ValidationError",  # pydantic
        "OutputParserException",  # langchain — reserved for parser failures by design
        "SchemaValidationError",
        "InvalidArgument",  # google-genai / grpc
        # NOTE: llamaindex's OutputParserError is deliberately NOT listed here.
        # Corpus D found CrewAI raises a class of the same name for an
        # unrelated failure (an unrecognized ReAct Action, not a schema
        # problem) — matching on this name alone misroutes it to
        # SCHEMA_MISMATCH. See the message pattern below instead, which is
        # specific to LlamaIndex's actual wording.
    }
)

# Structured error codes, read from Step.metadata rather than parsed from message
# text or exception type names. Nothing populates Step.metadata automatically —
# a wrapped agent callable must extract the code from the real exception object
# (e.g. an httpx/anthropic/openai APIStatusError's .status_code, or an MCP
# McpError's .error.code) and pass it via record_step(Step(..., metadata={...})).
# See docs/concepts/classifiers.md's "Structured error codes" section and
# docs/known-limitations.md's "Corpus E scoping" for the rationale and the
# scoring this is meant to enable.
#
# Only codes with an UNAMBIGUOUS single-FailureType mapping are listed here.
# A code shared across multiple failure types (JSON-RPC -32602 "Invalid
# params": both a bad tool name and a malformed argument shape use it; HTTP
# 404/400: too many unrelated causes) is deliberately excluded — the whole
# point of this table is to never turn a code into a confident wrong guess,
# the same 100%-precision-by-construction guarantee every other rule in this
# module keeps. An excluded code simply falls through to the message-text
# rules below, same as if metadata carried nothing at all.
#
# JSON-RPC 2.0 reserved codes (https://www.jsonrpc.org/specification#error_object).
# The -32000..-32099 "Server error" range is implementation-defined per server
# and NOT included — it has no spec-guaranteed meaning to map.
_JSON_RPC_WRONG_TOOL_CODES = frozenset({-32601})  # Method not found
# -32600 "Invalid Request" is deliberately NOT included here, despite the
# JSON-RPC spec describing it as an unambiguous "malformed request" code.
# Corpus E found a real MCP server (langgenius/dify#22675) using -32600 for
# what its own bug-report analysis could not rule out as a session/auth
# lifecycle condition, not a malformed request — the same "generic code
# reused for an unrelated failure" pattern that made corpus D drop
# OutputParserError from _SCHEMA_EXCEPTION_TYPES. -32700 (Parse error) stays:
# it can only mean the request body failed to parse as JSON at all, which has
# no such ambiguity.
_JSON_RPC_SCHEMA_CODES = frozenset({-32700})  # Parse error
_JSON_RPC_EXTERNAL_CODES = frozenset({-32603})  # Internal error

# HTTP status codes as a caller-supplied int (Step.metadata["http_status"]),
# not parsed from message text — see _EXTERNAL_CODE_RE for the message-text
# equivalent of the codes below. 408/504 are net-new: no message-text or
# exception-type rule covers a timeout HTTP status anywhere else in this file.
_HTTP_EXTERNAL_STATUS_CODES = frozenset({429, 500, 502, 503})
_HTTP_TIMEOUT_STATUS_CODES = frozenset({408, 504})

# botocore wraps all errors in a uniform envelope:
#   "An error occurred (ErrorCode) when calling the OperationName operation: message"
# Extract the ErrorCode and map known suffixes to failure types.
_BOTOCORE_RE = re.compile(
    r"An error occurred \((\w+)\) when calling the \w+ operation:",
    re.IGNORECASE,
)
# Suffix → FailureType mapping for botocore error codes.
# Checked in order; first match wins.
_BOTOCORE_CODE_MAP: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Throttl|TooManyRequest|RateExceed|RequestLimitExceed", re.IGNORECASE),
        "external_fault",
    ),
    (
        re.compile(r"ServiceUnavailable|Unavailable|ServiceDown", re.IGNORECASE),
        "external_fault",
    ),
    (
        re.compile(r"InternalFailure|InternalError|ServiceError|ServiceFault", re.IGNORECASE),
        "external_fault",
    ),
    (
        re.compile(r"Validation|InvalidInput|MalformedQuery|ParseError|SchemaError", re.IGNORECASE),
        "schema_mismatch",
    ),
    (re.compile(r"NotFound|NoSuch|DoesNotExist", re.IGNORECASE), "wrong_tool_called"),
    (re.compile(r"Timeout|TimedOut|DeadlineExceed", re.IGNORECASE), "timeout"),
]

# Per-framework patterns — activated when RulesClassifier(framework=...) is set.
# These supplement (OR) the generic patterns above; they do not replace them.
_WRONG_TOOL_FRAMEWORK: dict[str, re.Pattern[str]] = {
    "openai": re.compile(
        r"tool\s+['\"]?\w+['\"]?\s+does\s+not\s+exist",
        re.IGNORECASE,
    ),
    "anthropic": re.compile(
        r"does\s+not\s+exist\s+in\s+tools?\s+list|invalid\s+tool\s+use",
        re.IGNORECASE,
    ),
    "langgraph": re.compile(
        r"not\s+found\s+in\s+tool\s+map|no\s+tool\s+with\s+name",
        re.IGNORECASE,
    ),
}
_SCHEMA_FRAMEWORK: dict[str, re.Pattern[str]] = {
    "openai": re.compile(
        r"failed\s+to\s+parse\s+tool\s+arguments",
        re.IGNORECASE,
    ),
    "anthropic": re.compile(
        r"tool\s+input\s+schema|failed\s+to\s+parse\s+tool\s+input",
        re.IGNORECASE,
    ),
}
_EXTERNAL_FRAMEWORK: dict[str, re.Pattern[str]] = {
    "openai": re.compile(
        r"exceeded\s+your\s+current\s+quota|rate_limit_exceeded",
        re.IGNORECASE,
    ),
    "anthropic": re.compile(
        r"rate\s+limit\s+exceeded",
        re.IGNORECASE,
    ),
}


def _tool_input_key(tool_input: object) -> str:
    """Canonical string form of a tool_input dict for equality comparison."""
    if isinstance(tool_input, dict):
        return json.dumps(tool_input, sort_keys=True)
    return str(tool_input)


def _is_loop_window(steps: list[Step], threshold: float | None) -> bool:
    """True if every step in ``steps`` shares the same ``tool_called`` and
    their canonical ``tool_input`` strings are either identical (default) or,
    when ``threshold`` is set, similar enough consecutively (each step vs. the
    one before it) per ``difflib.SequenceMatcher.ratio()``.

    Consecutive comparison (not all-vs-first) so a loop where the query drifts
    gradually across the window is still caught — e.g. step 1 vs step 2 close,
    step 2 vs step 3 close, even if step 1 vs step 3 has drifted further apart.

    Does not look at ``Step.agent_id`` at all — a loop is the same tool call
    repeated, regardless of which agent(s) made it. This is intentional, not
    an oversight: see the caller's comment in ``classify()``.
    """
    if steps[0].tool_called is None:
        return False
    if not all(s.tool_called == steps[0].tool_called for s in steps):
        return False

    keys = [_tool_input_key(s.tool_input) for s in steps]
    if threshold is None:
        return all(k == keys[0] for k in keys)

    return all(
        keys[i] == keys[i - 1] or SequenceMatcher(None, keys[i - 1], keys[i]).ratio() >= threshold
        for i in range(1, len(keys))
    )


class RulesClassifier:
    """Pattern-based classifier. Instantiate with optional constraint strings.

    Parameters
    ----------
    constraints:
        Forbidden strings to detect in step ``llm_output``. If any of these
        strings appear verbatim in a step's LLM output, the failure is
        classified as ``CONSTRAINT_IGNORED``.

        Pass the **forbidden content itself**, not the rule description::

            # Correct: flag if the word "markdown" appears in output
            RulesClassifier(constraints=["markdown"])

            # Correct: flag if a specific phrase leaks into output
            RulesClassifier(constraints=["<script>", "DROP TABLE"])

            # Wrong: this passes the rule text, not the forbidden content
            RulesClassifier(constraints=["no markdown allowed"])

    loop_window:
        Number of consecutive steps required to declare a loop. Default 3.
        Set higher (e.g. 4–5) if your agent legitimately repeats the same
        tool call twice in a row.

    loop_similarity_threshold:
        If set (e.g. ``0.9``), loop detection additionally matches steps whose
        canonical ``tool_input`` strings are *similar* rather than identical —
        catching loops where the agent reworded a query slightly on each retry
        (e.g. ``{"q": "revenue Q1"}`` vs ``{"q": "revenue for Q1"}``).
        Similarity is computed with ``difflib.SequenceMatcher.ratio()`` on the
        canonical JSON string form of ``tool_input``, compared consecutively
        within the window (each step vs. the previous one) rather than all
        pairs against the first — this matches loops that drift gradually,
        not just loops identical to the very first step. ``tool_called`` must
        still match exactly across the whole window; only ``tool_input`` gets
        the fuzzy comparison. Default ``None`` disables fuzzy matching —
        behavior is unchanged from pre-v0.12 (exact match only).

    framework:
        Optional SDK/framework name. When set, per-framework error patterns
        are checked in addition to the generic patterns. Supported values:
        ``"openai"``, ``"anthropic"``, ``"langgraph"``. Case-insensitive.
        Unknown values (including ``"langchain"`` — covered by generic patterns)
        are silently ignored; generic patterns still apply.
    """

    def __init__(
        self,
        constraints: list[str] | None = None,
        loop_window: int = 3,
        loop_similarity_threshold: float | None = None,
        framework: str | None = None,
    ) -> None:
        self.constraints: list[str] = constraints or []
        if loop_window < 2:
            raise ValueError("loop_window must be >= 2")
        self.loop_window = loop_window
        if loop_similarity_threshold is not None and not (0.0 < loop_similarity_threshold <= 1.0):
            raise ValueError("loop_similarity_threshold must be in (0.0, 1.0]")
        self.loop_similarity_threshold = loop_similarity_threshold
        self._framework: str | None = framework.lower() if framework else None

    def _fw_match(self, error: str, table: dict[str, re.Pattern[str]]) -> bool:
        """Return True if a framework-specific pattern matches the error string."""
        if self._framework is None:
            return False
        pat = table.get(self._framework)
        return bool(pat and pat.search(error))

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:  # noqa: ARG002
        steps = trajectory.steps

        # 1. LOOP_DETECTED — last loop_window steps share identical tool_called,
        # and either identical (default) or fuzzy-similar (loop_similarity_threshold)
        # tool_input. Deliberately agent_id-agnostic: matching is purely on
        # tool_called/tool_input, so a step repeated across two different agents
        # (MAST's "Step Repetition" — a handoff-caused re-do of already-completed
        # work, not just one agent looping on itself) is caught by the exact same
        # check, with no separate cross-agent code path. See
        # test_loop_detected_across_different_agent_ids and
        # docs/concepts/multi-agent-failures.md.
        if len(steps) >= self.loop_window:
            window = steps[-self.loop_window :]
            if _is_loop_window(window, self.loop_similarity_threshold):
                return FailureType.LOOP_DETECTED

        # 2. WRONG_TOOL_CALLED — message patterns, framework patterns, or an
        # unambiguous structured code in step.metadata (see the _JSON_RPC_*/
        # _HTTP_* tables above).
        for step in steps:
            if (
                step.error
                and (
                    _WRONG_TOOL_RE.search(step.error)
                    or self._fw_match(step.error, _WRONG_TOOL_FRAMEWORK)
                )
            ) or step.metadata.get("json_rpc_code") in _JSON_RPC_WRONG_TOOL_CODES:
                return FailureType.WRONG_TOOL_CALLED

        # 2b. botocore envelope — parse error code before generic patterns fire
        for step in steps:
            if step.error:
                m = _BOTOCORE_RE.search(step.error)
                if m:
                    code = m.group(1)
                    for pat, kind in _BOTOCORE_CODE_MAP:
                        if pat.search(code):
                            return FailureType(kind)

        # 3. SCHEMA_MISMATCH — string patterns, framework patterns, exception type
        # name, or an unambiguous structured code in step.metadata.
        for step in steps:
            if step.error and (
                _SCHEMA_RE.search(step.error) or self._fw_match(step.error, _SCHEMA_FRAMEWORK)
            ):
                return FailureType.SCHEMA_MISMATCH
            if step.exception_type and step.exception_type in _SCHEMA_EXCEPTION_TYPES:
                return FailureType.SCHEMA_MISMATCH
            if step.metadata.get("json_rpc_code") in _JSON_RPC_SCHEMA_CODES:
                return FailureType.SCHEMA_MISMATCH

        # 4. EXTERNAL_FAULT — HTTP status codes (message text or step.metadata),
        # text-form rate-limit/server errors, exception type names, or a JSON-RPC
        # code, when the message alone is insufficient.
        for step in steps:
            if (
                step.error
                and (
                    _external_code_match(step.error)
                    or _EXTERNAL_TEXT_RE.search(step.error)
                    or self._fw_match(step.error, _EXTERNAL_FRAMEWORK)
                )
            ) or (
                (step.exception_type and step.exception_type in _EXTERNAL_EXCEPTION_TYPES)
                or step.metadata.get("http_status") in _HTTP_EXTERNAL_STATUS_CODES
                or step.metadata.get("json_rpc_code") in _JSON_RPC_EXTERNAL_CODES
            ):
                return FailureType.EXTERNAL_FAULT

        # 5. TIMEOUT — string match, exception type when str(exc) is empty/unhelpful,
        # or an HTTP timeout status code (408, 504) in step.metadata — the gap
        # that has no message-text or exception-type equivalent anywhere above.
        for step in steps:
            if step.error and _TIMEOUT_RE.search(step.error):
                return FailureType.TIMEOUT
            if step.exception_type and step.exception_type in _TIMEOUT_EXCEPTION_TYPES:
                return FailureType.TIMEOUT
            if step.metadata.get("http_status") in _HTTP_TIMEOUT_STATUS_CODES:
                return FailureType.TIMEOUT

        # 6. CONSTRAINT_IGNORED — llm_output contains a forbidden constraint string
        if self.constraints:
            for step in steps:
                if step.llm_output:
                    output_lower = step.llm_output.lower()
                    for constraint in self.constraints:
                        if constraint.lower() in output_lower:
                            return FailureType.CONSTRAINT_IGNORED

        # PLAN_INCOMPLETE and CONTEXT_OVERFLOW cannot be detected by pattern
        # matching — use LLMClassifier or HybridClassifier for those.
        return FailureType.UNKNOWN

    def fit(self, corrections_path: str = "corrections.jsonl") -> dict[str, dict[str, int]]:
        """Read a corrections JSONL file and report classifier coverage.

        For each correction where the rules classifier would have returned a
        different type than expected, a warning is logged. Use this to identify
        systematic misclassifications after calling
        ``agent.report_misclassification()``.

        Returns a coverage dict keyed by FailureType value::

            {
                "wrong_tool_called": {"correct": 5, "wrong": 1},
                "external_fault":    {"correct": 3, "wrong": 0},
            }

        Does not modify the classifier's rules or thresholds — purely diagnostic.
        """
        import logging as _logging

        from triage.feedback import load_corrections
        from triage.taxonomy import Step
        from triage.trajectory import Trajectory

        log = _logging.getLogger("triage")
        corrections = load_corrections(corrections_path)

        coverage: dict[str, dict[str, int]] = {}

        for c in corrections:
            # Reconstruct a minimal trajectory from the summary
            steps = [
                Step(
                    index=s.get("index", 0),
                    action=s.get("action", ""),
                    tool_called=s.get("tool_called"),
                    error=s.get("error"),
                )
                for s in c.steps_summary
            ]
            traj = Trajectory()
            for step in steps:
                traj.append(step)

            predicted = self.classify(traj, c.task).value
            expected = c.expected_type

            if expected not in coverage:
                coverage[expected] = {"correct": 0, "wrong": 0}

            if predicted == expected:
                coverage[expected]["correct"] += 1
            else:
                coverage[expected]["wrong"] += 1
                log.warning(
                    "[triage] fit: misclassification detected",
                    extra={
                        "triage_event": "fit_misclassification",
                        "task": c.task,
                        "expected": expected,
                        "predicted": predicted,
                        "steps": c.steps_summary,
                    },
                )

        return coverage
