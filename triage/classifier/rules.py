"""
triage.classifier.rules
~~~~~~~~~~~~~~~~~~~~~~~
Heuristic, synchronous, zero-API-calls classifier. MVP default.
Rules are evaluated in priority order; first match wins.

Scope: RulesClassifier reliably detects LOOP_DETECTED, WRONG_TOOL_CALLED,
SCHEMA_MISMATCH, EXTERNAL_FAULT, TIMEOUT, and CONSTRAINT_IGNORED. It returns
UNKNOWN for PLAN_INCOMPLETE and CONTEXT_OVERFLOW — those require semantic
understanding and are handled by LLMClassifier or HybridClassifier.
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
    r"tool[_\s]['\"]?\w+['\"]?\s+not\s+found"      # 'tool foo not found'
    r"|no\s+tool\s+named"                            # 'no tool named X'
    r"|tool_not_found"                               # OpenAI structured error code
    r"|function\s+['\"]?\w+['\"]?\s+does\s+not\s+exist"  # Anthropic-style
    r"|unknown\s+tool\s+['\"]?\w+"                   # generic
    r")",
    re.IGNORECASE,
)
_SCHEMA_RE = re.compile(
    r"validation\s+error|json.*?parse|jsondecodeerror|invalid\s+json|unexpected\s+token",
    re.IGNORECASE,
)
# HTTP status codes matched as whole tokens. Negative lookahead excludes common
# quantity contexts ("expected 500 items", "processed 503 records") that would
# otherwise false-positive on data-volume log lines.
_EXTERNAL_CODE_RE = re.compile(
    r"\b(429|500|502|503)\b"
    r"(?!\s*(?:item|record|result|element|byte|char|step|file|line|row|doc)s?\b)",
    re.IGNORECASE,
)
_TIMEOUT_RE = re.compile(
    r"\btimeout\b|\btimed[\s_]?out\b|\bdeadline[\s_]?exceeded\b|\btime[\s_]?limit\b",
    re.IGNORECASE,
)

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
        # tool_input.
        if len(steps) >= self.loop_window:
            window = steps[-self.loop_window:]
            if _is_loop_window(window, self.loop_similarity_threshold):
                return FailureType.LOOP_DETECTED

        # 2. WRONG_TOOL_CALLED
        for step in steps:
            if step.error and (
                _WRONG_TOOL_RE.search(step.error)
                or self._fw_match(step.error, _WRONG_TOOL_FRAMEWORK)
            ):
                return FailureType.WRONG_TOOL_CALLED

        # 3. SCHEMA_MISMATCH
        for step in steps:
            if step.error and (
                _SCHEMA_RE.search(step.error)
                or self._fw_match(step.error, _SCHEMA_FRAMEWORK)
            ):
                return FailureType.SCHEMA_MISMATCH

        # 4. EXTERNAL_FAULT — HTTP status codes as whole tokens
        for step in steps:
            if step.error and (
                _EXTERNAL_CODE_RE.search(step.error)
                or self._fw_match(step.error, _EXTERNAL_FRAMEWORK)
            ):
                return FailureType.EXTERNAL_FAULT

        # 5. TIMEOUT — Python-level timeout exceptions
        for step in steps:
            if step.error and _TIMEOUT_RE.search(step.error):
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
