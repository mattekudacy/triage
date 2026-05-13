"""
triage.classifier.rules
~~~~~~~~~~~~~~~~~~~~~~~
Heuristic, synchronous, zero-API-calls classifier. MVP default.
Rules are evaluated in priority order; first match wins.

Scope: RulesClassifier reliably detects LOOP_DETECTED, WRONG_TOOL_CALLED,
SCHEMA_MISMATCH, EXTERNAL_FAULT, TIMEOUT, and CONSTRAINT_IGNORED. It returns
UNKNOWN for HALLUCINATED_STATE, GOAL_DRIFT, PLAN_INCOMPLETE, and
CONTEXT_OVERFLOW — those require semantic understanding and are handled by
LLMClassifier or HybridClassifier.
"""

from __future__ import annotations

import json
import re

from triage.taxonomy import FailureType
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


def _tool_input_key(tool_input: object) -> str:
    """Canonical string form of a tool_input dict for equality comparison."""
    if isinstance(tool_input, dict):
        return json.dumps(tool_input, sort_keys=True)
    return str(tool_input)


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
        Number of consecutive identical steps required to declare a loop.
        Default 3. Set higher (e.g. 4–5) if your agent legitimately repeats
        the same tool call twice in a row.
    """

    def __init__(
        self,
        constraints: list[str] | None = None,
        loop_window: int = 3,
    ) -> None:
        self.constraints: list[str] = constraints or []
        if loop_window < 2:
            raise ValueError("loop_window must be >= 2")
        self.loop_window = loop_window

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:  # noqa: ARG002
        steps = trajectory.steps

        # 1. LOOP_DETECTED — last loop_window steps share identical tool_called + tool_input
        if len(steps) >= self.loop_window:
            window = steps[-self.loop_window:]
            if (
                window[0].tool_called is not None
                and all(s.tool_called == window[0].tool_called for s in window)
                and all(
                    _tool_input_key(s.tool_input) == _tool_input_key(window[0].tool_input)
                    for s in window
                )
            ):
                return FailureType.LOOP_DETECTED

        # 2. WRONG_TOOL_CALLED
        for step in steps:
            if step.error and _WRONG_TOOL_RE.search(step.error):
                return FailureType.WRONG_TOOL_CALLED

        # 3. SCHEMA_MISMATCH
        for step in steps:
            if step.error and _SCHEMA_RE.search(step.error):
                return FailureType.SCHEMA_MISMATCH

        # 4. EXTERNAL_FAULT — HTTP status codes as whole tokens
        for step in steps:
            if step.error and _EXTERNAL_CODE_RE.search(step.error):
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

        # HALLUCINATED_STATE, GOAL_DRIFT, PLAN_INCOMPLETE, CONTEXT_OVERFLOW cannot
        # be detected by pattern matching — use LLMClassifier or HybridClassifier.
        return FailureType.UNKNOWN
