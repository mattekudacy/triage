"""
triage.classifier.rules
~~~~~~~~~~~~~~~~~~~~~~~
Heuristic, synchronous, zero-API-calls classifier. MVP default.
Rules are evaluated in priority order; first match wins.
"""

from __future__ import annotations

import json
import re

from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

# Compiled patterns for speed (called on every failure)
_WRONG_TOOL_RE = re.compile(r"tool.{0,30}not found|no tool named", re.IGNORECASE)
_SCHEMA_RE = re.compile(r"validation error|json.*parse|jsondecodeerror", re.IGNORECASE)
_EXTERNAL_CODES = ("429", "500", "502", "503")


def _tool_input_key(tool_input: object) -> str:
    """Canonical string form of a tool_input dict for equality comparison."""
    if isinstance(tool_input, dict):
        return json.dumps(tool_input, sort_keys=True)
    return str(tool_input)


class RulesClassifier:
    """Pattern-based classifier. Instantiate with optional constraint strings."""

    def __init__(self, constraints: list[str] | None = None) -> None:
        self.constraints: list[str] = constraints or []

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:  # noqa: ARG002
        steps = trajectory.steps

        # 1. LOOP_DETECTED — last 3 steps identical tool_called + tool_input
        if len(steps) >= 3:
            last3 = steps[-3:]
            if (
                last3[0].tool_called is not None
                and all(s.tool_called == last3[0].tool_called for s in last3)
                and all(
                    _tool_input_key(s.tool_input) == _tool_input_key(last3[0].tool_input)
                    for s in last3
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

        # 4. EXTERNAL_FAULT — HTTP status codes in error message
        for step in steps:
            if step.error and any(code in step.error for code in _EXTERNAL_CODES):
                return FailureType.EXTERNAL_FAULT

        # 5. CONSTRAINT_IGNORED — llm_output contains a forbidden constraint string
        if self.constraints:
            for step in steps:
                if step.llm_output:
                    output_lower = step.llm_output.lower()
                    for constraint in self.constraints:
                        if constraint.lower() in output_lower:
                            return FailureType.CONSTRAINT_IGNORED

        return FailureType.UNKNOWN
