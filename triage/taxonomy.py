"""
triage.taxonomy
~~~~~~~~~~~~~~~
Failure type enum and the FailureContext dataclass passed to every
recovery hook and strategy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(Enum):
    """Semantic classification of why an agent step or plan failed.

    Each type maps to a distinct recovery strategy. The classifier
    (rules-based or LLM) assigns one of these to every detected failure.
    """

    # Agent called a tool that doesn't exist or isn't in the current manifest
    WRONG_TOOL_CALLED = "wrong_tool_called"

    # Output violates an explicit constraint from the original task
    CONSTRAINT_IGNORED = "constraint_ignored"

    # Agent is repeating the same action or plan branch across N steps
    LOOP_DETECTED = "loop_detected"

    # Agent declared success but not all required sub-goals were completed
    PLAN_INCOMPLETE = "plan_incomplete"

    # Tool output or LLM response didn't match the expected structure/schema
    SCHEMA_MISMATCH = "schema_mismatch"

    # Agent lost track of earlier task context due to long-horizon drift
    CONTEXT_OVERFLOW = "context_overflow"

    # Tool/API returned an error that is not the agent's fault (rate limit, 500)
    EXTERNAL_FAULT = "external_fault"

    # Agent hit a wall-clock or async deadline (asyncio.TimeoutError, anyio, trio)
    TIMEOUT = "timeout"

    # Could not determine failure type
    UNKNOWN = "unknown"


@dataclass
class Step:
    """A single recorded step in an agent's execution trajectory."""

    index: int
    action: str
    tool_called: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None
    llm_output: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    state_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotent: bool = False
    partial: bool = False


@dataclass
class FailureContext:
    """Everything a recovery strategy needs to know about a failure.

    Passed to every strategy callable and raised inside TriageEscalationError
    and TriageAbortError.
    """

    failure_type: FailureType
    trajectory: list[Step]
    critical_step_index: int
    original_task: str
    last_checkpoint_id: str | None = None
    loop_steps: list[int] | None = None
    violated_constraint: str | None = None
    expected_schema: dict[str, Any] | None = None
    raw_error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attempt_history: list[tuple[FailureType, str]] = field(default_factory=list)

    @property
    def failed_step(self) -> Step | None:
        """The step at the critical failure index."""
        if 0 <= self.critical_step_index < len(self.trajectory):
            return self.trajectory[self.critical_step_index]
        return None

    @property
    def steps_after_failure(self) -> list[Step]:
        """All steps that executed after the critical failure."""
        return self.trajectory[self.critical_step_index + 1:]


@dataclass
class TriageContext:
    """Structured recovery context injected into agents as ``_triage_context``.

    Replaces the scattered ``_triage_hint``, ``_triage_subgoal``, and
    ``_triage_state`` kwargs with a single typed object. The individual kwargs
    are still injected for backward compatibility.

    Usage inside a wrapped agent::

        async def my_agent(task: str, *, record_step, update_state, **kwargs) -> Any:
            ctx: TriageContext | None = kwargs.get("_triage_context")
            if ctx:
                print(ctx.failure_type, ctx.hint, ctx.attempt_number)
    """

    failure_type: FailureType
    attempt_number: int
    hint: str | None = None
    subgoal: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
