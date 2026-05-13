"""
triage.testing
~~~~~~~~~~~~~~
Utilities for testing triage-wrapped agents.

Intended to be used in your own test suite::

    from triage.testing import make_step, RecordingAgent, assert_classifies_as
"""

from __future__ import annotations

from typing import Any

from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict[str, Any] | None = None,
    error: str | None = None,
    llm_output: str | None = None,
    *,
    action: str = "test step",
) -> Step:
    """Build a ``Step`` with sensible defaults for use in tests.

    Mirrors the canonical local helper used across the triage test suite so
    user tests can import it instead of redefining it.
    """
    return Step(
        index=index,
        action=action,
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
    )


class RecordingAgent:
    """Async agent callable that records every call and fails a configurable
    number of times before succeeding.

    Useful for asserting that triage injects the right recovery context on
    subsequent attempts::

        agent_fn = RecordingAgent(succeed_after=1)
        ag = triage.Agent(agent_fn, policy=policy)
        await ag.run("task")

        # Second call received the structured recovery context
        tc = agent_fn.calls[1].get("_triage_context")
        assert tc is not None
        assert tc.failure_type == FailureType.UNKNOWN

    Parameters
    ----------
    succeed_after:
        How many attempts to fail before returning ``result``. Default 0
        (succeeds on the first call).
    error:
        Exception to raise on each failing attempt. Defaults to
        ``RuntimeError("synthetic failure")``.
    result:
        Value returned on success. Defaults to ``"ok"``.
    """

    def __init__(
        self,
        *,
        succeed_after: int = 0,
        error: Exception | None = None,
        result: Any = "ok",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._succeed_after = succeed_after
        self._error: Exception = error if error is not None else RuntimeError("synthetic failure")
        self._result = result

    async def __call__(self, task: str, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if len(self.calls) <= self._succeed_after:
            raise self._error
        return self._result


def assert_classifies_as(
    steps: list[Step],
    task: str,
    expected: FailureType,
    classifier: Classifier | None = None,
) -> None:
    """Assert that a trajectory classifies as ``expected``.

    Raises ``AssertionError`` with a descriptive message on mismatch.

    Parameters
    ----------
    steps:
        Ordered list of ``Step`` objects forming the trajectory.
    task:
        The original task string (used by some classifiers as context).
    expected:
        The ``FailureType`` the trajectory should produce.
    classifier:
        Classifier to use. Defaults to a fresh ``RulesClassifier``.
    """
    clf: Classifier = classifier if classifier is not None else RulesClassifier()
    traj = Trajectory()
    for step in steps:
        traj.append(step)
    actual = clf.classify(traj, task)
    if actual != expected:
        raise AssertionError(
            f"Expected trajectory to classify as {expected.value!r}, "
            f"but got {actual.value!r}.\n"
            f"Steps: {[s.action for s in steps]}"
        )
