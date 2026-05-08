"""
tests/test_classifier_hybrid.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for HybridClassifier — rules first, LLM fallback on UNKNOWN.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from triage.classifier.hybrid import HybridClassifier
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


def _mock_llm(return_value: FailureType) -> MagicMock:
    llm = MagicMock()
    llm.classify.return_value = return_value
    return llm


# ---------------------------------------------------------------------------
# Rules handles it — LLM never called
# ---------------------------------------------------------------------------

def test_rules_result_returned_without_calling_llm():
    llm = _mock_llm(FailureType.GOAL_DRIFT)
    clf = HybridClassifier(llm=llm)

    # WRONG_TOOL_CALLED — rules catches this
    step = make_step(0, error="no tool named 'missing_tool'")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.WRONG_TOOL_CALLED
    llm.classify.assert_not_called()


def test_schema_mismatch_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.GOAL_DRIFT)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="validation error: field 'name' is required")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.SCHEMA_MISMATCH
    llm.classify.assert_not_called()


def test_external_fault_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.GOAL_DRIFT)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="received 429 Too Many Requests")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.EXTERNAL_FAULT
    llm.classify.assert_not_called()


def test_loop_detected_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.GOAL_DRIFT)
    clf = HybridClassifier(llm=llm)

    steps = [
        make_step(i, tool_called="search", tool_input={"q": "same"})
        for i in range(3)
    ]
    result = clf.classify(traj(*steps), "task")

    assert result == FailureType.LOOP_DETECTED
    llm.classify.assert_not_called()


# ---------------------------------------------------------------------------
# Rules returns UNKNOWN — LLM called exactly once
# ---------------------------------------------------------------------------

def test_llm_called_when_rules_returns_unknown():
    llm = _mock_llm(FailureType.GOAL_DRIFT)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="something ambiguous happened")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.GOAL_DRIFT
    llm.classify.assert_called_once()


def test_llm_result_returned_on_unknown():
    for ft in FailureType:
        llm = _mock_llm(ft)
        clf = HybridClassifier(llm=llm)
        step = make_step(0, error="ambiguous")
        result = clf.classify(traj(step), "task")
        assert result == ft


def test_llm_receives_same_trajectory_and_task():
    llm = _mock_llm(FailureType.HALLUCINATED_STATE)
    clf = HybridClassifier(llm=llm)

    t = traj(make_step(0, error="unclear"))
    result = clf.classify(t, "my task")

    call_args = llm.classify.call_args
    assert call_args[0][0] is t
    assert call_args[0][1] == "my task"


# ---------------------------------------------------------------------------
# LLM also returns UNKNOWN
# ---------------------------------------------------------------------------

def test_unknown_returned_when_both_return_unknown():
    llm = _mock_llm(FailureType.UNKNOWN)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="still ambiguous")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.UNKNOWN
    llm.classify.assert_called_once()


# ---------------------------------------------------------------------------
# LLM error propagates — agent.py catches it via to_thread wrapper
# ---------------------------------------------------------------------------

def test_llm_exception_propagates_from_hybrid():
    llm = MagicMock()
    llm.classify.side_effect = Exception("LLM unavailable")
    clf = HybridClassifier(llm=llm)

    clf._rules = MagicMock()
    clf._rules.classify.return_value = FailureType.UNKNOWN

    # HybridClassifier does not swallow LLM errors — agent.py handles that
    with pytest.raises(Exception, match="LLM unavailable"):
        clf.classify(traj(make_step(0)), "task")


# ---------------------------------------------------------------------------
# Satisfies Classifier protocol
# ---------------------------------------------------------------------------

def test_hybrid_satisfies_classifier_protocol():
    from triage.classifier.base import Classifier
    llm = _mock_llm(FailureType.UNKNOWN)
    clf = HybridClassifier(llm=llm)
    assert isinstance(clf, Classifier)
