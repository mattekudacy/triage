"""
tests/test_classifier_hybrid.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for HybridClassifier — rules first, LLM fallback on UNKNOWN.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from triage.classifier.hybrid import HybridClassifier
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
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    # WRONG_TOOL_CALLED — rules catches this
    step = make_step(0, error="no tool named 'missing_tool'")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.WRONG_TOOL_CALLED
    llm.classify.assert_not_called()


def test_schema_mismatch_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="validation error: field 'name' is required")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.SCHEMA_MISMATCH
    llm.classify.assert_not_called()


def test_external_fault_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="received 429 Too Many Requests")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.EXTERNAL_FAULT
    llm.classify.assert_not_called()


def test_loop_detected_handled_by_rules_without_llm():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
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
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="something ambiguous happened")
    result = clf.classify(traj(step), "task")

    assert result == FailureType.PLAN_INCOMPLETE
    llm.classify.assert_called_once()


def test_llm_result_returned_on_unknown():
    for ft in FailureType:
        llm = _mock_llm(ft)
        clf = HybridClassifier(llm=llm)
        step = make_step(0, error="ambiguous")
        result = clf.classify(traj(step), "task")
        assert result == ft


def test_llm_receives_same_trajectory_and_task():
    llm = _mock_llm(FailureType.CONTEXT_OVERFLOW)
    clf = HybridClassifier(llm=llm)

    t = traj(make_step(0, error="unclear"))
    clf.classify(t, "my task")

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


# ---------------------------------------------------------------------------
# aclassify() — async counterpart, prefers llm.aclassify() when present
# ---------------------------------------------------------------------------

async def test_aclassify_rules_result_returned_without_calling_llm():
    llm = MagicMock()
    llm.aclassify = AsyncMock(return_value=FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="no tool named 'missing_tool'")
    result = await clf.aclassify(traj(step), "task")

    assert result == FailureType.WRONG_TOOL_CALLED
    llm.aclassify.assert_not_called()


async def test_aclassify_uses_llm_aclassify_when_available():
    llm = MagicMock()
    llm.aclassify = AsyncMock(return_value=FailureType.CONTEXT_OVERFLOW)
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="ambiguous")
    result = await clf.aclassify(traj(step), "task")

    assert result == FailureType.CONTEXT_OVERFLOW
    llm.aclassify.assert_called_once()
    llm.classify.assert_not_called()


async def test_aclassify_falls_back_to_sync_classify_when_llm_has_no_aclassify():
    """LLM object without an aclassify() method — hybrid falls back to classify()."""
    llm = MagicMock(spec=["classify"])
    llm.classify.return_value = FailureType.EXTERNAL_FAULT
    clf = HybridClassifier(llm=llm)

    step = make_step(0, error="ambiguous")
    result = await clf.aclassify(traj(step), "task")

    assert result == FailureType.EXTERNAL_FAULT
    llm.classify.assert_called_once()


# ---------------------------------------------------------------------------
# max_llm_calls_per_run — cost cap
# ---------------------------------------------------------------------------

def test_max_llm_calls_per_run_none_by_default_allows_unlimited_calls():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm)

    for _ in range(10):
        clf.classify(traj(make_step(0, error="ambiguous")), "task")

    assert llm.classify.call_count == 10


def test_classify_stops_calling_llm_once_cap_reached():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=2)

    results = [clf.classify(traj(make_step(0, error="ambiguous")), "task") for _ in range(4)]

    assert results == [
        FailureType.PLAN_INCOMPLETE,
        FailureType.PLAN_INCOMPLETE,
        FailureType.UNKNOWN,
        FailureType.UNKNOWN,
    ]
    assert llm.classify.call_count == 2


async def test_aclassify_stops_calling_llm_once_cap_reached():
    llm = MagicMock()
    llm.aclassify = AsyncMock(return_value=FailureType.CONTEXT_OVERFLOW)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=1)

    first = await clf.aclassify(traj(make_step(0, error="ambiguous")), "task")
    second = await clf.aclassify(traj(make_step(0, error="ambiguous")), "task")

    assert first == FailureType.CONTEXT_OVERFLOW
    assert second == FailureType.UNKNOWN
    llm.aclassify.assert_called_once()


def test_max_llm_calls_per_run_zero_disables_llm_entirely():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=0)

    result = clf.classify(traj(make_step(0, error="ambiguous")), "task")

    assert result == FailureType.UNKNOWN
    llm.classify.assert_not_called()


def test_reset_call_count_restores_budget():
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=1)

    clf.classify(traj(make_step(0, error="ambiguous")), "task")
    capped = clf.classify(traj(make_step(0, error="ambiguous")), "task")
    assert capped == FailureType.UNKNOWN

    clf.reset_call_count()
    after_reset = clf.classify(traj(make_step(0, error="ambiguous")), "task")

    assert after_reset == FailureType.PLAN_INCOMPLETE
    assert llm.classify.call_count == 2


def test_rules_hits_do_not_consume_llm_call_budget():
    """A rules-resolved failure must not count against max_llm_calls_per_run."""
    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=1)

    # WRONG_TOOL_CALLED — resolved by rules, no LLM call
    for _ in range(5):
        result = clf.classify(traj(make_step(0, error="no tool named 'missing_tool'")), "task")
        assert result == FailureType.WRONG_TOOL_CALLED

    llm.classify.assert_not_called()

    # The LLM budget (1 call) is still fully available
    result = clf.classify(traj(make_step(0, error="ambiguous")), "task")
    assert result == FailureType.PLAN_INCOMPLETE
    llm.classify.assert_called_once()


def test_cap_persists_across_calls_dispatched_via_to_thread():
    """Regression: the counter must be visible across anyio.to_thread.run_sync
    dispatches, not just within a single synchronous call chain. A ContextVar-
    based counter would silently reset to 0 on every thread dispatch and let
    the cap never engage — see HybridClassifier's docstring on why this uses a
    threading.Lock-guarded plain counter instead.
    """
    import anyio

    llm = _mock_llm(FailureType.PLAN_INCOMPLETE)
    clf = HybridClassifier(llm=llm, max_llm_calls_per_run=1)
    t = traj(make_step(0, error="ambiguous"))

    async def main():
        results = []
        for _ in range(3):
            results.append(await anyio.to_thread.run_sync(clf.classify, t, "task"))
        return results

    results = anyio.run(main)

    assert results == [
        FailureType.PLAN_INCOMPLETE,
        FailureType.UNKNOWN,
        FailureType.UNKNOWN,
    ]
    assert llm.classify.call_count == 1
