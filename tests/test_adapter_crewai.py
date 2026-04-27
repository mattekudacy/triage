"""
tests/test_adapter_crewai.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the CrewAI adapter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("crewai")

from triage.adapters.crewai import wrap_crewai
from triage.policy import FailurePolicy


def _make_crew(output: str = "crew result") -> MagicMock:
    crew = MagicMock()
    crew.step_callback = None
    result_mock = MagicMock()
    result_mock.raw = output
    crew.kickoff_async = AsyncMock(return_value=result_mock)
    return crew


# ---------------------------------------------------------------------------
# Basic wrapping
# ---------------------------------------------------------------------------

async def test_wrap_crewai_returns_agent():
    from triage.agent import Agent
    crew = _make_crew()
    ag = wrap_crewai(crew, FailurePolicy())
    assert isinstance(ag, Agent)


async def test_crew_result_returned():
    crew = _make_crew("my crew output")
    ag = wrap_crewai(crew, FailurePolicy())
    result = await ag.run("task")
    assert result == "my crew output"


async def test_step_callback_invoked_for_each_step():
    recorded_steps = []
    crew = _make_crew()

    async def _kickoff(inputs=None):
        # Simulate the crew calling the step_callback during execution
        for i in range(3):
            step_out = MagicMock()
            step_out.tool = "search"
            step_out.tool_input = "query"
            step_out.log = f"step {i} result"
            crew.step_callback(step_out)
        result = MagicMock()
        result.raw = "done"
        return result

    crew.kickoff_async = _kickoff

    ag = wrap_crewai(crew, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def capturing_record(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=capturing_record, **kw)

    ag._fn = capturing_fn
    await ag.run("task")
    assert len(recorded_steps) == 3


async def test_original_callback_restored_after_run():
    original_cb = MagicMock()
    crew = _make_crew()
    crew.step_callback = original_cb

    ag = wrap_crewai(crew, FailurePolicy())
    await ag.run("task")

    assert crew.step_callback is original_cb


async def test_original_callback_restored_on_exception():
    original_cb = MagicMock()
    crew = _make_crew()
    crew.step_callback = original_cb
    crew.kickoff_async = AsyncMock(side_effect=RuntimeError("crew crashed"))

    from triage.policy import RecoveryAction as RA

    async def _abort(ctx):
        return RA.ABORT(reason="abort")

    policy = FailurePolicy(UNKNOWN=_abort)
    ag = wrap_crewai(crew, policy)
    with pytest.raises(Exception):
        await ag.run("task")

    assert crew.step_callback is original_cb


async def test_step_action_includes_type_name():
    recorded_steps = []
    crew = _make_crew()

    class MyStepOutput:
        tool = "calc"
        tool_input = "1+1"
        log = "2"

    async def _kickoff(inputs=None):
        crew.step_callback(MyStepOutput())
        result = MagicMock()
        result.raw = "done"
        return result

    crew.kickoff_async = _kickoff
    ag = wrap_crewai(crew, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def r(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=r, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    assert "MyStepOutput" in recorded_steps[0].action


async def test_result_raw_fallback_to_str():
    crew = MagicMock()
    crew.step_callback = None
    result_mock = MagicMock(spec=[])  # no .raw attribute
    result_mock.__str__ = lambda self: "str result"
    crew.kickoff_async = AsyncMock(return_value=result_mock)

    ag = wrap_crewai(crew, FailurePolicy())
    result = await ag.run("task")
    assert result == "str result"
