"""
tests/test_adapter_langgraph.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the LangGraph adapter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langgraph")

from triage.adapters.langgraph import wrap_langgraph
from triage.policy import FailurePolicy


def _make_graph(events: list[dict]) -> MagicMock:
    graph = MagicMock()
    graph.name = "test_graph"

    async def _astream_events(*args, **kwargs):
        for event in events:
            yield event

    graph.astream_events = _astream_events
    return graph


# ---------------------------------------------------------------------------
# Basic wrapping
# ---------------------------------------------------------------------------

async def test_wrap_langgraph_returns_agent():
    from triage.agent import Agent
    graph = _make_graph([])
    ag = wrap_langgraph(graph, FailurePolicy())
    assert isinstance(ag, Agent)


async def test_final_output_from_chain_end():
    events = [
        {
            "event": "on_chain_end",
            "name": "test_graph",
            "data": {"output": {"answer": "42"}},
        }
    ]
    graph = _make_graph(events)
    ag = wrap_langgraph(graph, FailurePolicy())
    result = await ag.run("what is the answer")
    assert result == {"answer": "42"}


async def test_tool_start_recorded_as_step():
    recorded_steps = []

    events = [
        {
            "event": "on_tool_start",
            "name": "search",
            "data": {"input": {"query": "hello"}},
        },
        {
            "event": "on_chain_end",
            "name": "test_graph",
            "data": {"output": "done"},
        },
    ]
    graph = _make_graph(events)

    async def agent_fn(task, *, record_step, **kw):
        pass  # replaced by wrapper

    ag = wrap_langgraph(graph, FailurePolicy())
    # Patch record_step to capture
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def capturing_record(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=capturing_record, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    tool_steps = [s for s in recorded_steps if "tool_start" in s.action]
    assert len(tool_steps) == 1
    assert tool_steps[0].tool_called == "search"


async def test_tool_end_recorded_as_step():
    recorded_steps = []

    events = [
        {
            "event": "on_tool_end",
            "name": "search",
            "data": {"output": "results", "error": None},
        },
        {
            "event": "on_chain_end",
            "name": "test_graph",
            "data": {"output": "done"},
        },
    ]
    graph = _make_graph(events)
    ag = wrap_langgraph(graph, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def capturing_record(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=capturing_record, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    end_steps = [s for s in recorded_steps if "tool_end" in s.action]
    assert len(end_steps) == 1
    assert end_steps[0].tool_output == "results"


async def test_llm_turn_recorded_as_step():
    recorded_steps = []

    output_mock = MagicMock()
    output_mock.content = "Here is my answer."
    events = [
        {
            "event": "on_chat_model_end",
            "name": "model",
            "data": {"output": output_mock},
        },
        {
            "event": "on_chain_end",
            "name": "test_graph",
            "data": {"output": "done"},
        },
    ]
    graph = _make_graph(events)
    ag = wrap_langgraph(graph, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def capturing_record(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=capturing_record, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    llm_steps = [s for s in recorded_steps if s.action == "llm_turn"]
    assert len(llm_steps) == 1
    assert "Here is my answer." in llm_steps[0].llm_output


async def test_tool_end_with_error_sets_error_field():
    recorded_steps = []

    events = [
        {
            "event": "on_tool_end",
            "name": "search",
            "data": {"output": None, "error": ValueError("not found")},
        },
        {
            "event": "on_chain_end",
            "name": "test_graph",
            "data": {"output": "done"},
        },
    ]
    graph = _make_graph(events)
    ag = wrap_langgraph(graph, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def capturing_record(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=capturing_record, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    end_steps = [s for s in recorded_steps if "tool_end" in s.action]
    assert end_steps[0].error is not None
    assert "not found" in end_steps[0].error
