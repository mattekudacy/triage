"""
tests/test_adapter_openai_agents.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the OpenAI Agents SDK adapter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("agents")

from triage.adapters.openai_agents import wrap_openai_agents
from triage.policy import FailurePolicy


def _make_stream_event(item):
    from agents.stream_events import RunItemStreamEvent
    evt = MagicMock(spec=RunItemStreamEvent)
    evt.item = item
    return evt


def _make_tool_call_item(name: str, arguments: str = "{}"):
    from agents.items import ToolCallItem
    item = MagicMock(spec=ToolCallItem)
    item.raw_item = MagicMock()
    item.raw_item.name = name
    item.raw_item.arguments = arguments
    return item


def _make_tool_output_item(call_id: str, output: str):
    from agents.items import ToolCallOutputItem
    item = MagicMock(spec=ToolCallOutputItem)
    item.raw_item = MagicMock()
    item.raw_item.call_id = call_id
    item.output = output
    return item


def _make_message_output_item(text: str):
    from agents.items import MessageOutputItem
    item = MagicMock(spec=MessageOutputItem)
    item.raw_item = MagicMock()
    content_part = MagicMock()
    content_part.text = text
    item.raw_item.content = [content_part]
    return item


def _make_runner_result(events, final_output="done"):
    result = MagicMock()
    result.final_output = final_output

    async def _stream():
        for evt in events:
            yield evt

    result.stream_events = _stream
    return result


# ---------------------------------------------------------------------------
# Basic wrapping
# ---------------------------------------------------------------------------

async def test_wrap_openai_agents_returns_triage_agent():
    from triage.agent import Agent as TriageAgent
    sdk_agent = MagicMock()
    ag = wrap_openai_agents(sdk_agent, FailurePolicy())
    assert isinstance(ag, TriageAgent)


async def test_final_output_returned():
    events = []
    result = _make_runner_result(events, final_output="my answer")

    sdk_agent = MagicMock()
    with patch("triage.adapters.openai_agents.Runner") as MockRunner:
        MockRunner.run_streamed.return_value = result
        ag = wrap_openai_agents(sdk_agent, FailurePolicy())
        output = await ag.run("task")
    assert output == "my answer"


async def test_tool_call_step_recorded():
    recorded_steps = []
    tool_item = _make_tool_call_item("web_search", '{"query": "test"}')
    events = [_make_stream_event(tool_item)]
    result = _make_runner_result(events)

    sdk_agent = MagicMock()
    with patch("triage.adapters.openai_agents.Runner") as MockRunner:
        MockRunner.run_streamed.return_value = result
        ag = wrap_openai_agents(sdk_agent, FailurePolicy())
        original_fn = ag._fn

        async def capturing_fn(task, *, record_step, **kw):
            def r(step):
                recorded_steps.append(step)
            return await original_fn(task, record_step=r, **kw)

        ag._fn = capturing_fn
        await ag.run("task")

    tool_steps = [s for s in recorded_steps if "tool_call" in s.action]
    assert len(tool_steps) == 1
    assert tool_steps[0].tool_called == "web_search"


async def test_tool_output_step_recorded():
    recorded_steps = []
    output_item = _make_tool_output_item("call_123", "search results")
    events = [_make_stream_event(output_item)]
    result = _make_runner_result(events)

    sdk_agent = MagicMock()
    with patch("triage.adapters.openai_agents.Runner") as MockRunner:
        MockRunner.run_streamed.return_value = result
        ag = wrap_openai_agents(sdk_agent, FailurePolicy())
        original_fn = ag._fn

        async def capturing_fn(task, *, record_step, **kw):
            def r(step):
                recorded_steps.append(step)
            return await original_fn(task, record_step=r, **kw)

        ag._fn = capturing_fn
        await ag.run("task")

    output_steps = [s for s in recorded_steps if "tool_output" in s.action]
    assert len(output_steps) == 1
    assert output_steps[0].tool_output == "search results"


async def test_message_output_step_recorded():
    recorded_steps = []
    msg_item = _make_message_output_item("The answer is 42.")
    events = [_make_stream_event(msg_item)]
    result = _make_runner_result(events)

    sdk_agent = MagicMock()
    with patch("triage.adapters.openai_agents.Runner") as MockRunner:
        MockRunner.run_streamed.return_value = result
        ag = wrap_openai_agents(sdk_agent, FailurePolicy())
        original_fn = ag._fn

        async def capturing_fn(task, *, record_step, **kw):
            def r(step):
                recorded_steps.append(step)
            return await original_fn(task, record_step=r, **kw)

        ag._fn = capturing_fn
        await ag.run("task")

    llm_steps = [s for s in recorded_steps if s.action == "llm_message"]
    assert len(llm_steps) == 1
    assert "42" in llm_steps[0].llm_output


async def test_non_run_item_events_ignored():
    recorded_steps = []
    from agents.stream_events import RunItemStreamEvent

    # An event that is NOT a RunItemStreamEvent — should be skipped
    other_event = MagicMock()
    other_event.__class__ = type("OtherEvent", (), {})  # not RunItemStreamEvent

    events = [other_event]
    result = _make_runner_result(events)

    sdk_agent = MagicMock()
    with patch("triage.adapters.openai_agents.Runner") as MockRunner:
        MockRunner.run_streamed.return_value = result
        ag = wrap_openai_agents(sdk_agent, FailurePolicy())
        original_fn = ag._fn

        async def capturing_fn(task, *, record_step, **kw):
            def r(step):
                recorded_steps.append(step)
            return await original_fn(task, record_step=r, **kw)

        ag._fn = capturing_fn
        await ag.run("task")

    assert len(recorded_steps) == 0
