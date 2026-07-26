"""
tests/test_adapter_langchain.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the LangChain adapter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")
# Guard against langchain versions that have moved or removed AgentExecutor
try:
    from langchain.agents import AgentExecutor as _AgentExecutor  # noqa: F401
except ImportError:
    pytest.skip(
        "langchain.agents.AgentExecutor not available in this version",
        allow_module_level=True,
    )

from triage.adapters.langchain import wrap_langchain
from triage.policy import FailurePolicy


def _make_executor(output: str = "executor result") -> MagicMock:
    executor = MagicMock()
    executor.ainvoke = AsyncMock(return_value={"output": output})
    return executor


# ---------------------------------------------------------------------------
# Basic wrapping
# ---------------------------------------------------------------------------

async def test_wrap_langchain_returns_agent():
    from triage.agent import Agent
    executor = _make_executor()
    ag = wrap_langchain(executor, FailurePolicy())
    assert isinstance(ag, Agent)


async def test_executor_output_returned():
    executor = _make_executor("the answer")
    ag = wrap_langchain(executor, FailurePolicy())
    result = await ag.run("what is 1+1")
    assert result == "the answer"


async def test_callback_passed_to_ainvoke():
    executor = _make_executor()
    ag = wrap_langchain(executor, FailurePolicy())
    await ag.run("task")

    call_kwargs = executor.ainvoke.call_args[1]
    assert "config" in call_kwargs
    assert "callbacks" in call_kwargs["config"]
    assert len(call_kwargs["config"]["callbacks"]) == 1


async def test_tool_start_recorded():
    recorded_steps = []
    executor = MagicMock()

    async def _ainvoke(inputs, config=None):
        # Trigger the callback
        handler = config["callbacks"][0]
        handler.on_tool_start({"name": "calculator"}, "1+1")
        return {"output": "2"}

    executor.ainvoke = _ainvoke

    ag = wrap_langchain(executor, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def r(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=r, **kw)

    ag._fn = capturing_fn
    await ag.run("compute")

    tool_starts = [s for s in recorded_steps if "tool_start" in s.action]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_called == "calculator"
    assert tool_starts[0].tool_input == {"input": "1+1"}


async def test_tool_end_recorded():
    recorded_steps = []
    executor = MagicMock()

    async def _ainvoke(inputs, config=None):
        handler = config["callbacks"][0]
        handler.on_tool_end("computation complete")
        return {"output": "done"}

    executor.ainvoke = _ainvoke

    ag = wrap_langchain(executor, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def r(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=r, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    end_steps = [s for s in recorded_steps if s.action == "tool_end"]
    assert len(end_steps) == 1
    assert end_steps[0].tool_output == "computation complete"


async def test_tool_error_recorded():
    recorded_steps = []
    executor = MagicMock()

    async def _ainvoke(inputs, config=None):
        handler = config["callbacks"][0]
        handler.on_tool_error(ValueError("bad input"))
        return {"output": "done"}

    executor.ainvoke = _ainvoke

    ag = wrap_langchain(executor, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def r(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=r, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    err_steps = [s for s in recorded_steps if s.action == "tool_error"]
    assert len(err_steps) == 1
    assert "bad input" in err_steps[0].error


async def test_llm_end_recorded():
    recorded_steps = []
    executor = MagicMock()

    async def _ainvoke(inputs, config=None):
        handler = config["callbacks"][0]
        gen = MagicMock()
        gen.text = "I think the answer is 7."
        response = MagicMock()
        response.generations = [[gen]]
        handler.on_llm_end(response)
        return {"output": "done"}

    executor.ainvoke = _ainvoke

    ag = wrap_langchain(executor, FailurePolicy())
    original_fn = ag._fn

    async def capturing_fn(task, *, record_step, **kw):
        def r(step):
            recorded_steps.append(step)
        return await original_fn(task, record_step=r, **kw)

    ag._fn = capturing_fn
    await ag.run("task")

    llm_steps = [s for s in recorded_steps if s.action == "llm_end"]
    assert len(llm_steps) == 1
    assert "7" in llm_steps[0].llm_output


async def test_output_dict_without_output_key_returned_raw():
    executor = MagicMock()
    executor.ainvoke = AsyncMock(return_value={"result": "raw value"})
    ag = wrap_langchain(executor, FailurePolicy())
    result = await ag.run("task")
    assert result == {"result": "raw value"}
