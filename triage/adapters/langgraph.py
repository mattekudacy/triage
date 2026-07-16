"""
triage.adapters.langgraph
~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps a compiled LangGraph ``StateGraph`` with triage failure recovery.

Install: pip install triage-agent[langgraph]
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.graph.state import CompiledStateGraph
except ImportError as exc:
    raise ImportError(
        "LangGraph adapter requires 'langgraph'. "
        "Install it with: pip install triage-agent[langgraph]"
    ) from exc

from triage.agent import Agent
from triage.policy import FailurePolicy
from triage.taxonomy import Step


def wrap_langgraph(
    graph: CompiledStateGraph,
    policy: FailurePolicy,
    **kwargs: Any,
) -> Agent:
    """Wrap a compiled LangGraph graph with triage recovery.

    Streams events via ``graph.astream_events(..., version="v2")`` to
    capture per-step tool calls and LLM outputs before returning the
    final chain output.
    """
    async def wrapped_fn(task: str, *, record_step: Any, **kw: Any) -> Any:
        i = 0
        final_output: Any = None
        graph_name = getattr(graph, "name", "graph")
        async for event in graph.astream_events(
            {"messages": [("user", task)]}, version="v2", **kw
        ):
            etype = event["event"]
            if etype == "on_tool_start":
                record_step(Step(
                    index=i,
                    action=f"tool_start:{event['name']}",
                    tool_called=event["name"],
                    tool_input=event["data"].get("input"),
                ))
                i += 1
            elif etype == "on_tool_end":
                err_val = event["data"].get("error")
                record_step(Step(
                    index=i,
                    action=f"tool_end:{event['name']}",
                    tool_called=event["name"],
                    tool_output=event["data"].get("output"),
                    error=str(err_val) if err_val else None,
                ))
                i += 1
            elif etype == "on_chat_model_end":
                content = str(event["data"]["output"].content)
                record_step(Step(index=i, action="llm_turn", llm_output=content))
                i += 1
            elif etype == "on_chain_end" and event.get("name") == graph_name:
                final_output = event["data"].get("output")
        return final_output

    return Agent(wrapped_fn, policy, **kwargs)
