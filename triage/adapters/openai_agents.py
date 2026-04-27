"""
triage.adapters.openai_agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps an OpenAI Agents SDK ``Agent`` with triage failure recovery.

Install: pip install triage-agent[openai-agents]
"""

from __future__ import annotations

from typing import Any

try:
    from agents import Agent as SDKAgent, Runner
    from agents.stream_events import RunItemStreamEvent
    from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem
except ImportError as exc:
    raise ImportError(
        "OpenAI Agents adapter requires 'openai-agents'. "
        "Install it with: pip install triage-agent[openai-agents]"
    ) from exc

from triage.agent import Agent as TriageAgent
from triage.policy import FailurePolicy
from triage.taxonomy import Step


def wrap_openai_agents(
    agent_obj: "SDKAgent",
    policy: FailurePolicy,
    **kwargs: Any,
) -> TriageAgent:
    """Wrap an OpenAI Agents SDK Agent with triage recovery.

    Uses ``Runner.run_streamed`` and iterates ``stream_events()`` to capture
    tool calls, tool outputs, and LLM message outputs.
    """
    async def wrapped_fn(task: str, *, record_step: Any, **kw: Any) -> Any:
        i = 0
        result = Runner.run_streamed(agent_obj, task, **kw)
        async for event in result.stream_events():
            if not isinstance(event, RunItemStreamEvent):
                continue
            item = event.item
            if isinstance(item, ToolCallItem):
                record_step(Step(
                    index=i,
                    action=f"tool_call:{item.raw_item.name}",
                    tool_called=item.raw_item.name,
                    tool_input={"arguments": item.raw_item.arguments},
                ))
                i += 1
            elif isinstance(item, ToolCallOutputItem):
                record_step(Step(
                    index=i,
                    action=f"tool_output:{item.raw_item.call_id}",
                    tool_output=str(item.output),
                ))
                i += 1
            elif isinstance(item, MessageOutputItem):
                content = "".join(
                    c.text for c in item.raw_item.content
                    if hasattr(c, "text")
                )
                record_step(Step(index=i, action="llm_message", llm_output=content))
                i += 1
        return result.final_output

    return TriageAgent(wrapped_fn, policy, **kwargs)
