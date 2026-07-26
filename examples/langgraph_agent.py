"""
examples/langgraph_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: wrapping a LangGraph agent with triage using wrap_langgraph().

Run with:
    pip install "triage-agent[langgraph]" langchain-openai
    OPENAI_API_KEY=sk-... python examples/langgraph_agent.py

What happens:
  A simple LangGraph ReAct agent is built with a calculator tool.
  wrap_langgraph() streams per-step events via astream_events so triage
  can observe every tool call and LLM turn.

  On the first attempt the tool raises a deliberate schema error, triage
  classifies it as SCHEMA_MISMATCH, and retries with a corrective hint.
  The second attempt succeeds.
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n  pip install 'triage-agent[langgraph]' langchain-openai"
    ) from None

import triage  # noqa: E402
from triage.adapters.langgraph import wrap_langgraph  # noqa: E402
from triage.strategies.replan import replan  # noqa: E402
from triage.strategies.retry import retry_with_tool_manifest  # noqa: E402

# ── Tool definition ───────────────────────────────────────────────────────────

_fail_once = [True]


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '42 * 17'."""
    if _fail_once[0]:
        _fail_once[0] = False
        # Simulate a validation error to trigger SCHEMA_MISMATCH
        raise ValueError(
            "validation error: expression must not contain spaces — got: " + repr(expression)
        )
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ── Build the LangGraph agent ─────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
graph = create_react_agent(llm, tools=[calculator])

# ── Wire up triage ────────────────────────────────────────────────────────────

policy = triage.FailurePolicy(
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=3),
    EXTERNAL_FAULT=replan(hint="The external service may be down. Try a simpler approach."),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = wrap_langgraph(graph, policy=policy, auto_checkpoint=True)


async def main() -> None:
    task = "What is 42 * 17?"
    print(f"\nTask: {task}\n")
    try:
        result = await agent.run(task)
        if isinstance(result, dict) and "messages" in result:
            last = result["messages"][-1]
            print(f"\nResult: {getattr(last, 'content', last)}")
        else:
            print(f"\nResult: {result}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
