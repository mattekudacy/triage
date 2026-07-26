"""
examples/raw_openai.py
~~~~~~~~~~~~~~~~~~~~~~
End-to-end demo: triage catching and recovering a WRONG_TOOL_CALLED failure.

Run with:
    pip install triage-agent openai
    OPENAI_API_KEY=sk-... python examples/raw_openai.py

What happens:
  Attempt 1 — the agent deliberately passes a non-existent tool name
              ("nonexistent_calculator"). The handler raises an error,
              triage classifies it as WRONG_TOOL_CALLED, and dispatches
              retry_with_tool_manifest().
  Attempt 2 — the agent uses the correct "calculator" tool and succeeds.
"""

from __future__ import annotations

import asyncio
import json
import logging

import openai

import triage
from triage.strategies.retry import retry_with_tool_manifest
from triage.taxonomy import Step

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Tool manifest (correct version) ──────────────────────────────────────────

CORRECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. '42 * 17'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

BAD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nonexistent_calculator",
            "description": "Does not exist.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def run_calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ── Agent function ────────────────────────────────────────────────────────────


async def openai_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    client = openai.AsyncOpenAI()
    messages: list[dict] = [{"role": "user", "content": task}]

    if _triage_hint:
        messages.insert(0, {"role": "system", "content": f"Hint from triage: {_triage_hint}"})

    # First call (no hint) deliberately uses the bad tool manifest to trigger
    # WRONG_TOOL_CALLED. Subsequent calls (with hint injected by triage) use
    # the correct manifest.
    tools = CORRECT_TOOLS if _triage_hint else BAD_TOOLS

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if not message.tool_calls:
        # Model responded without a tool call
        step = Step(index=0, action="llm response (no tool call)", llm_output=message.content)
        record_step(step)
        return message.content or ""

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    valid_names = {t["function"]["name"] for t in CORRECT_TOOLS}

    if tool_name not in valid_names:
        err = f"no tool named {tool_name!r}"
        record_step(
            Step(
                index=0,
                action=f"attempted to call {tool_name!r}",
                tool_called=tool_name,
                tool_input=json.loads(tool_call.function.arguments or "{}"),
                error=err,
            )
        )
        raise RuntimeError(err)

    # Execute the real tool
    args = json.loads(tool_call.function.arguments)
    result = run_calculator(args["expression"])

    record_step(
        Step(
            index=0,
            action=f"called {tool_name!r}",
            tool_called=tool_name,
            tool_input=args,
            tool_output=result,
        )
    )
    return f"Result: {result}"


# ── Wire up triage and run ────────────────────────────────────────────────────

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

wrapped = triage.Agent(openai_agent, policy=policy)


async def main() -> None:
    task = "What is 42 * 17?"
    print(f"\nTask: {task}\n")
    try:
        answer = await wrapped.run(task)
        print(f"\n{answer}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated: {exc}")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
