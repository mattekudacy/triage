"""
examples/anthropic_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: triage with Claude (Anthropic) — tool use + LOOP_DETECTED recovery.

Requirements:
    pip install "triage-agent[anthropic]"

Run with:
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_agent.py

What happens:
  An agent uses Claude Haiku to answer a question with a calculator tool.
  On the first two attempts the agent is nudged into calling the same tool
  with the same input repeatedly, triggering LOOP_DETECTED. triage routes
  to replan(), injects a hint, and the third attempt succeeds.

  LLMClassifier is also Claude Haiku — used for semantic failure
  classification instead of the default RulesClassifier, though
  LOOP_DETECTED is also detectable by RulesClassifier for comparison.
"""

from __future__ import annotations

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import anthropic
except ImportError:
    raise SystemExit("Run: pip install 'triage-agent[anthropic]'")

import triage
from triage.classifier.llm import LLMClassifier
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

MODEL = "claude-haiku-4-5-20251001"

# ── Tool ─────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '12 * 9'"}
            },
            "required": ["expression"],
        },
    }
]

# ── Agent ─────────────────────────────────────────────────────────────────────

_attempt = [0]


async def anthropic_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1
    client = anthropic.Anthropic()

    system = "You are a helpful assistant. Use the calculator tool for arithmetic."
    if _triage_hint:
        system += f"\n\nRecovery hint: {_triage_hint}"

    messages = [{"role": "user", "content": task}]

    step_index = 0
    for _ in range(5):  # cap iterations
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any tool uses
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses:
            # Final answer
            text = " ".join(b.text for b in text_blocks)
            record_step(Step(index=step_index, action="final_answer", llm_output=text))
            return text

        # First two attempts: force a loop by always using the same expression
        tool_results = []
        for tool_use in tool_uses:
            expr = tool_use.input.get("expression", "")

            # Simulate loop: first two attempts ignore the actual expression
            if _attempt[0] <= 2:
                expr = "1 + 1"  # always the same — triggers LOOP_DETECTED

            result = calculator(expr)
            record_step(Step(
                index=step_index,
                action=f"tool_call:{tool_use.name}",
                tool_called=tool_use.name,
                tool_input={"expression": expr},
                tool_output=result,
            ))
            step_index += 1
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached."


# ── Wire up triage ────────────────────────────────────────────────────────────

classifier = LLMClassifier(model=MODEL, max_trajectory_steps=6)

policy = triage.FailurePolicy(
    LOOP_DETECTED=replan(hint="You are repeating the same tool call. Use the actual numbers from the task."),
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(
    anthropic_agent,
    policy=policy,
    classifier=classifier,
    max_recovery_attempts=3,
)


async def main() -> None:
    task = "What is 12 * 9?"
    print(f"\nTask: {task}")
    print(f"Model: {MODEL} (Anthropic)\n")
    try:
        result = await agent.run(task)
        print(f"\n{result}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated: {exc}")
        print(f"  failure_type: {exc.context.failure_type.value}")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
