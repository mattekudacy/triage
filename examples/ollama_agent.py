"""
examples/ollama_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~
Demo: triage with a local Ollama model — no API key required.

Requirements:
    pip install "triage-agent" openai
    # Install and start Ollama: https://ollama.com
    ollama pull llama3.2
    ollama serve

Run with:
    python examples/ollama_agent.py

What happens:
  An agent queries a local Llama 3.2 model via Ollama's OpenAI-compatible
  API to answer a factual question with a calculator tool.

  On the first attempt the agent deliberately passes bad JSON to the tool,
  triggering a SCHEMA_MISMATCH. LLMClassifier (pointed at Ollama) classifies
  the failure semantically, and triage retries with a corrective hint.

  Both the agent LLM and the classifier LLM run locally — this example
  makes zero external API calls.
"""

from __future__ import annotations

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Run: pip install openai") from None

import triage  # noqa: E402
from triage.classifier.llm import LLMClassifier  # noqa: E402
from triage.strategies.retry import retry_with_tool_manifest  # noqa: E402
from triage.taxonomy import Step  # noqa: E402

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

# ── Tool ─────────────────────────────────────────────────────────────────────


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ── Agent ─────────────────────────────────────────────────────────────────────

_attempt = [0]


async def ollama_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    system = "You are a helpful assistant. Use the calculator tool to answer math questions."
    if _triage_hint:
        system += f"\n\nRecovery hint: {_triage_hint}"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "e.g. '6 * 7'"}},
                    "required": ["expression"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if not message.tool_calls:
        record_step(Step(index=0, action="llm_response", llm_output=message.content or ""))
        return message.content or ""

    tool_call = message.tool_calls[0]
    raw_args = tool_call.function.arguments

    # First attempt: simulate bad JSON to trigger SCHEMA_MISMATCH
    if _attempt[0] == 1:
        raw_args = "not valid json {"

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        record_step(
            Step(
                index=0,
                action="tool_call:calculator",
                tool_called="calculator",
                error=f"JSONDecodeError: {exc}",
            )
        )
        raise RuntimeError(f"JSONDecodeError: {exc}") from exc

    result = calculator(args["expression"])
    record_step(
        Step(
            index=0,
            action="tool_call:calculator",
            tool_called="calculator",
            tool_input=args,
            tool_output=result,
        )
    )
    return f"Result: {result}"


# ── Wire up triage ────────────────────────────────────────────────────────────

classifier = LLMClassifier(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    max_trajectory_steps=5,
)

policy = triage.FailurePolicy(
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(ollama_agent, policy=policy, classifier=classifier)


async def main() -> None:
    task = "What is 6 * 7?"
    print(f"\nTask: {task}")
    print(f"Model: {MODEL} via Ollama ({OLLAMA_BASE_URL})\n")
    try:
        result = await agent.run(task)
        print(f"\n{result}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated: {exc}")
        print(f"  failure_type: {exc.context.failure_type.value}")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")
    except Exception as exc:
        print(f"\nError: {exc}")
        print("Is Ollama running? Try: ollama serve")


if __name__ == "__main__":
    asyncio.run(main())
