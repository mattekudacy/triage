"""
examples/huggingface_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: triage with HuggingFace Inference API (OpenAI-compatible endpoint).

Requirements:
    pip install "triage-agent" openai
    # Get a free token at https://huggingface.co/settings/tokens

Run with:
    HF_TOKEN=hf_... python examples/huggingface_agent.py

What happens:
  An agent uses Llama 3.2 3B on HuggingFace's serverless Inference API to
  answer a question with a calculator tool.

  On the first attempt the tool receives malformed arguments, triggering a
  SCHEMA_MISMATCH. LLMClassifier (also on HuggingFace) classifies the failure
  semantically, triage retries with a corrective hint, and the second attempt
  succeeds.

  Both the agent and the classifier use the same HF token — no other API
  key is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Run: pip install openai")

import triage
from triage.classifier.llm import LLMClassifier
from triage.strategies.retry import retry_with_tool_manifest
from triage.strategies.replan import replan
from triage.taxonomy import Step

HF_BASE_URL = "https://api-inference.huggingface.co/v1"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ── Tool ─────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ── Agent ─────────────────────────────────────────────────────────────────────

_attempt = [0]


async def hf_agent(
    task: str,
    *,
    record_step,
    update_state,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable.")

    client = OpenAI(base_url=HF_BASE_URL, api_key=token)

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
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression, e.g. '15 * 4'",
                        }
                    },
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
        max_tokens=256,
    )

    message = response.choices[0].message

    if not message.tool_calls:
        text = message.content or ""
        record_step(Step(index=0, action="llm_response", llm_output=text))
        return text

    tool_call = message.tool_calls[0]
    raw_args = tool_call.function.arguments

    # First attempt: inject bad JSON to trigger SCHEMA_MISMATCH
    if _attempt[0] == 1:
        raw_args = "{expression: missing quotes}"

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        record_step(Step(
            index=0,
            action="tool_call:calculator",
            tool_called="calculator",
            error=f"JSONDecodeError: {exc}",
        ))
        raise RuntimeError(f"JSONDecodeError: {exc}") from exc

    result = calculator(args["expression"])
    record_step(Step(
        index=0,
        action="tool_call:calculator",
        tool_called="calculator",
        tool_input=args,
        tool_output=result,
    ))
    update_state({"last_result": result})
    return f"Result: {result}"


# ── Wire up triage ────────────────────────────────────────────────────────────

def _make_classifier() -> LLMClassifier:
    token = os.environ.get("HF_TOKEN", "")
    return LLMClassifier(
        base_url=HF_BASE_URL,
        api_key=token,
        model=MODEL,
        max_trajectory_steps=5,
    )


policy = triage.FailurePolicy(
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=3),
    EXTERNAL_FAULT=replan(hint="The HuggingFace API may be overloaded. Simplify the request."),
    default=triage.FailurePolicy.escalate_by_default(),
)


async def main() -> None:
    task = "What is 15 * 4?"
    print(f"\nTask: {task}")
    print(f"Model: {MODEL} via HuggingFace Inference API\n")

    agent = triage.Agent(
        hf_agent,
        policy=policy,
        classifier=_make_classifier(),
        max_recovery_attempts=3,
    )

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
        print("Check your HF_TOKEN and that the model supports tool use.")


if __name__ == "__main__":
    asyncio.run(main())
