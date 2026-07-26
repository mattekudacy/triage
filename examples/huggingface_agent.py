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

import ast
import asyncio
import json
import logging
import operator
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Run: pip install openai") from None

import triage  # noqa: E402
from triage.classifier.llm import LLMClassifier  # noqa: E402
from triage.strategies.replan import replan  # noqa: E402
from triage.strategies.retry import retry_with_tool_manifest  # noqa: E402
from triage.taxonomy import Step  # noqa: E402

HF_BASE_URL = "https://api-inference.huggingface.co/v1"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ── Tool ─────────────────────────────────────────────────────────────────────

_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a pure arithmetic expression without eval()."""

    def _visit(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_visit(node.left), _visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_visit(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _visit(ast.parse(expr, mode="eval").body)


def calculator(expression: str) -> str:
    try:
        return str(_safe_eval(expression))
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
