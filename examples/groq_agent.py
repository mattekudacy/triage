"""
examples/groq_agent.py
~~~~~~~~~~~~~~~~~~~~~~
Demo: triage with Groq (Llama 3.1, cloud, free tier).

Requirements:
    pip install "triage-agent" openai
    # Get a free API key at https://console.groq.com

Run with:
    GROQ_API_KEY=gsk_... python examples/groq_agent.py

What happens:
  An agent uses Llama 3.1 8B on Groq to answer a question using a search
  tool. On the first attempt the tool raises a simulated 503 error.
  triage classifies it as EXTERNAL_FAULT and retries with backoff.
  The second attempt succeeds.

  LLMClassifier is also pointed at Groq so both agent and classifier use
  the same provider — no Anthropic key needed.
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Run: pip install openai")

import triage
from triage.classifier.llm import LLMClassifier
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan
from triage.taxonomy import Step

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.1-8b-instant"

# ── Tool ─────────────────────────────────────────────────────────────────────

_search_call = [0]


def web_search(query: str) -> str:
    _search_call[0] += 1
    if _search_call[0] == 1:
        # Simulate a transient 503 from the search provider
        raise RuntimeError("503 Service Unavailable: search backend is overloaded")
    return f"Search results for '{query}': [result 1, result 2, result 3]"


# ── Agent ─────────────────────────────────────────────────────────────────────

async def groq_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable.")

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)

    system = "You are a helpful research assistant. Use the web_search tool to find information."
    if _triage_hint:
        system += f"\n\nRecovery hint: {_triage_hint}"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
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

    import json
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    try:
        result = web_search(args["query"])
    except RuntimeError as exc:
        record_step(Step(
            index=0,
            action="tool_call:web_search",
            tool_called="web_search",
            tool_input=args,
            error=str(exc),
        ))
        raise

    record_step(Step(
        index=0,
        action="tool_call:web_search",
        tool_called="web_search",
        tool_input=args,
        tool_output=result,
    ))

    # Get final answer from the model
    followup = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            },
        ],
    )
    return followup.choices[0].message.content or result


# ── Wire up triage ────────────────────────────────────────────────────────────

def _make_classifier() -> LLMClassifier:
    api_key = os.environ.get("GROQ_API_KEY", "")
    return LLMClassifier(base_url=GROQ_BASE_URL, api_key=api_key, model=MODEL)


policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    GOAL_DRIFT=replan(hint="Stay focused on the original task."),
    default=triage.FailurePolicy.escalate_by_default(),
)


async def main() -> None:
    task = "What are the key differences between GPT-4 and Llama 3?"
    print(f"\nTask: {task}")
    print(f"Model: {MODEL} via Groq ({GROQ_BASE_URL})\n")

    agent = triage.Agent(
        groq_agent,
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


if __name__ == "__main__":
    asyncio.run(main())
