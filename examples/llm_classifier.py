"""
examples/llm_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo: swapping in LLMClassifier for semantic failure classification.

Run with:
    pip install "triage-agent[anthropic]"
    ANTHROPIC_API_KEY=sk-ant-... python examples/llm_classifier.py

What happens:
  An agent fails with an ambiguous error message that the RulesClassifier
  would classify as UNKNOWN. LLMClassifier asks Claude to classify it
  semantically and returns the correct FailureType, which routes to a
  more specific recovery strategy.

  No real LLM agent is needed — the agent function is synthetic. Only
  the classifier makes an API call (via the Anthropic sync client).
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

import triage  # noqa: E402
from triage.strategies.replan import replan  # noqa: E402
from triage.strategies.retry import backoff_and_retry  # noqa: E402
from triage.taxonomy import Step  # noqa: E402

try:
    from triage.classifier.llm import LLMClassifier
except ImportError:
    raise SystemExit(
        "Missing dependency. Run:\n"
        "  pip install 'triage-agent[anthropic]'"
    ) from None

# ── Synthetic agent ───────────────────────────────────────────────────────────

_attempt = [0]


async def research_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _attempt[0] += 1

    if _triage_hint:
        print(f"  [agent] Received recovery hint: {_triage_hint!r}")

    # First attempt: simulate the agent drifting from the original goal
    if _attempt[0] == 1:
        record_step(Step(
            index=0,
            action="web_search",
            tool_called="search",
            tool_input={"q": "unrelated topic"},
            llm_output="I got distracted and started researching something else entirely.",
        ))
        raise RuntimeError(
            "The agent appears to have deviated from the original objective "
            "and is now pursuing an unrelated sub-task."
        )

    # Second attempt succeeds
    record_step(Step(index=0, action="web_search", tool_called="search",
                     tool_input={"q": task}, tool_output="relevant results"))
    return f"Completed: {task}"


# ── Wire up triage with LLMClassifier ────────────────────────────────────────

classifier = LLMClassifier(
    model="claude-haiku-4-5-20251001",
    max_trajectory_steps=10,
)

policy = triage.FailurePolicy(
    CONSTRAINT_IGNORED=replan(hint="Stay focused on the original task. Do not pursue sub-topics."),
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(
    research_agent,
    policy=policy,
    classifier=classifier,
    max_recovery_attempts=3,
)


async def main() -> None:
    task = "Summarise the latest research on transformer architectures."
    print(f"\nTask: {task}")
    print("Classifier: LLMClassifier (claude-haiku-4-5-20251001)\n")
    try:
        result = await agent.run(task)
        print(f"\n{result}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated: {exc}")
        print(f"  failure_type : {exc.context.failure_type.value}")
        print(f"  trajectory   : {len(exc.context.trajectory)} step(s)")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
