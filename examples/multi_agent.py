"""
examples/multi_agent.py
~~~~~~~~~~~~~~~~~~~~~~~
Demo: multi-agent pipeline with triage context propagation.

Run with:
    python examples/multi_agent.py

What this shows:

  A two-level agent pipeline:

    orchestrator
      └─ researcher  (triage-wrapped)
      └─ writer      (triage-wrapped)

  The researcher agent hits a rate limit (EXTERNAL_FAULT) and exhausts its
  recovery budget, raising TriageEscalationError. The orchestrator catches
  it as a chained exception.

  Because triage propagates failure context through exception chaining
  (exc.__cause__), the outer orchestrator's triage instance reuses the
  child's failure_type instead of re-classifying. The outer policy then
  decides to replan — routing to a fallback research strategy — rather than
  escalating to the user.

  This demonstrates:
    - Child TriageEscalationError context propagation
    - Per-agent policies with different recovery budgets
    - Agent.clone() for safe concurrent re-use
"""

from __future__ import annotations

import asyncio
import logging

import triage
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Researcher: hits rate limits, limited budget ──────────────────────────────

_researcher_calls = [0]


async def researcher_agent(
    task: str, *, record_step, _triage_hint: str | None = None, **_kwargs
) -> str:
    _researcher_calls[0] += 1
    call = _researcher_calls[0]

    if _triage_hint:
        print(f"    [researcher] hint: {_triage_hint!r}")

    # First two calls: simulate rate limiting
    if call <= 2:
        record_step(
            Step(
                index=0,
                action="web_search",
                tool_called="search",
                tool_input={"q": task},
                error="HTTP 429 Too Many Requests — rate limit exceeded",
            )
        )
        raise RuntimeError("HTTP 429 Too Many Requests — rate limit exceeded")

    # Third call (fallback strategy, different query): succeeds
    record_step(
        Step(
            index=0,
            action="web_search",
            tool_called="search",
            tool_input={"q": f"cached: {task}"},
            tool_output="research results from cache",
        )
    )
    return f"Research on '{task}': found 3 relevant papers (from cache)."


researcher_policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=1),  # try once, then escalate up
    default=triage.FailurePolicy.escalate_by_default(),
)

_researcher_agent = triage.Agent(
    researcher_agent,
    policy=researcher_policy,
    max_recovery_attempts=1,
)


# ── Writer: depends on researcher output ─────────────────────────────────────


async def writer_agent(task: str, *, record_step, research: str = "", **_kwargs) -> str:
    record_step(
        Step(
            index=0,
            action="draft_section",
            tool_called="write",
            tool_input={"topic": task, "research": research[:80]},
            tool_output="draft written",
        )
    )
    return f"Written section for '{task}' using: {research[:60]}..."


writer_policy = triage.FailurePolicy(
    default=triage.FailurePolicy.escalate_by_default(),
)

_writer_agent = triage.Agent(writer_agent, policy=writer_policy)


# ── Orchestrator ──────────────────────────────────────────────────────────────

_orch_attempt = [0]


async def orchestrator_agent(
    task: str,
    *,
    record_step,
    _triage_hint: str | None = None,
    **_kwargs,
) -> str:
    _orch_attempt[0] += 1
    print(f"\n  [orchestrator] attempt {_orch_attempt[0]}")

    if _triage_hint:
        print(f"  [orchestrator] hint: {_triage_hint!r}")

    # On the first attempt, use the normal search strategy.
    # On a replan (hint present), fall back to cached results.
    search_task = f"cached: {task}" if _triage_hint else task

    # Step 1: delegate to researcher
    print(f"  [orchestrator] launching researcher for: {search_task!r}")
    try:
        # Clone so concurrent runs don't share state
        research = await _researcher_agent.clone().run(search_task)
    except triage.TriageEscalationError as exc:
        # Re-raise as a chained exception so the outer triage agent can
        # inspect exc.__cause__ and reuse the child's failure_type.
        raise RuntimeError(f"Researcher pipeline exhausted: {exc}") from exc

    record_step(
        Step(
            index=0,
            action="research_complete",
            tool_called="researcher",
            tool_input={"task": search_task},
            tool_output=research[:80],
        )
    )

    # Step 2: delegate to writer
    print("  [orchestrator] launching writer")
    draft = await _writer_agent.clone().run(task, research=research)

    record_step(
        Step(
            index=1,
            action="draft_complete",
            tool_called="writer",
            tool_input={"task": task},
            tool_output=draft[:80],
        )
    )

    return draft


orchestrator_policy = triage.FailurePolicy(
    # Child escalated EXTERNAL_FAULT — replan with a fallback hint
    EXTERNAL_FAULT=replan(hint="Primary search is rate-limited. Use cached results instead."),
    default=triage.FailurePolicy.escalate_by_default(),
)

orchestrator = triage.Agent(
    orchestrator_agent,
    policy=orchestrator_policy,
    max_recovery_attempts=2,
)


# ── Run ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    task = "recent advances in protein folding"
    print(f"\nTask: {task}")
    print("Pipeline: orchestrator → researcher + writer\n")

    try:
        result = await orchestrator.run(task)
        print(f"\nResult:\n  {result}")
    except triage.TriageEscalationError as exc:
        print(f"\nEscalated to human: {exc}")
        if exc.context:
            print(f"  failure_type : {exc.context.failure_type.value}")
            print(f"  attempts     : {len(exc.context.attempt_history)}")
    except triage.TriageAbortError as exc:
        print(f"\nAborted: {exc}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
