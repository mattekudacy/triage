"""
triage.adapters.crewai
~~~~~~~~~~~~~~~~~~~~~~
Wraps a CrewAI ``Crew`` with triage failure recovery.

Install: pip install triage-agent[crewai]
"""

from __future__ import annotations

from typing import Any

try:
    from crewai import Crew
except ImportError as exc:
    raise ImportError(
        "CrewAI adapter requires 'crewai'. "
        "Install it with: pip install triage-agent[crewai]"
    ) from exc

from triage.agent import Agent
from triage.policy import FailurePolicy
from triage.taxonomy import Step


def wrap_crewai(
    crew: "Crew",
    policy: FailurePolicy,
    **kwargs: Any,
) -> Agent:
    """Wrap a CrewAI Crew with triage recovery.

    Patches ``crew.step_callback`` for the duration of each call to capture
    per-step outputs. The original callback (if any) is restored in ``finally``.

    Note: not safe for concurrent calls on the same Crew instance.
    """
    async def wrapped_fn(task: str, *, record_step: Any, **kw: Any) -> Any:
        step_index = 0

        def _callback(step_output: Any) -> None:
            nonlocal step_index
            tool = getattr(step_output, "tool", None)
            tool_input = getattr(step_output, "tool_input", None)
            llm_out = getattr(step_output, "log", None) or getattr(step_output, "output", None)
            record_step(Step(
                index=step_index,
                action=f"crewai_step:{type(step_output).__name__}",
                tool_called=str(tool) if tool else None,
                tool_input={"input": str(tool_input)} if tool_input else None,
                llm_output=str(llm_out) if llm_out else None,
            ))
            step_index += 1

        original = getattr(crew, "step_callback", None)
        crew.step_callback = _callback
        try:
            result = await crew.kickoff_async(inputs={"task": task, **kw})
        finally:
            crew.step_callback = original

        return result.raw if hasattr(result, "raw") else str(result)

    return Agent(wrapped_fn, policy, **kwargs)
