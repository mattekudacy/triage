"""
triage.adapters.langchain
~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps a LangChain ``AgentExecutor`` with triage failure recovery.

Install: pip install triage-agent[langchain]
"""

from __future__ import annotations

from typing import Any

from triage.agent import Agent
from triage.policy import FailurePolicy
from triage.taxonomy import Step


def wrap_langchain(
    executor: Any,
    policy: FailurePolicy,
    **kwargs: Any,
) -> Agent:
    """Wrap a LangChain AgentExecutor with triage recovery.

    Creates a fresh ``BaseCallbackHandler`` per call that records tool starts,
    tool ends, tool errors, and LLM completions as triage Steps.
    """
    async def wrapped_fn(task: str, *, record_step: Any, **kw: Any) -> Any:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError as exc:
            raise ImportError(
                "LangChain adapter requires 'langchain' and 'langchain-core'. "
                "Install them with: pip install triage-agent[langchain]"
            ) from exc

        step_index = 0

        class TriageCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
            def on_tool_start(
                self, serialized: dict[str, Any], input_str: str, **cb_kw: Any
            ) -> None:
                nonlocal step_index
                name = serialized.get("name", "unknown_tool")
                record_step(Step(
                    index=step_index,
                    action=f"tool_start:{name}",
                    tool_called=name,
                    tool_input={"input": input_str},
                ))
                step_index += 1

            def on_tool_end(self, output: str, **cb_kw: Any) -> None:
                nonlocal step_index
                record_step(Step(
                    index=step_index,
                    action="tool_end",
                    tool_output=output,
                ))
                step_index += 1

            def on_tool_error(
                self, error: BaseException | str, **cb_kw: Any
            ) -> None:
                nonlocal step_index
                record_step(Step(
                    index=step_index,
                    action="tool_error",
                    error=str(error),
                ))
                step_index += 1

            def on_llm_end(self, response: Any, **cb_kw: Any) -> None:
                nonlocal step_index
                try:
                    text = response.generations[0][0].text
                except (IndexError, AttributeError):
                    text = str(response)
                record_step(Step(index=step_index, action="llm_end", llm_output=text))
                step_index += 1

        result = await executor.ainvoke(
            {"input": task, **kw},
            config={"callbacks": [TriageCallbackHandler()]},
        )
        return result.get("output", result)

    return Agent(wrapped_fn, policy, **kwargs)
