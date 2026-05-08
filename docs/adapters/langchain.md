# LangChain Adapter

Wraps a LangChain `AgentExecutor` with triage failure classification and recovery.

## Install

```bash
pip install "triage-agent[langchain]"
```

## Usage

```python
import triage
from triage.adapters.langchain import wrap_langchain
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.strategies.replan import replan

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_openai_tools_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT    = backoff_and_retry(max_attempts=3),
    LOOP_DETECTED     = replan(hint="Try a different approach."),
    default           = triage.FailurePolicy.escalate_by_default(),
)

agent = wrap_langchain(executor, policy=policy)
result = await agent.run("What is the capital of France?")
```

## How it works

The adapter creates a fresh `BaseCallbackHandler` subclass per call and passes it via `config={"callbacks": [TriageCallbackHandler()]}` to `executor.ainvoke()`. This ensures callback state doesn't leak between calls.

The callback records steps for:

| LangChain callback | triage Step |
|---|---|
| `on_tool_start` | `action="tool_start:<name>"`, `tool_called=<name>`, `tool_input={"input": ...}` |
| `on_tool_end` | `action="tool_end"`, `tool_output=<output>` |
| `on_tool_error` | `action="tool_error"`, `error=<error string>` |
| `on_llm_end` | `action="llm_end"`, `llm_output=<first generation text>` |

The executor's final output is extracted from `result["output"]`.

## Signature

```python
def wrap_langchain(
    executor: AgentExecutor,
    policy: FailurePolicy,
    **kwargs,               # passed to triage.Agent.__init__
) -> triage.Agent
```

## Notes

- Requires `langchain>=0.1` and `langchain-core>=0.1`
- Extra kwargs passed to `agent.run()` are merged into the `ainvoke()` input dict
- A fresh callback handler is created on every `agent.run()` call — no shared state
