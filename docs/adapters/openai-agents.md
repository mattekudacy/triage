# OpenAI Agents SDK Adapter

Wraps an OpenAI Agents SDK `Agent` with triage failure classification and recovery.

## Install

```bash
pip install "triage-agent[openai-agents]"
```

## Usage

```python
import triage
from triage.adapters.openai_agents import wrap_openai_agents
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.strategies.replan import replan

from agents import Agent, function_tool

@function_tool
def search_web(query: str) -> str:
    return f"Results for: {query}"

sdk_agent = Agent(
    name="ResearchAgent",
    instructions="You are a helpful research assistant.",
    tools=[search_web],
)

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT    = backoff_and_retry(max_attempts=3),
    LOOP_DETECTED     = replan(hint="Try a different search strategy."),
    default           = triage.FailurePolicy.escalate_by_default(),
)

agent = wrap_openai_agents(sdk_agent, policy=policy)
result = await agent.run("What are the latest developments in quantum computing?")
```

## How it works

The adapter uses `Runner.run_streamed(agent_obj, task)` and iterates `result.stream_events()`, filtering for `RunItemStreamEvent` items:

| Item type | triage Step |
|---|---|
| `ToolCallItem` | `action="tool_call:<name>"`, `tool_called=<name>`, `tool_input={"arguments": ...}` |
| `ToolCallOutputItem` | `action="tool_output:<call_id>"`, `tool_output=<output>` |
| `MessageOutputItem` | `action="llm_message"`, `llm_output=<content>` |

The final output is `result.final_output`.

## Signature

```python
def wrap_openai_agents(
    agent_obj: Agent,       # SDK Agent (not triage.Agent)
    policy: FailurePolicy,
    **kwargs,               # passed to triage.Agent.__init__
) -> triage.Agent
```

!!! note "Name collision"
    The adapter imports the SDK's `Agent` as `SDKAgent` internally to avoid shadowing `triage.Agent`. In your code, `wrap_openai_agents` returns a `triage.Agent`.

## Notes

- Requires `openai-agents>=0.0.3`
- Extra kwargs passed to `agent.run()` are forwarded to `Runner.run_streamed()`
