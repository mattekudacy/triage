# OpenAI Agents SDK Adapter

> **Removed in v0.7.** The OpenAI Agents SDK adapter was removed because the SDK was
> still pre-stable and its streaming API changed in ways that broke the adapter across
> minor versions.
>
> To use triage with the OpenAI Agents SDK, wrap a plain async callable around `Runner.run()`:

```python
import triage
from triage.taxonomy import Step
from agents import Agent, Runner

sdk_agent = Agent(name="MyAgent", instructions="...", tools=[...])

async def openai_agent(task: str, *, record_step, **kwargs) -> str:
    result = await Runner.run(sdk_agent, task)
    record_step(Step(index=0, action="runner_run", tool_output=str(result.final_output)))
    return str(result.final_output)

policy = triage.FailurePolicy(...)
agent = triage.Agent(openai_agent, policy=policy)
result = await agent.run("your task")
```

This pattern gives you full triage coverage without depending on SDK internals.
