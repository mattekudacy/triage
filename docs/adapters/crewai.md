# CrewAI Adapter

> **Removed in v0.7.** The CrewAI adapter was removed because it relied on patching
> `crew.step_callback`, an internal that changed across CrewAI versions and broke silently.
>
> To use triage with CrewAI, wrap a plain async callable around your crew's `kickoff_async()`
> and use `triage.Agent` directly:

```python
import triage
from triage.taxonomy import Step
from crewai import Crew

crew = Crew(...)

async def crew_agent(task: str, *, record_step, **kwargs) -> str:
    result = await crew.kickoff_async(inputs={"task": task})
    record_step(Step(index=0, action="crew_kickoff", tool_output=str(result.raw)))
    return str(result.raw)

policy = triage.FailurePolicy(...)
agent = triage.Agent(crew_agent, policy=policy)
result = await agent.run("your task")
```

This pattern gives you full triage coverage without depending on CrewAI internals.
