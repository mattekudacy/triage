# CrewAI Adapter

Wraps a CrewAI `Crew` with triage failure classification and recovery.

## Install

```bash
pip install "triage-agent[crewai]"
```

## Usage

```python
import triage
from triage.adapters.crewai import wrap_crewai
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan

from crewai import Agent, Crew, Task

researcher = Agent(role="Researcher", goal="Find accurate information", backstory="...")
task = Task(description="Research the topic: {task}", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])

policy = triage.FailurePolicy(
    EXTERNAL_FAULT = backoff_and_retry(max_attempts=3),
    LOOP_DETECTED  = replan(hint="The crew is repeating itself. Change approach."),
    default        = triage.FailurePolicy.escalate_by_default(),
)

agent = wrap_crewai(crew, policy=policy)
result = await agent.run("Latest AI research papers")
```

## How it works

The adapter patches `crew.step_callback` for the duration of each `kickoff_async()` call. The original callback (if any) is restored in a `finally` block.

Each `step_callback` invocation records a `Step`:

| Field | Source |
|---|---|
| `action` | `"crewai_step:<StepOutput type name>"` |
| `tool_called` | `step_output.tool` |
| `tool_input` | `{"input": step_output.tool_input}` |
| `llm_output` | `step_output.log` or `step_output.output` |

The crew's final output is extracted from `result.raw` (or `str(result)` as fallback).

## Signature

```python
def wrap_crewai(
    crew: Crew,
    policy: FailurePolicy,
    **kwargs,               # passed to triage.Agent.__init__
) -> triage.Agent
```

## Notes

- **Not safe for concurrent calls on the same `Crew` instance.** `step_callback` patching is not thread-safe. Run one task at a time per crew.
- Requires `crewai>=0.1`
- The `task` string is passed as `inputs={"task": task}` to `kickoff_async()`. Make sure your task descriptions reference `{task}` if they need it.
