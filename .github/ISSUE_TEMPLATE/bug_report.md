---
name: Bug report
about: Something is broken or behaving unexpectedly
labels: bug
---

**triage-agent version**
<!-- pip show triage-agent -->

**Python version**

**Description**
<!-- What went wrong? What did you expect? -->

**Minimal reproduction**

```python
import triage
from triage.taxonomy import Step

async def my_agent(task, *, record_step, **kwargs):
    ...

policy = triage.FailurePolicy(...)
agent = triage.Agent(my_agent, policy=policy)
# await agent.run(...)
```

**Error output**

```
Traceback (most recent call last):
  ...
```

**Additional context**
<!-- Framework (LangGraph, raw OpenAI, etc.), checkpoint store, classifier used -->
