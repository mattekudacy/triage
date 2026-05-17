# Framework Adapters

triage ships adapters for two popular agent frameworks. Each adapter wraps the framework's native agent object and returns a `triage.Agent` that adds failure classification and recovery on top.

## How adapters work

Every adapter exposes one function:

```python
wrap_<framework>(agent_obj, policy, **kwargs) -> triage.Agent
```

The `**kwargs` are passed through to `triage.Agent.__init__`, so you can configure the classifier, checkpoint store, max recovery attempts, and auto-checkpoint alongside the policy.

The adapter intercepts the framework's native step events (tool calls, LLM outputs, errors) and translates them into `triage.Step` objects by calling `record_step()`. The framework's own error handling is bypassed — triage's recovery loop takes over.

## Available adapters

| Adapter | Install | Function |
|---|---|---|
| [LangGraph](langgraph.md) | `pip install "triage-agent[langgraph]"` | `wrap_langgraph(graph, policy)` |
| [LangChain](langchain.md) | `pip install "triage-agent[langchain]"` | `wrap_langchain(executor, policy)` |

## Common pattern

```python
import triage
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan

policy = triage.FailurePolicy(
    EXTERNAL_FAULT = backoff_and_retry(max_attempts=3),
    LOOP_DETECTED  = replan(hint="Try a different approach."),
    default        = triage.FailurePolicy.escalate_by_default(),
)

# Swap the adapter for your framework — everything else is identical
from triage.adapters.langgraph import wrap_langgraph
agent = wrap_langgraph(compiled_graph, policy=policy, auto_checkpoint=True)

result = await agent.run("Summarize the latest earnings report.")
```

## Lazy imports

Framework imports are lazy — the adapter module only raises `ImportError` at import time if the optional dependency is missing. Installing `triage-agent` without any extras does not pull in LangGraph or any other framework.
