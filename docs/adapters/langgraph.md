# LangGraph Adapter

Wraps a compiled LangGraph `StateGraph` with triage failure classification and recovery.

## Install

```bash
pip install "triage-agent[langgraph]"
```

## Usage

```python
import triage
from triage.adapters.langgraph import wrap_langgraph
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan

# Your compiled LangGraph graph
from my_graph import graph   # CompiledStateGraph

policy = triage.FailurePolicy(
    EXTERNAL_FAULT = backoff_and_retry(max_attempts=3),
    LOOP_DETECTED  = replan(hint="Try a different approach."),
    default        = triage.FailurePolicy.escalate_by_default(),
)

agent = wrap_langgraph(graph, policy=policy)
result = await agent.run("Summarize the latest earnings report.")
```

## How it works

The adapter calls `graph.astream_events({"messages": [("user", task)]}, version="v2")` and maps each event to a `Step`:

| LangGraph event | triage Step |
|---|---|
| `on_tool_start` | `action="tool_start:<name>"`, `tool_called=<name>`, `tool_input=<input>` |
| `on_tool_end` | `action="tool_end:<name>"`, `tool_output=<output>`, `error=<error>` |
| `on_chat_model_end` | `action="llm_turn"`, `llm_output=<content>` |
| `on_chain_end` (graph name) | Captures final output |

## Signature

```python
def wrap_langgraph(
    graph: CompiledStateGraph,
    policy: FailurePolicy,
    **kwargs,               # passed to triage.Agent.__init__
) -> triage.Agent
```

## Passing kwargs to the graph

Any extra keyword arguments passed to `agent.run()` beyond `task` are forwarded to `graph.astream_events()`:

```python
result = await agent.run("task", config={"recursion_limit": 10})
```

## Notes

- Requires `langgraph>=0.2` for `astream_events` v2 support
- The adapter captures the graph's final output from the last `on_chain_end` event for the top-level graph
- If the graph raises, triage catches the exception and classifies it before dispatching recovery
