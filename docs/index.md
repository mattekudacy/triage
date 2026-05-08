# triage

**Classify why your agent failed. Recover intelligently.**

```bash
pip install triage-agent
```

---

Current agent frameworks know *that* your agent failed. They don't know *why* — and without knowing why, every failure gets the same blunt response: retry from scratch or give up.

`triage` adds a classification-and-routing layer between the failure and the recovery:

```
agent fails → classify failure type → route to matching strategy → recover
```

It wraps any async agent callable — OpenAI, LangGraph, CrewAI, raw LLM loops — without requiring you to change your framework.

## At a glance

```python
import triage
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan
from triage.strategies.rollback import rollback_to_checkpoint
from triage.taxonomy import Step

async def my_agent(task: str, *, record_step, update_state, **kwargs):
    data = fetch_data(task)
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})
    return process(data)

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    HALLUCINATED_STATE = rollback_to_checkpoint(),
    default            = triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(my_agent, policy=policy)
result = await agent.run("your task")
```

## Key features

- **10 failure types** — from `WRONG_TOOL_CALLED` to `GOAL_DRIFT`, each with a distinct recovery path
- **Zero-dependency classifier** — `RulesClassifier` catches most failures with pattern matching and zero API calls
- **Semantic classifier** — `LLMClassifier` handles ambiguous failures using any LLM (Anthropic, Ollama, Groq, HuggingFace)
- **Hybrid mode** — rules first, LLM only when rules return `UNKNOWN`
- **Framework adapters** — drop-in wrappers for LangGraph, CrewAI, OpenAI Agents SDK, LangChain
- **Durable checkpoints** — in-memory, SQLite, or Redis; real agent state persisted and restored on rollback
- **Attempt history** — strategies see every prior `(failure_type, action)` pair and can escalate intelligently

## Install

```bash
# Core
pip install triage-agent

# With extras
pip install "triage-agent[anthropic]"       # LLMClassifier via Claude
pip install "triage-agent[sqlite]"          # SQLiteCheckpointStore
pip install "triage-agent[redis]"           # RedisCheckpointStore
pip install "triage-agent[langgraph]"       # LangGraph adapter
pip install "triage-agent[crewai]"          # CrewAI adapter
pip install "triage-agent[openai-agents]"   # OpenAI Agents SDK adapter
pip install "triage-agent[langchain]"       # LangChain adapter
```

## Why not just if/else?

See [How It Works](getting-started/how-it-works.md) for a full explanation. The short version: agent failures don't give you clean discriminators. The same `RuntimeError` can mean ten different things depending on what the agent was doing before it failed — and the signal lives in the trajectory, not the exception string.
