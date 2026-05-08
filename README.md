# triage

**Classify why your agent failed. Recover intelligently.**

```
pip install triage-agent
```

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The problem

Current agent frameworks know *that* your agent failed. They don't know *why* — and without knowing why, every failure gets the same blunt response: retry from scratch or give up.

`triage` adds a classification-and-routing layer between the failure and the recovery:

```
agent fails → classify failure type → route to matching strategy → recover
```

It works with any async agent callable — OpenAI, LangGraph, CrewAI, raw LLM loops — without requiring you to change your framework.

---

## Installation

```bash
# Core only
pip install triage-agent

# With framework adapters
pip install "triage-agent[langgraph]"
pip install "triage-agent[crewai]"
pip install "triage-agent[openai-agents]"
pip install "triage-agent[langchain]"

# With LLM-based classifier
pip install "triage-agent[anthropic]"

# With durable checkpoint storage
pip install "triage-agent[sqlite]"
pip install "triage-agent[redis]"
```

Python 3.10+ required. Core dependencies: `anyio>=4.0`, `pydantic>=2.0`.

---

## Quick start

```python
import triage
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan
from triage.strategies.rollback import rollback_to_checkpoint
from triage.taxonomy import Step

# 1. Define your agent — it receives record_step and update_state callbacks
async def my_agent(task: str, *, record_step, update_state, _triage_hint=None, **kwargs):
    # ... your agent logic ...
    data = fetch_data(task)
    record_step(Step(index=0, action="called search", tool_called="search",
                     tool_input={"q": task}, tool_output=data))
    update_state({"data": data})   # persisted into checkpoints; restored on rollback
    return "done"

# 2. Declare a recovery policy
policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    HALLUCINATED_STATE = rollback_to_checkpoint(),
    default            = triage.FailurePolicy.escalate_by_default(),
)

# 3. Wrap and run
agent = triage.Agent(my_agent, policy=policy)
result = await agent.run("search for recent AI papers")
```

Or use the decorator form:

```python
@triage.agent(policy=policy)
async def my_agent(task: str, *, record_step, **kwargs):
    ...
```

---

## Framework adapters

Drop-in wrappers let you add triage to an existing agent without changing its internals.

### LangGraph

```python
from triage.adapters.langgraph import wrap_langgraph

agent = wrap_langgraph(compiled_graph, policy=policy)
result = await agent.run("your task")
```

Streams events via `graph.astream_events(..., version="v2")` to capture tool calls and LLM turns.

### CrewAI

```python
from triage.adapters.crewai import wrap_crewai

agent = wrap_crewai(crew, policy=policy)
result = await agent.run("your task")
```

Patches `crew.step_callback` for each run (original restored in `finally`).

### OpenAI Agents SDK

```python
from triage.adapters.openai_agents import wrap_openai_agents

agent = wrap_openai_agents(sdk_agent, policy=policy)
result = await agent.run("your task")
```

Uses `Runner.run_streamed` and iterates `stream_events()`.

### LangChain

```python
from triage.adapters.langchain import wrap_langchain

agent = wrap_langchain(executor, policy=policy)
result = await agent.run("your task")
```

Injects a fresh `BaseCallbackHandler` per call via `config={"callbacks": [...]}`.

All adapters accept the same optional kwargs as `triage.Agent`: `classifier`, `checkpoint_store`, `max_recovery_attempts`, `auto_checkpoint`.

---

## How it works

### 1. Record steps

Your agent calls `record_step(Step(...))` for each observable action. `triage` injects the callback — you don't need to import or construct anything:

```python
async def my_agent(task: str, *, record_step, **kwargs):
    result = call_tool("search", {"q": task})
    record_step(Step(
        index=0,
        action="called search tool",
        tool_called="search",
        tool_input={"q": task},
        tool_output=result,
    ))
```

### 2. Classify the failure

When your agent raises an exception, `triage` runs the classifier over the recorded trajectory and returns one of 10 `FailureType` values:

| FailureType | Trigger | Default recovery |
|---|---|---|
| `WRONG_TOOL_CALLED` | Error matches `"tool not found"` / `"no tool named"` | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | LLM output contains a forbidden string | Replan with constraint reminder |
| `LOOP_DETECTED` | Last 3 steps identical tool + input | Replan or rollback |
| `HALLUCINATED_STATE` | Agent asserts facts contradicting tool output | Rollback to checkpoint |
| `PLAN_INCOMPLETE` | Success declared but sub-goals incomplete | Resume from subgoal |
| `SCHEMA_MISMATCH` | Error matches `"validation error"` / JSON parse failure | Retry with schema hint |
| `CONTEXT_OVERFLOW` | Agent lost earlier context | Replan with compressed context |
| `GOAL_DRIFT` | Agent making progress toward the wrong goal | Replan with goal restatement |
| `EXTERNAL_FAULT` | HTTP 429 / 500 / 502 / 503 in error | Exponential backoff + retry |
| `UNKNOWN` | None of the above | Escalate to human |

The default `RulesClassifier` is pattern-based and makes zero API calls. For semantic classification, use `LLMClassifier`:

```python
from triage.classifier.llm import LLMClassifier

agent = triage.Agent(
    my_agent,
    policy=policy,
    classifier=LLMClassifier(model="claude-haiku-4-5-20251001"),
)
```

`LLMClassifier` uses the Anthropic sync client and falls back to `UNKNOWN` silently on any error. Requires `pip install "triage-agent[anthropic]"`.

### 3. Dispatch to a strategy

The policy maps each `FailureType` to a strategy callable. The strategy returns a `RecoveryAction` that tells `triage` what to do next.

### 4. Execute the recovery

`triage` executes the action and re-runs your agent with injected context:

| Action | What happens |
|---|---|
| `RETRY` | Re-runs the agent; injects `_triage_hint` into kwargs |
| `REPLAN` | Re-runs the agent; injects `_triage_hint` with new plan instruction |
| `ROLLBACK` | Restores trajectory from checkpoint, re-runs agent |
| `RESUME` | Re-runs agent; injects `_triage_subgoal` pointing at incomplete subgoal |
| `ESCALATE` | Raises `TriageEscalationError(message, context)` |
| `ABORT` | Raises `TriageAbortError(reason, context)` |

---

## Failure policy

`FailurePolicy` is a plain dataclass — one field per `FailureType`:

```python
policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    CONSTRAINT_IGNORED = replan(hint="Re-read the task constraints carefully."),
    LOOP_DETECTED      = replan(max_replans=2),
    HALLUCINATED_STATE = rollback_to_checkpoint(),
    PLAN_INCOMPLETE    = resume_from_subgoal(),
    SCHEMA_MISMATCH    = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    default            = triage.FailurePolicy.escalate_by_default(),
)
```

Any `FailureType` not explicitly listed falls through to `default`. If `default` is also unset, triage escalates automatically.

---

## Built-in strategies

### `triage.strategies.retry`

```python
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry

# Retry with a hint to use the correct tool manifest
retry_with_tool_manifest(max_attempts=3)

# Retry with exponential backoff (2^attempt seconds). Good for rate limits.
backoff_and_retry(max_attempts=5)
```

### `triage.strategies.replan`

```python
from triage.strategies.replan import replan, resume_from_subgoal

# Restart with a new plan, optionally injecting a hint
replan(hint="The previous approach used the wrong API endpoint.")

# Continue from the first incomplete sub-goal
resume_from_subgoal()
```

### `triage.strategies.rollback`

```python
from triage.strategies.rollback import rollback_to_checkpoint

# Restore to latest checkpoint (or a named one)
rollback_to_checkpoint()
rollback_to_checkpoint(checkpoint_id="before-api-call")
```

---

## Checkpoints

Save agent state at key points so triage can roll back to them on failure.

### In-memory (default)

```python
from triage.checkpoint import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### SQLite (persistent, single-process)

```bash
pip install "triage-agent[sqlite]"
```

```python
from triage.checkpoint.sqlite import SQLiteCheckpointStore

store = SQLiteCheckpointStore("runs/checkpoints.db")
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### Redis (distributed)

```bash
pip install "triage-agent[redis]"
```

```python
import redis.asyncio as aioredis
from triage.checkpoint.redis import RedisCheckpointStore

client = aioredis.Redis.from_url("redis://localhost:6379")
store = RedisCheckpointStore(client)
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### Auto-checkpoint

Enable automatic checkpointing after every successful step:

```python
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

Checkpoints are always awaited before `run()` returns or any recovery action executes, so a `ROLLBACK` always has a checkpoint available.

---

## Recovery context in your agent

Two callbacks are always injected, plus recovery context on retry:

```python
async def my_agent(
    task: str,
    *,
    record_step,
    update_state,
    _triage_hint=None,
    _triage_subgoal=None,
    _triage_state=None,
    **kwargs,
):
    # On rollback, _triage_state contains the state saved at the checkpoint
    if _triage_state:
        data = _triage_state["data"]   # skip re-fetching, use restored state
    else:
        data = fetch_data(task)

    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})       # saved into every auto_checkpoint

    if _triage_hint:
        print(f"Recovery hint: {_triage_hint}")
    if _triage_subgoal:
        task = _triage_subgoal
```

| Key | Set when |
|---|---|
| `record_step` | Always — injected on every call |
| `update_state` | Always — injected on every call |
| `_triage_hint` | `RETRY`, `REPLAN`, or `ROLLBACK` action |
| `_triage_subgoal` | `RESUME` action |
| `_triage_state` | `ROLLBACK` action, when checkpoint has non-empty state |

---

## Handling escalation and abort

```python
try:
    result = await agent.run(task)
except triage.TriageEscalationError as exc:
    # exc.context is a FailureContext with the full trajectory and failure type
    print(f"Needs human review: {exc}")
    print(f"Failure type: {exc.context.failure_type.value}")
    print(f"Failed at step: {exc.context.critical_step_index}")
except triage.TriageAbortError as exc:
    print(f"Hard stop: {exc}")
```

---

## Custom classifier

Any class implementing `classify(trajectory, task) -> FailureType` satisfies the protocol:

```python
from triage.classifier.base import Classifier
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```

---

## Example: OpenAI tool-calling loop

See [`examples/raw_openai.py`](examples/raw_openai.py) for a full working example. It deliberately triggers a `WRONG_TOOL_CALLED` failure on the first attempt and shows triage catching and recovering it automatically:

```bash
OPENAI_API_KEY=sk-... python examples/raw_openai.py
```

Expected output:

```
Task: What is 42 * 17?

[triage] wrong_tool_called detected at step 0
[triage] Dispatching: RecoveryAction.RETRY(hint='Re-run using only tools in the current manifest.', inject={'max_attempts': 3})
[triage] Attempt 1...

Result: 714
```

---

## Project layout

```
triage/
  taxonomy.py        FailureType enum, Step, FailureContext
  trajectory.py      Trajectory (append / replay_from / last_n_steps)
  checkpoint/
    base.py          Checkpoint, CheckpointStore protocol, serialization helpers
    memory.py        InMemoryCheckpointStore
    sqlite.py        SQLiteCheckpointStore (requires aiosqlite)
    redis.py         RedisCheckpointStore (requires redis[asyncio])
  policy.py          RecoveryAction (6 constructors), FailurePolicy
  agent.py           Agent class, TriageEscalationError, TriageAbortError, @agent decorator
  classifier/
    base.py          Classifier protocol
    rules.py         RulesClassifier — 6 rules, sync, zero API calls
    llm.py           LLMClassifier — semantic via Anthropic (requires anthropic)
  strategies/
    retry.py         retry_with_tool_manifest(), backoff_and_retry()
    replan.py        replan(), resume_from_subgoal()
    rollback.py      rollback_to_checkpoint()
  adapters/
    langgraph.py     wrap_langgraph() (requires langgraph)
    crewai.py        wrap_crewai() (requires crewai)
    openai_agents.py wrap_openai_agents() (requires openai-agents)
    langchain.py     wrap_langchain() (requires langchain)
```

---

## License

MIT
