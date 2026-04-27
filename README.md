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
pip install triage-agent
```

With OpenAI support:
```bash
pip install "triage-agent[openai]"
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

# 1. Define your agent — it receives a record_step callback
async def my_agent(task: str, *, record_step, _triage_hint=None, **kwargs):
    # ... your agent logic ...
    record_step(Step(index=0, action="called search", tool_called="search",
                     tool_input={"q": task}, tool_output="some result"))
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

## How it works

### 1. Record steps

Your agent calls `record_step(Step(...))` for each observable action. `triage` injects the callback — you don't need to import or construct anything:

```python
async def my_agent(task: str, *, record_step, **kwargs):
    # Tool call
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

When your agent raises an exception, `triage` runs the `RulesClassifier` over the recorded trajectory and returns one of 10 `FailureType` values:

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

Save agent state at key points so triage can roll back to them on failure:

```python
from triage.checkpoint import InMemoryCheckpointStore, make_checkpoint

store = InMemoryCheckpointStore()
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)

# In your agent: save a checkpoint manually
async def my_agent(task: str, *, record_step, **kwargs):
    # ... first phase ...
    checkpoint = make_checkpoint(
        state={"phase": "data-fetched", "data": fetched_data},
        trajectory_steps=current_trajectory,
        id="after-fetch",
    )
    await store.save(checkpoint)
    # ... second phase, which might fail ...
```

Enable automatic checkpointing after every successful step:

```python
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

---

## Recovery context in your agent

When triage retries your agent after a failure, it injects context into `**kwargs`:

```python
async def my_agent(task: str, *, record_step, _triage_hint=None, _triage_subgoal=None, **kwargs):
    if _triage_hint:
        # Modify behaviour based on what went wrong last time
        print(f"Recovery hint: {_triage_hint}")
    if _triage_subgoal:
        # Skip to the incomplete subgoal instead of restarting from scratch
        task = _triage_subgoal
```

| Key | Set when |
|---|---|
| `_triage_hint` | `RETRY`, `REPLAN`, or `ROLLBACK` action |
| `_triage_subgoal` | `RESUME` action |

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

The default `RulesClassifier` is pattern-based and makes zero API calls. You can swap in your own:

```python
from triage.classifier.base import Classifier
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

class MyLLMClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        # Call your LLM, inspect the trajectory, return a FailureType
        ...

agent = triage.Agent(my_agent, policy=policy, classifier=MyLLMClassifier())
```

Any class implementing `classify(trajectory, task) -> FailureType` satisfies the protocol.

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
  checkpoint.py      Checkpoint, CheckpointStore protocol, InMemoryCheckpointStore
  policy.py          RecoveryAction (6 constructors), FailurePolicy
  agent.py           Agent class, TriageEscalationError, TriageAbortError, @agent decorator
  classifier/
    base.py          Classifier protocol
    rules.py         RulesClassifier — 6 rules, sync, zero API calls
  strategies/
    retry.py         retry_with_tool_manifest(), backoff_and_retry()
    replan.py        replan(), resume_from_subgoal()
    rollback.py      rollback_to_checkpoint()
```

---

## Roadmap (v0.2)

- **Adapters** — drop-in wrappers for LangGraph, CrewAI, OpenAI Agents SDK
- **LLM classifier** — semantic classification behind the same `Classifier` protocol
- **Async-safe checkpoints** — clean anyio task-group implementation
- **Storage backends** — Redis and SQLite `CheckpointStore` implementations

---

## License

MIT
