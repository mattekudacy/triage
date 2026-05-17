# Quick Start

## 1. Define your agent

Your agent receives two injected callbacks — `record_step` to log what it did, and `update_state` to persist data into checkpoints:

```python
from triage.taxonomy import Step

async def my_agent(task: str, *, record_step, update_state, _triage_hint=None, **kwargs):
    # Use _triage_hint if triage is retrying after a failure
    if _triage_hint:
        print(f"Recovery hint: {_triage_hint}")

    # Do your work
    result = call_some_tool(task)

    # Record what happened
    record_step(Step(
        index=0,
        action="called tool",
        tool_called="my_tool",
        tool_input={"task": task},
        tool_output=result,
        idempotent=True,   # mark True only for safe-to-retry (read-only) steps
    ))

    # Persist state so rollback can restore it
    update_state({"result": result})

    return result
```

## 2. Declare a recovery policy

```python
import triage
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan, resume_from_subgoal
from triage.strategies.rollback import rollback_to_checkpoint

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    SCHEMA_MISMATCH    = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    CONSTRAINT_IGNORED = replan(hint="Re-read the constraints carefully."),
    PLAN_INCOMPLETE    = resume_from_subgoal(),
    default            = triage.FailurePolicy.escalate_by_default(),
)
```

Any `FailureType` not listed falls through to `default`. If `default` is unset, triage escalates automatically.

## 3. Wrap and run

```python
agent = triage.Agent(my_agent, policy=policy)

try:
    result = await agent.run("your task here")
except triage.TriageEscalationError as exc:
    print(f"Needs human review: {exc.context.failure_type.value}")
except triage.TriageAbortError as exc:
    print(f"Hard stop: {exc}")
```

## Decorator form

```python
@triage.agent(policy=policy)
async def my_agent(task: str, *, record_step, update_state, **kwargs):
    ...
```

## With checkpoints

```python
from triage.checkpoint import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
agent = triage.Agent(
    my_agent,
    policy=policy,
    checkpoint_store=store,
    auto_checkpoint=True,   # saves a checkpoint after every record_step call
    max_recovery_attempts=3,
)
```

## With a semantic classifier

```python
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier

# Rules first (free), LLM only when rules return UNKNOWN
classifier = HybridClassifier(llm=LLMClassifier())

agent = triage.Agent(my_agent, policy=policy, classifier=classifier)
```

## Safety options

```python
# Escalate instead of retrying when any non-idempotent step is in the trajectory
agent = triage.Agent(my_agent, policy=policy, strict_idempotency=True)

# Cap total wall-clock time spent on recovery
agent = triage.Agent(my_agent, policy=policy, max_recovery_seconds=30.0)
```
