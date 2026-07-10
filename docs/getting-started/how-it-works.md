# How It Works

## The problem with if/else

With a normal program, exceptions carry enough information to branch on:

```python
except ValueError:
    handle_bad_input()
except HTTPError as e:
    if e.status == 429:
        retry()
```

With an LLM agent, the same `RuntimeError("something went wrong")` can mean nine different things depending on what the agent was doing, what it called, and what it said before failing. The failure lives in the **trajectory**, not the exception:

- A loop isn't an exception — it's three identical steps
- A hallucination isn't an exception — it's the LLM asserting something the tools didn't return
- A schema mismatch might surface as a `ValueError`, a `JSONDecodeError`, or a model-specific error string

`if/else` on the exception string misses most of this. It also collapses distinct failures into one code path and has to be reimplemented in every agent.

## The triage loop

```
┌─────────────────────────────────────────────────────┐
│                    Agent.run(task)                   │
│                                                      │
│  while True:                                         │
│    try:                                              │
│      result = await fn(task, record_step=...,  ─────┼──▶ your agent fn
│                         update_state=...)            │    calls record_step()
│      await drain_checkpoints()                       │    and update_state()
│      return result          ◀── success              │
│                                                      │
│    except Exception:                                 │
│      failure_type = classifier.classify(trajectory)  │
│      ctx = FailureContext(failure_type, trajectory,  │
│                           attempt_history, ...)      │
│      action = await policy.dispatch(ctx)             │
│      kwargs = execute_action(action)  ───────────────┼──▶ RETRY / REPLAN /
│      attempt += 1                                    │    ROLLBACK / RESUME /
└─────────────────────────────────────────────────────┘    ESCALATE / ABORT
```

### Step 1 — Record steps

Your agent calls `record_step(Step(...))` for each observable action. triage injects the callback — nothing to import. Optionally call `update_state(dict)` to persist data into checkpoints.

### Step 2 — Classify the failure

When your agent raises, triage runs the classifier over the recorded trajectory and returns one `FailureType` from 9 possible values. If the classifier defines an `aclassify()` method (e.g. `LLMClassifier`, `HybridClassifier`), triage awaits it directly; otherwise it runs the synchronous `classify()` in a thread — either way, the event loop is never blocked.

The default `RulesClassifier` is pattern-based and makes zero API calls. For ambiguous failures, `LLMClassifier` or `HybridClassifier` can be used.

### Step 3 — Dispatch to a strategy

The `FailurePolicy` maps each `FailureType` to a strategy callable. The strategy receives a `FailureContext` (trajectory, failure type, attempt history, checkpoint ID) and returns a `RecoveryAction`.

### Step 4 — Execute the recovery

triage executes the action and re-runs your agent with injected context:

| Action | What happens |
|---|---|
| `RETRY` | Re-runs the agent; injects `_triage_hint` |
| `REPLAN` | Re-runs the agent; injects `_triage_hint` with new plan instruction |
| `ROLLBACK` | Restores trajectory + state from checkpoint; injects `_triage_hint` and `_triage_state` |
| `RESUME` | Re-runs agent; injects `_triage_subgoal` |
| `ESCALATE` | Raises `TriageEscalationError` |
| `ABORT` | Raises `TriageAbortError` |

## Separation of concerns

- **Strategies declare intent** — a `RecoveryAction.ROLLBACK` tells the agent to roll back, it doesn't do it
- **`agent.py` executes it** — all state mutation, checkpoint loading, and kwargs injection happens in one place
- **Classifiers are synchronous** — `classify()` is a plain `def`, not `async def`. It runs in a thread when called from the async loop, unless the classifier also defines an optional `aclassify()`, which triage awaits directly
- **No framework imports in core** — `triage/` only imports `anyio`, `pydantic`, and stdlib. Adapters live in `triage/adapters/`
