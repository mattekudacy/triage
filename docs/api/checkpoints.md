# Checkpoints

## CheckpointStore protocol

::: triage.checkpoint.base.CheckpointStore

## Checkpoint

::: triage.checkpoint.base.Checkpoint

## make_checkpoint

::: triage.checkpoint.base.make_checkpoint

## InMemoryCheckpointStore

::: triage.checkpoint.memory.InMemoryCheckpointStore

## SQLiteCheckpointStore

Requires `pip install triage-agent[sqlite]`.

::: triage.checkpoint.sqlite.SQLiteCheckpointStore

## RedisCheckpointStore

Requires `pip install triage-agent[redis]`.

::: triage.checkpoint.redis.RedisCheckpointStore

---

## Conceptual notes

### Auto-checkpointing

Pass `auto_checkpoint=True` to save a checkpoint after every `record_step()` call.
The ROLLBACK action then restores from the most recent checkpoint in the current run:

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    checkpoint_store=SQLiteCheckpointStore("prod.db"),
    auto_checkpoint=True,
)
```

### Manual checkpointing

Call `update_state` to accumulate state and let auto-checkpoint persist it, or save
explicitly at key points:

```python
async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
    data = await fetch(task)
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data, "step": 0})  # saved into the next checkpoint
    return process(data)
```

### Custom store

Implement the `CheckpointStore` protocol to use any backend:

```python
class MyStore:
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def load(self, checkpoint_id: str) -> Checkpoint: ...
    async def latest(self, run_id: str | None = None) -> Checkpoint | None: ...
```
