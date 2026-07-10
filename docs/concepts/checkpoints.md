# Checkpoints

Checkpoints let triage restore your agent to a known-good state before retrying after a `ROLLBACK` action. Each checkpoint is a snapshot of the agent's trajectory and any state explicitly saved via `update_state()`.

## What a checkpoint contains

```python
@dataclass
class Checkpoint:
    id: str                          # UUID
    timestamp: float                 # Unix timestamp
    state: dict[str, Any]            # data saved by update_state()
    trajectory_snapshot: list[Step]  # steps recorded up to this point
```

## Enabling checkpoints

### Manual checkpoints

Call `update_state()` to persist data, and configure a `checkpoint_store`. The store saves a checkpoint whenever `record_step` is called with `auto_checkpoint=True`, or you can rely on triage to save them automatically.

### Auto-checkpoint

```python
from triage.checkpoint import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
agent = triage.Agent(
    my_agent,
    policy=policy,
    checkpoint_store=store,
    auto_checkpoint=True,    # saves a checkpoint after every record_step() call
)
```

With `auto_checkpoint=True`, triage queues a checkpoint save after each `record_step()` call and drains the queue before executing any recovery action. This guarantees a checkpoint is always available when `ROLLBACK` runs.

## Saving agent state

Call `update_state()` from your agent to persist data into checkpoints:

```python
async def my_agent(task: str, *, record_step, update_state, **kwargs):
    data = fetch_data(task)
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data, "step": 0})   # saved into the next checkpoint
    return process(data)
```

On `ROLLBACK`, triage restores `_current_state` and injects `_triage_state` into the next call's kwargs:

```python
async def my_agent(task: str, *, record_step, update_state, **kwargs):
    state = kwargs.get("_triage_state", {})
    if state:
        # skip re-fetching — triage restored it
        data = state["data"]
    else:
        data = fetch_data(task)
```

---

## Storage backends

### InMemoryCheckpointStore

Default. Holds checkpoints in a Python list. Lost when the process exits.

```python
from triage.checkpoint import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
```

No extra dependencies. As of v0.11, its internal dict is guarded by an `anyio.Lock`, so it's safe to share across multiple concurrent `Agent.run()` calls on the same store instance.

### SQLiteCheckpointStore

Persists checkpoints to a SQLite database file. Survives process restarts.

```python
pip install "triage-agent[sqlite]"
```

```python
from triage.checkpoint.sqlite import SQLiteCheckpointStore

store = SQLiteCheckpointStore("checkpoints.db")
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

| Detail | Value |
|---|---|
| Dependency | `aiosqlite>=0.19` |
| Table | `checkpoints (id TEXT PK, timestamp REAL, state TEXT, trajectory TEXT)` |
| `latest()` | `SELECT ... ORDER BY timestamp DESC LIMIT 1` |
| Connection | New connection per operation — simple and safe |

Do not use `":memory:"` as the path — each `aiosqlite.connect(":memory:")` call opens a fresh, empty database. Use a real file path.

### RedisCheckpointStore

Persists checkpoints to Redis. Good for distributed agents or when you need TTL/expiration.

```python
pip install "triage-agent[redis]"
```

```python
import redis.asyncio as aioredis
from triage.checkpoint.redis import RedisCheckpointStore

redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
store = RedisCheckpointStore(redis=redis_client)
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

| Detail | Value |
|---|---|
| Dependency | `redis[asyncio]>=5.0` |
| Key scheme | `triage:checkpoint:<id>` for data; `triage:checkpoint:index` sorted set |
| `save()` | Pipeline transaction: `SET key` + `ZADD index` (atomic) |
| `latest()` | `ZREVRANGE index 0 0` → `load(id)` |

`RedisCheckpointStore` uses dependency injection — pass a pre-configured `Redis` instance. This makes it testable with `fakeredis`.

---

## Implementing a custom store

Any object satisfying the `CheckpointStore` protocol works:

```python
from triage.checkpoint import CheckpointStore, Checkpoint

class MyStore:
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def load(self, id: str) -> Checkpoint: ...
    async def latest(self) -> Checkpoint | None: ...
```

All three methods must be `async def`. `latest()` returns `None` if no checkpoints have been saved.

---

## Serialization

SQLite and Redis backends serialize checkpoints to JSON. The serializer uses a `_safe_json()` fallback that converts non-serializable values to their `repr()` string. This means complex objects in `state` or `tool_output` are stored lossily. For full fidelity, store only JSON-native types (strings, numbers, lists, dicts) in `update_state()`.

---

## The _pending_checkpoints drain pattern

Under `auto_checkpoint=True`, `record_step()` queues a coroutine rather than immediately awaiting it. triage drains the queue at three points in `Agent.run()`:

1. After a successful agent return — before returning the result
2. Before classifying a failure — so the checkpoint is available for `ROLLBACK`
3. Before re-raising `TriageEscalationError` or `TriageAbortError`

This keeps `record_step()` synchronous (no `async def`) while guaranteeing checkpoints are always committed before any policy action runs.
