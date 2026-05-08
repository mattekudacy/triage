# Checkpoints

## CheckpointStore protocol

```python
class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def load(self, id: str) -> Checkpoint: ...
    async def latest(self) -> Checkpoint | None: ...
```

All methods are `async def`. `latest()` returns `None` if no checkpoints have been saved.

## Checkpoint dataclass

```python
@dataclass
class Checkpoint:
    id: str                          # UUID string
    timestamp: float                 # Unix timestamp
    state: dict[str, Any]            # data saved by update_state()
    trajectory_snapshot: list[Step]  # steps at the time of save
```

## make_checkpoint()

```python
from triage.checkpoint import make_checkpoint

checkpoint = make_checkpoint(
    state={"data": ...},
    trajectory_steps=trajectory.steps,
    id=None,   # auto-generates UUID if omitted
)
```

## InMemoryCheckpointStore

```python
from triage.checkpoint import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
```

Default. Holds checkpoints in memory. Not persisted across process restarts. Not concurrency-safe for concurrent `run()` calls on the same instance.

## SQLiteCheckpointStore

```python
from triage.checkpoint.sqlite import SQLiteCheckpointStore

store = SQLiteCheckpointStore("checkpoints.db")
```

Requires `pip install "triage-agent[sqlite]"` (`aiosqlite>=0.19`).

Opens a new `aiosqlite` connection per operation. Use a real file path — `":memory:"` creates a fresh database on each connection.

## RedisCheckpointStore

```python
import redis.asyncio as aioredis
from triage.checkpoint.redis import RedisCheckpointStore

redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
store = RedisCheckpointStore(redis=redis_client)
```

Requires `pip install "triage-agent[redis]"` (`redis[asyncio]>=5.0`).

Accepts a pre-configured `Redis` instance (dependency injection). Key scheme: `triage:checkpoint:<id>` for data, `triage:checkpoint:index` sorted set for ordering.

## Serialization

SQLite and Redis backends serialize checkpoints to JSON with a `_safe_json()` fallback — non-serializable values are stored as `repr()` strings. Store only JSON-native types (`str`, `int`, `float`, `list`, `dict`) in `update_state()` for full fidelity on restore.
