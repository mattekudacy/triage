"""
triage.checkpoint.redis
~~~~~~~~~~~~~~~~~~~~~~~
Distributed CheckpointStore backed by Redis via redis.asyncio.

Install: pip install triage-agent[redis]
"""

from __future__ import annotations

import json
from typing import Any

try:
    from redis.asyncio import Redis
except ImportError as exc:
    raise ImportError(
        "RedisCheckpointStore requires 'redis[asyncio]'. "
        "Install it with: pip install triage-agent[redis]"
    ) from exc

from triage.checkpoint.base import (
    Checkpoint,
    _dict_to_step,
    _safe_json,
    _step_to_dict,
)

_KEY_PREFIX = "triage:checkpoint:"
_INDEX_KEY = "triage:checkpoint:index"  # sorted set: member=id, score=timestamp


class RedisCheckpointStore:
    """Distributed CheckpointStore backed by Redis.

    Pass a pre-configured ``redis.asyncio.Redis`` client::

        import redis.asyncio as aioredis
        client = aioredis.Redis.from_url("redis://localhost:6379")
        store = RedisCheckpointStore(client)

    The client is the caller's responsibility to close.
    ``save`` is atomic: checkpoint data and the timestamp index are written
    in a single pipeline transaction.
    """

    def __init__(self, redis: "Redis") -> None:
        self._redis = redis

    async def save(self, checkpoint: Checkpoint) -> None:
        data = json.dumps({
            "id": checkpoint.id,
            "timestamp": checkpoint.timestamp,
            "state": _safe_json(checkpoint.state),
            "trajectory": [_step_to_dict(s) for s in checkpoint.trajectory_snapshot],
        })
        key = _KEY_PREFIX + checkpoint.id
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(key, data)
            pipe.zadd(_INDEX_KEY, {checkpoint.id: checkpoint.timestamp})
            await pipe.execute()

    async def load(self, id: str) -> Checkpoint:
        key = _KEY_PREFIX + id
        data = await self._redis.get(key)
        if data is None:
            raise KeyError(f"No checkpoint with id {id!r}")
        d = json.loads(data)
        return Checkpoint(
            id=d["id"],
            timestamp=d["timestamp"],
            state=d["state"],
            trajectory_snapshot=[_dict_to_step(s) for s in d["trajectory"]],
        )

    async def latest(self) -> Checkpoint | None:
        ids = await self._redis.zrevrange(_INDEX_KEY, 0, 0)
        if not ids:
            return None
        latest_id = ids[0].decode() if isinstance(ids[0], bytes) else ids[0]
        return await self.load(latest_id)
