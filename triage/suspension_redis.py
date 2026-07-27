"""
triage.suspension_redis
~~~~~~~~~~~~~~~~~~~~~~~
Redis-backed SuspensionStore for durable human-in-the-loop pause/resume.

Install: pip install triage-agent[redis]
"""

from __future__ import annotations

try:
    from redis.asyncio import Redis
except ImportError as exc:
    raise ImportError(
        "RedisSuspensionStore requires 'redis[asyncio]'. "
        "Install it with: pip install triage-agent[redis]"
    ) from exc

from triage.suspension import SuspendedRun, deserialize_run, serialize_run


class RedisSuspensionStore:
    """Durable SuspensionStore backed by Redis.

    Pass a pre-configured ``redis.asyncio.Redis`` client::

        import redis.asyncio as aioredis
        client = aioredis.Redis.from_url("redis://localhost:6379")
        store = RedisSuspensionStore(client)

    Parameters
    ----------
    redis:
        Pre-configured ``redis.asyncio.Redis`` client.  The caller is
        responsible for closing it.
    key_prefix:
        Prefix for all Redis keys.  Use a unique prefix per store instance
        when multiple stores share a Redis instance.
    ttl_seconds:
        Optional TTL applied to each key after ``save()``.  Useful for
        auto-expiring runs that are never resumed.  Default: no TTL.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = "triage:suspension",
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis = redis
        self._key_prefix = key_prefix
        self._ttl = ttl_seconds

    def _key(self, token: str) -> str:
        return f"{self._key_prefix}:{token}"

    async def save(self, run: SuspendedRun) -> None:
        data = serialize_run(run)
        key = self._key(run.token)
        if self._ttl is not None:
            await self._redis.set(key, data, ex=self._ttl)
        else:
            await self._redis.set(key, data)

    async def load(self, token: str) -> SuspendedRun:
        key = self._key(token)
        raw = await self._redis.get(key)
        if raw is None:
            raise KeyError(f"No suspended run with token {token!r}")
        text = raw.decode() if isinstance(raw, bytes) else raw
        return deserialize_run(text)

    async def delete(self, token: str) -> None:
        await self._redis.delete(self._key(token))
