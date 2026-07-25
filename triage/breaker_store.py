"""
triage.breaker_store
~~~~~~~~~~~~~~~~~~~~
Persistence protocol for CircuitBreaker state.

The in-process CircuitBreaker stores all state in memory — safe for a single
worker, but invisible to other processes. Swap in a ``BreakerStore`` to share
OPEN/HALF_OPEN state across workers and survive process restarts.

Usage::

    import redis.asyncio as aioredis
    from triage.breaker import CircuitBreaker
    from triage.breaker_store import RedisBreakerStore

    client = aioredis.Redis.from_url("redis://localhost:6379")
    store = RedisBreakerStore(client, key_prefix="myapp:breaker")
    breaker = CircuitBreaker(failure_threshold=5, window_seconds=60,
                             cooldown_seconds=30, store=store)

Clock note
----------
``time.monotonic()`` is process-local and cannot be compared across machines.
When a ``store`` is attached, ``CircuitBreaker`` switches to ``time.time()``
(wall clock, UTC seconds) for all timestamps. The ``_now`` injectable in
``record_failure`` / ``record_success`` / ``state`` / etc. is therefore a
wall-clock float when a store is present. Tests that pass explicit ``_now``
values will continue to work unchanged — just pass wall-clock floats instead
of monotonic floats.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from triage.breaker import BreakerState


@runtime_checkable
class BreakerStore(Protocol):
    """Persistence backend for a single CircuitBreaker instance.

    All methods are synchronous so they can be called from within a
    ``threading.Lock``-guarded section without requiring an event loop.
    Implementations must be thread-safe.
    """

    def load(self) -> BreakerSnapshot:
        """Load the current persisted state. Returns defaults if absent."""
        ...

    def save(self, snapshot: BreakerSnapshot) -> None:
        """Persist the full snapshot atomically."""
        ...

    def add_failure(self, ts: float) -> list[float]:
        """Append a failure timestamp and return the updated list."""
        ...

    def evict_before(self, cutoff: float) -> list[float]:
        """Remove timestamps older than cutoff and return the remaining list."""
        ...

    def set_probe_in_flight(self, value: bool) -> None:
        """Set the probe-in-flight flag."""
        ...

    def get_probe_in_flight(self) -> bool:
        """Return the probe-in-flight flag."""
        ...


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class BreakerSnapshot:
    """Full serialisable state of a CircuitBreaker at a point in time."""

    state: BreakerState = BreakerState.CLOSED
    failure_times: list[float] = field(default_factory=list)
    opened_at: float | None = None
    probe_in_flight: bool = False


# ── Redis implementation ──────────────────────────────────────────────────────

class RedisBreakerStore:
    """Thread-safe BreakerStore backed by Redis.

    Uses synchronous ``redis`` (not asyncio) so it can be called under the
    ``threading.Lock`` in CircuitBreaker without an event loop. The
    ``redis[asyncio]`` extra installs both the sync and async clients.

    Parameters
    ----------
    redis_client:
        A synchronous ``redis.Redis`` instance (not ``redis.asyncio.Redis``).
    key_prefix:
        Prefix for all Redis keys. Use a unique prefix per CircuitBreaker
        when multiple breakers share a Redis instance.
    ttl_seconds:
        Optional TTL applied to the state key after every write. Useful for
        auto-expiring breaker state in serverless environments. Default: no TTL.
    """

    def __init__(
        self,
        redis_client: object,
        *,
        key_prefix: str = "triage:breaker",
        ttl_seconds: int | None = None,
    ) -> None:
        try:
            import redis as _redis_mod
        except ImportError as exc:
            raise ImportError(
                "RedisBreakerStore requires 'redis'. "
                "Install it with: pip install triage-agent[redis]"
            ) from exc
        if not isinstance(redis_client, _redis_mod.Redis):
            raise TypeError(
                "RedisBreakerStore expects a synchronous redis.Redis client, "
                f"got {type(redis_client).__name__}. "
                "Use redis.Redis(...), not redis.asyncio.Redis(...)."
            )
        self._r: Any = redis_client
        self._state_key = f"{key_prefix}:state"
        self._opened_at_key = f"{key_prefix}:opened_at"
        self._failures_key = f"{key_prefix}:failures"
        self._probe_key = f"{key_prefix}:probe_in_flight"
        self._ttl = ttl_seconds

    # ── BreakerStore protocol ─────────────────────────────────────────────────

    def load(self) -> BreakerSnapshot:
        pipe = self._r.pipeline()
        pipe.get(self._state_key)
        pipe.get(self._opened_at_key)
        pipe.zrangebyscore(self._failures_key, "-inf", "+inf", withscores=True)
        pipe.get(self._probe_key)
        state_raw, opened_at_raw, failures_raw, probe_raw = pipe.execute()

        state = BreakerState(state_raw.decode()) if state_raw else BreakerState.CLOSED
        opened_at = float(opened_at_raw) if opened_at_raw else None
        failure_times = [score for (_member, score) in failures_raw]
        probe_in_flight = probe_raw == b"1"

        return BreakerSnapshot(
            state=state,
            failure_times=failure_times,
            opened_at=opened_at,
            probe_in_flight=probe_in_flight,
        )

    def save(self, snapshot: BreakerSnapshot) -> None:
        pipe = self._r.pipeline()
        pipe.set(self._state_key, snapshot.state.value)
        if snapshot.opened_at is not None:
            pipe.set(self._opened_at_key, str(snapshot.opened_at))
        else:
            pipe.delete(self._opened_at_key)
        pipe.set(self._probe_key, "1" if snapshot.probe_in_flight else "0")
        if self._ttl is not None:
            pipe.expire(self._state_key, self._ttl)
            pipe.expire(self._opened_at_key, self._ttl)
            pipe.expire(self._failures_key, self._ttl)
            pipe.expire(self._probe_key, self._ttl)
        pipe.execute()

    def add_failure(self, ts: float) -> list[float]:
        import uuid as _uuid
        member = f"{ts}:{_uuid.uuid4().hex}"
        self._r.zadd(self._failures_key, {member: ts})
        pairs = self._r.zrangebyscore(self._failures_key, "-inf", "+inf", withscores=True)
        return [score for (_member, score) in pairs]

    def evict_before(self, cutoff: float) -> list[float]:
        self._r.zremrangebyscore(self._failures_key, "-inf", f"({cutoff}")
        pairs = self._r.zrangebyscore(self._failures_key, "-inf", "+inf", withscores=True)
        return [score for (_member, score) in pairs]

    def set_probe_in_flight(self, value: bool) -> None:
        self._r.set(self._probe_key, "1" if value else "0")

    def get_probe_in_flight(self) -> bool:
        raw = self._r.get(self._probe_key)
        return bool(raw == b"1")
