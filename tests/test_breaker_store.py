"""
tests/test_breaker_store.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for BreakerStore protocol + RedisBreakerStore + CircuitBreaker(store=...).
Requires fakeredis; skipped automatically if not installed.
"""

from __future__ import annotations

import pytest

from triage.breaker import BreakerState, CircuitBreaker

# ── fakeredis availability ────────────────────────────────────────────────────

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")


def _make_store(prefix: str = "test:breaker") -> RedisBreakerStore:  # noqa: F821
    from triage.breaker_store import RedisBreakerStore

    r = fakeredis.FakeRedis()
    return RedisBreakerStore(r, key_prefix=prefix)


# ── BreakerSnapshot defaults ──────────────────────────────────────────────────


def test_snapshot_default_state_is_closed():
    from triage.breaker_store import BreakerSnapshot

    snap = BreakerSnapshot()
    assert snap.state == BreakerState.CLOSED
    assert snap.failure_times == []
    assert snap.opened_at is None
    assert snap.probe_in_flight is False


# ── RedisBreakerStore unit tests ──────────────────────────────────────────────


def test_load_returns_defaults_when_empty():
    store = _make_store()
    snap = store.load()
    assert snap.state == BreakerState.CLOSED
    assert snap.failure_times == []
    assert snap.opened_at is None
    assert snap.probe_in_flight is False


def test_save_and_load_roundtrip():
    from triage.breaker_store import BreakerSnapshot

    store = _make_store()
    snap = BreakerSnapshot(
        state=BreakerState.OPEN,
        failure_times=[1.0, 2.0],
        opened_at=2.0,
        probe_in_flight=False,
    )
    store.save(snap)
    loaded = store.load()
    assert loaded.state == BreakerState.OPEN
    assert loaded.opened_at == pytest.approx(2.0)
    assert loaded.probe_in_flight is False


def test_add_failure_returns_accumulated_times():
    store = _make_store()
    times = store.add_failure(10.0)
    assert len(times) == 1
    times = store.add_failure(20.0)
    assert len(times) == 2


def test_evict_before_removes_old_entries():
    store = _make_store()
    store.add_failure(1.0)
    store.add_failure(2.0)
    store.add_failure(100.0)
    remaining = store.evict_before(50.0)
    assert len(remaining) == 1
    assert remaining[0] == pytest.approx(100.0)


def test_probe_in_flight_roundtrip():
    store = _make_store()
    assert store.get_probe_in_flight() is False
    store.set_probe_in_flight(True)
    assert store.get_probe_in_flight() is True
    store.set_probe_in_flight(False)
    assert store.get_probe_in_flight() is False


def test_wrong_client_type_raises():
    import redis.asyncio as aioredis

    from triage.breaker_store import RedisBreakerStore

    async_client = aioredis.Redis()
    with pytest.raises(TypeError, match="synchronous redis.Redis"):
        RedisBreakerStore(async_client)


# ── CircuitBreaker(store=...) integration ─────────────────────────────────────


def _breaker(prefix: str = "test:cb", **kwargs) -> CircuitBreaker:
    store = _make_store(prefix)
    return CircuitBreaker(
        failure_threshold=kwargs.get("failure_threshold", 3),
        window_seconds=kwargs.get("window_seconds", 60.0),
        cooldown_seconds=kwargs.get("cooldown_seconds", 30.0),
        store=store,
    )


def test_store_initial_state_is_closed():
    b = _breaker()
    assert b.state(_now=0.0) == BreakerState.CLOSED


def test_store_failures_below_threshold_stay_closed():
    b = _breaker(failure_threshold=3)
    b.record_failure(_now=1.0)
    b.record_failure(_now=2.0)
    assert b.state(_now=3.0) == BreakerState.CLOSED
    assert b.failure_count(_now=3.0) == 2


def test_store_threshold_reached_trips_open():
    b = _breaker(failure_threshold=3)
    b.record_failure(_now=1.0)
    b.record_failure(_now=2.0)
    state = b.record_failure(_now=3.0)
    assert state == BreakerState.OPEN
    assert b.state(_now=3.0) == BreakerState.OPEN


def test_store_open_transitions_to_half_open_after_cooldown():
    b = _breaker(failure_threshold=2, window_seconds=60.0, cooldown_seconds=30.0)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=2.0) == BreakerState.OPEN
    assert b.is_open(_now=20.0) is True
    assert b.is_open(_now=31.0) is False
    assert b.state(_now=31.0) == BreakerState.HALF_OPEN


def test_store_half_open_to_closed_on_success():
    b = _breaker(failure_threshold=2, window_seconds=60.0, cooldown_seconds=30.0)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.is_open(_now=31.0) is False  # → HALF_OPEN
    b.record_success(_now=32.0)
    assert b.state(_now=32.0) == BreakerState.CLOSED


def test_store_half_open_to_open_on_failure():
    b = _breaker(failure_threshold=2, window_seconds=60.0, cooldown_seconds=30.0)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=31.0) == BreakerState.HALF_OPEN
    b.record_failure(_now=32.0)
    assert b.state(_now=32.0) == BreakerState.OPEN


def test_store_failures_outside_window_evicted():
    b = _breaker(failure_threshold=3, window_seconds=10.0, cooldown_seconds=5.0)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    b.record_failure(_now=15.0)
    assert b.state(_now=15.0) == BreakerState.CLOSED
    assert b.failure_count(_now=15.0) == 1


def test_store_allow_request_closed_returns_true():
    b = _breaker()
    assert b.allow_request(_now=0.0) is True


def test_store_allow_request_open_returns_false():
    b = _breaker(failure_threshold=1)
    b.record_failure(_now=0.0)
    assert b.allow_request(_now=1.0) is False


def test_store_allow_request_half_open_first_call_allowed():
    b = _breaker(failure_threshold=1, cooldown_seconds=30.0)
    b.record_failure(_now=0.0)
    assert b.allow_request(_now=31.0) is True


def test_store_allow_request_half_open_blocks_second_probe():
    b = _breaker(failure_threshold=1, cooldown_seconds=30.0)
    b.record_failure(_now=0.0)
    assert b.allow_request(_now=31.0) is True
    assert b.allow_request(_now=31.0) is False


def test_store_reset_clears_all_state():
    b = _breaker(failure_threshold=2)
    b.record_failure(_now=0.0)
    b.record_failure(_now=1.0)
    assert b.state(_now=2.0) == BreakerState.OPEN
    b.reset()
    assert b.state(_now=2.0) == BreakerState.CLOSED
    assert b.failure_count(_now=2.0) == 0


def test_store_success_in_closed_is_noop():
    b = _breaker(failure_threshold=3)
    b.record_failure(_now=1.0)
    state = b.record_success(_now=2.0)
    assert state == BreakerState.CLOSED
    assert b.failure_count(_now=2.0) == 1


# ── Cross-instance state sharing (the key multi-worker scenario) ──────────────


def test_state_shared_across_instances_same_store():
    """Two CircuitBreakers sharing the same Redis client see each other's state."""
    from triage.breaker_store import RedisBreakerStore

    r = fakeredis.FakeRedis()
    prefix = "test:shared"
    store_a = RedisBreakerStore(r, key_prefix=prefix)
    store_b = RedisBreakerStore(r, key_prefix=prefix)

    b1 = CircuitBreaker(
        failure_threshold=2, window_seconds=60.0, cooldown_seconds=30.0, store=store_a
    )
    b2 = CircuitBreaker(
        failure_threshold=2, window_seconds=60.0, cooldown_seconds=30.0, store=store_b
    )

    b1.record_failure(_now=1.0)
    b1.record_failure(_now=2.0)
    # b1 tripped — b2 should also see OPEN
    assert b2.state(_now=3.0) == BreakerState.OPEN


def test_reset_on_one_instance_visible_to_other():
    from triage.breaker_store import RedisBreakerStore

    r = fakeredis.FakeRedis()
    prefix = "test:reset_shared"
    store_a = RedisBreakerStore(r, key_prefix=prefix)
    store_b = RedisBreakerStore(r, key_prefix=prefix)

    b1 = CircuitBreaker(
        failure_threshold=1, window_seconds=60.0, cooldown_seconds=30.0, store=store_a
    )
    b2 = CircuitBreaker(
        failure_threshold=1, window_seconds=60.0, cooldown_seconds=30.0, store=store_b
    )

    b1.record_failure(_now=1.0)
    assert b2.state(_now=2.0) == BreakerState.OPEN
    b1.reset()
    assert b2.state(_now=2.0) == BreakerState.CLOSED


# ── ttl_seconds ───────────────────────────────────────────────────────────────


def test_ttl_param_accepted():

    from triage.breaker_store import RedisBreakerStore

    r = fakeredis.FakeRedis()
    store = RedisBreakerStore(r, key_prefix="test:ttl", ttl_seconds=3600)
    b = CircuitBreaker(failure_threshold=1, window_seconds=60.0, cooldown_seconds=30.0, store=store)
    b.record_failure(_now=1.0)
    # State persisted and TTLs set — just verify no crash and state is correct
    assert b.state(_now=2.0) == BreakerState.OPEN
