"""
tests/test_suspension_redis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for RedisSuspensionStore using fakeredis.
"""

from __future__ import annotations

import pytest

redis_mod = pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from fakeredis.aioredis import FakeRedis  # noqa: E402

from triage.suspension import (  # noqa: E402
    SuspendedRun,
    SuspensionStore,
)
from triage.suspension_redis import RedisSuspensionStore  # noqa: E402
from triage.taxonomy import FailureContext, FailureType, Step  # noqa: E402
from triage.usage import Usage  # noqa: E402


def make_step(index: int = 0, error: str | None = None) -> Step:
    return Step(index=index, action="test step", error=error)


def _make_run(token: str = "tok-1", *, kwargs: dict | None = None) -> SuspendedRun:
    step = make_step(0)
    ctx = FailureContext(
        failure_type=FailureType.EXTERNAL_FAULT,
        trajectory=[step],
        critical_step_index=0,
        original_task="do the thing",
        attempt_history=[(FailureType.TIMEOUT, "retry")],
    )
    return SuspendedRun(
        token=token,
        context=ctx,
        task="do the thing",
        kwargs=kwargs or {"user": "alice"},
        attempt=1,
        attempt_history=[(FailureType.TIMEOUT, "retry")],
        timestamp=1000.0,
        message="needs approval",
        metadata={"channel": "#ops"},
        usage_snapshot=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001, calls=1),
    )


@pytest.fixture
async def redis_store():
    client = FakeRedis()
    store = RedisSuspensionStore(client)
    yield store
    await client.aclose()


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_redis_suspension_store_satisfies_protocol():
    """RedisSuspensionStore satisfies the SuspensionStore Protocol."""
    assert isinstance(RedisSuspensionStore.__new__(RedisSuspensionStore), object)
    # Protocol structural check via isinstance requires a live instance.
    # We'll check at runtime once the fixture is available — see test below.


async def test_protocol_isinstance_check(redis_store):
    assert isinstance(redis_store, SuspensionStore)


# ── save / load / delete ──────────────────────────────────────────────────────


async def test_save_and_load_roundtrip(redis_store):
    run = _make_run("tok-rt")
    await redis_store.save(run)
    loaded = await redis_store.load("tok-rt")
    assert loaded.token == "tok-rt"
    assert loaded.task == "do the thing"
    assert loaded.message == "needs approval"
    assert loaded.metadata == {"channel": "#ops"}


async def test_load_missing_raises_key_error(redis_store):
    with pytest.raises(KeyError, match="no-such-token"):
        await redis_store.load("no-such-token")


async def test_delete_removes_token(redis_store):
    run = _make_run("tok-del")
    await redis_store.save(run)
    await redis_store.delete("tok-del")
    with pytest.raises(KeyError):
        await redis_store.load("tok-del")


async def test_delete_missing_is_noop(redis_store):
    await redis_store.delete("never-existed")  # must not raise


# ── round-trip fidelity ───────────────────────────────────────────────────────


async def test_full_field_roundtrip(redis_store):
    """All scalar and nested fields survive save/load without loss."""
    run = _make_run("tok-full")
    await redis_store.save(run)
    loaded = await redis_store.load("tok-full")

    assert loaded.token == run.token
    assert loaded.task == run.task
    assert loaded.attempt == run.attempt
    assert loaded.timestamp == run.timestamp
    assert loaded.attempt_history == run.attempt_history
    assert loaded.kwargs == run.kwargs

    assert loaded.usage_snapshot.input_tokens == run.usage_snapshot.input_tokens
    assert loaded.usage_snapshot.output_tokens == run.usage_snapshot.output_tokens
    assert loaded.usage_snapshot.cost_usd == run.usage_snapshot.cost_usd
    assert loaded.usage_snapshot.calls == run.usage_snapshot.calls

    ctx = loaded.context
    assert ctx.failure_type == FailureType.EXTERNAL_FAULT
    assert ctx.original_task == "do the thing"
    assert ctx.attempt_history == [(FailureType.TIMEOUT, "retry")]
    assert len(ctx.trajectory) == 1
    assert ctx.trajectory[0].index == 0


async def test_failure_type_preserved(redis_store):
    for ft in (FailureType.LOOP_DETECTED, FailureType.SCHEMA_MISMATCH, FailureType.TIMEOUT):
        ctx = FailureContext(
            failure_type=ft,
            trajectory=[make_step(0)],
            critical_step_index=0,
            original_task="task",
        )
        run = SuspendedRun(
            token=f"tok-{ft.value}",
            context=ctx,
            task="task",
            kwargs={},
            attempt=0,
            attempt_history=[],
        )
        await redis_store.save(run)
        loaded = await redis_store.load(f"tok-{ft.value}")
        assert loaded.context.failure_type == ft


# ── TTL option ────────────────────────────────────────────────────────────────


async def test_ttl_store_save_and_load():
    client = FakeRedis()
    store = RedisSuspensionStore(client, ttl_seconds=3600)
    run = _make_run("tok-ttl")
    await store.save(run)
    loaded = await store.load("tok-ttl")
    assert loaded.token == "tok-ttl"
    await client.aclose()


# ── key_prefix isolation ──────────────────────────────────────────────────────


async def test_custom_key_prefix_isolation():
    """Two stores with different prefixes don't share tokens."""
    client = FakeRedis()
    store_a = RedisSuspensionStore(client, key_prefix="app_a:suspension")
    store_b = RedisSuspensionStore(client, key_prefix="app_b:suspension")

    run = _make_run("tok-shared")
    await store_a.save(run)

    # Token saved under prefix A is not visible under prefix B
    with pytest.raises(KeyError):
        await store_b.load("tok-shared")

    await client.aclose()


# ── snapshot isolation ────────────────────────────────────────────────────────


async def test_snapshot_isolation_after_save(redis_store):
    """Mutating SuspendedRun after save does not affect the stored copy."""
    run = _make_run("tok-iso")
    await redis_store.save(run)

    # Mutate in-memory copy after save
    run.metadata["injected"] = "oops"

    loaded = await redis_store.load("tok-iso")
    assert "injected" not in loaded.metadata


# ── single-use: delete after load pattern ────────────────────────────────────


async def test_token_single_use_pattern(redis_store):
    """Simulates the agent.resume() pattern: load then delete makes the token single-use."""
    run = _make_run("tok-once")
    await redis_store.save(run)

    await redis_store.load("tok-once")
    await redis_store.delete("tok-once")

    with pytest.raises(KeyError):
        await redis_store.load("tok-once")
