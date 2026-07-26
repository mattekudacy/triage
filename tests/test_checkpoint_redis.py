"""
tests/test_checkpoint_redis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for RedisCheckpointStore using fakeredis.
"""

from __future__ import annotations

import pytest

redis_mod = pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from fakeredis.aioredis import FakeRedis  # noqa: E402

from triage.checkpoint.base import Checkpoint, make_checkpoint  # noqa: E402
from triage.checkpoint.redis import RedisCheckpointStore  # noqa: E402
from triage.taxonomy import Step  # noqa: E402


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
) -> Step:
    return Step(
        index=index,
        action="test step",
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
    )


@pytest.fixture
async def redis_store():
    client = FakeRedis()
    store = RedisCheckpointStore(client)
    yield store
    await client.aclose()


async def test_save_and_load_roundtrip(redis_store):
    cp = make_checkpoint(
        state={"key": "value"},
        trajectory_steps=[make_step(0), make_step(1)],
    )
    await redis_store.save(cp)
    loaded = await redis_store.load(cp.id)
    assert loaded.id == cp.id
    assert loaded.state == {"key": "value"}
    assert len(loaded.trajectory_snapshot) == 2


async def test_load_raises_for_missing_id(redis_store):
    with pytest.raises(KeyError, match="ghost"):
        await redis_store.load("ghost")


async def test_latest_returns_none_when_empty(redis_store):
    result = await redis_store.latest()
    assert result is None


async def test_latest_returns_most_recent(redis_store):
    import time

    cp_old = make_checkpoint(state={}, trajectory_steps=[])
    await redis_store.save(cp_old)
    time.sleep(0.01)
    cp_new = make_checkpoint(state={"newer": True}, trajectory_steps=[])
    await redis_store.save(cp_new)

    latest = await redis_store.latest()
    assert latest is not None
    assert latest.id == cp_new.id


async def test_save_overwrites_existing_key(redis_store):
    cp = make_checkpoint(state={"v": 1}, trajectory_steps=[])
    await redis_store.save(cp)

    updated = Checkpoint(
        id=cp.id,
        timestamp=cp.timestamp + 1,
        state={"v": 2},
        trajectory_snapshot=[],
    )
    await redis_store.save(updated)

    loaded = await redis_store.load(cp.id)
    assert loaded.state == {"v": 2}


async def test_snapshot_isolation_after_save(redis_store):
    steps = [make_step(0)]
    cp = make_checkpoint(state={}, trajectory_steps=steps)
    await redis_store.save(cp)

    steps.append(make_step(1))  # mutate original list

    loaded = await redis_store.load(cp.id)
    assert len(loaded.trajectory_snapshot) == 1


async def test_step_fields_preserved(redis_store):
    step = make_step(
        index=3,
        tool_called="fetch",
        tool_input={"url": "http://example.com"},
        error=None,
        llm_output="fetched content",
    )
    cp = make_checkpoint(state={}, trajectory_steps=[step])
    await redis_store.save(cp)

    loaded = await redis_store.load(cp.id)
    s = loaded.trajectory_snapshot[0]
    assert s.index == 3
    assert s.tool_called == "fetch"
    assert s.tool_input == {"url": "http://example.com"}
    assert s.llm_output == "fetched content"


async def test_multiple_checkpoints_indexed(redis_store):
    import time

    cps = []
    for i in range(3):
        cp = make_checkpoint(state={"i": i}, trajectory_steps=[])
        cps.append(cp)
        await redis_store.save(cp)
        time.sleep(0.005)

    latest = await redis_store.latest()
    assert latest is not None
    assert latest.id == cps[-1].id
