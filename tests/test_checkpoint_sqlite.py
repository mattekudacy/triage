"""
tests/test_checkpoint_sqlite.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for SQLiteCheckpointStore.
"""

from __future__ import annotations

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

from triage.checkpoint.base import Checkpoint, make_checkpoint
from triage.checkpoint.sqlite import SQLiteCheckpointStore
from triage.taxonomy import Step


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


async def test_save_and_load_roundtrip(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    store = SQLiteCheckpointStore(db)
    cp = make_checkpoint(
        state={"key": "value"},
        trajectory_steps=[make_step(0), make_step(1)],
    )
    await store.save(cp)
    loaded = await store.load(cp.id)
    assert loaded.id == cp.id
    assert loaded.state == {"key": "value"}
    assert len(loaded.trajectory_snapshot) == 2


async def test_load_raises_for_missing_id(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.db"))
    with pytest.raises(KeyError, match="nonexistent"):
        await store.load("nonexistent")


async def test_latest_returns_none_when_empty(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.db"))
    result = await store.latest()
    assert result is None


async def test_latest_returns_most_recent(tmp_path):
    import time
    db = str(tmp_path / "checkpoints.db")
    store = SQLiteCheckpointStore(db)

    cp_old = make_checkpoint(state={}, trajectory_steps=[])
    await store.save(cp_old)
    time.sleep(0.01)
    cp_new = make_checkpoint(state={"newer": True}, trajectory_steps=[])
    await store.save(cp_new)

    latest = await store.latest()
    assert latest is not None
    assert latest.id == cp_new.id


async def test_save_replaces_existing_id(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    store = SQLiteCheckpointStore(db)
    cp = make_checkpoint(state={"v": 1}, trajectory_steps=[])
    await store.save(cp)

    updated = Checkpoint(
        id=cp.id,
        timestamp=cp.timestamp + 1,
        state={"v": 2},
        trajectory_snapshot=[],
    )
    await store.save(updated)

    loaded = await store.load(cp.id)
    assert loaded.state == {"v": 2}


async def test_snapshot_isolation_after_save(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    store = SQLiteCheckpointStore(db)
    steps = [make_step(0)]
    cp = make_checkpoint(state={}, trajectory_steps=steps)
    await store.save(cp)

    steps.append(make_step(1))  # mutate original list

    loaded = await store.load(cp.id)
    assert len(loaded.trajectory_snapshot) == 1


async def test_step_fields_preserved(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    store = SQLiteCheckpointStore(db)
    step = make_step(
        index=5,
        tool_called="search",
        tool_input={"q": "hello"},
        error="timeout",
        llm_output="result text",
    )
    cp = make_checkpoint(state={}, trajectory_steps=[step])
    await store.save(cp)

    loaded = await store.load(cp.id)
    s = loaded.trajectory_snapshot[0]
    assert s.index == 5
    assert s.tool_called == "search"
    assert s.tool_input == {"q": "hello"}
    assert s.error == "timeout"
    assert s.llm_output == "result text"
