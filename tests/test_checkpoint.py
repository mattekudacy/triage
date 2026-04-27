"""Tests for triage.checkpoint — Checkpoint, InMemoryCheckpointStore, make_checkpoint."""

import time

import pytest

from triage.checkpoint import Checkpoint, InMemoryCheckpointStore, make_checkpoint
from triage.taxonomy import Step


def make_step(index: int = 0) -> Step:
    return Step(index=index, action="step")


# ── make_checkpoint helper ────────────────────────────────────────────────────

def test_make_checkpoint_generates_id():
    cp = make_checkpoint(state={}, trajectory_steps=[])
    assert isinstance(cp.id, str)
    assert cp.id  # non-empty


def test_make_checkpoint_uses_supplied_id():
    cp = make_checkpoint(state={}, trajectory_steps=[], id="my-checkpoint")
    assert cp.id == "my-checkpoint"


def test_make_checkpoint_copies_steps():
    steps = [make_step(0)]
    cp = make_checkpoint(state={}, trajectory_steps=steps)
    steps.append(make_step(1))
    assert len(cp.trajectory_snapshot) == 1  # not affected by external mutation


# ── save / load round-trip ────────────────────────────────────────────────────

async def test_save_and_load():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint(state={"key": "val"}, trajectory_steps=[make_step()])
    await store.save(cp)
    loaded = await store.load(cp.id)
    assert loaded.id == cp.id
    assert loaded.state == {"key": "val"}
    assert len(loaded.trajectory_snapshot) == 1


async def test_load_missing_raises_key_error():
    store = InMemoryCheckpointStore()
    with pytest.raises(KeyError):
        await store.load("nonexistent-id")


# ── latest() ─────────────────────────────────────────────────────────────────

async def test_latest_empty_store():
    store = InMemoryCheckpointStore()
    assert await store.latest() is None


async def test_latest_single_checkpoint():
    store = InMemoryCheckpointStore()
    cp = make_checkpoint(state={}, trajectory_steps=[])
    await store.save(cp)
    assert (await store.latest()).id == cp.id


async def test_latest_returns_most_recent_by_timestamp():
    store = InMemoryCheckpointStore()
    old = Checkpoint(id="old", timestamp=1000.0, state={}, trajectory_snapshot=[])
    new = Checkpoint(id="new", timestamp=9999.0, state={}, trajectory_snapshot=[])
    # Save old first so insertion order would return old if implementation is wrong
    await store.save(old)
    await store.save(new)
    assert (await store.latest()).id == "new"


# ── snapshot isolation ────────────────────────────────────────────────────────

async def test_snapshot_copy_on_save():
    store = InMemoryCheckpointStore()
    steps = [make_step(0)]
    cp = Checkpoint(id="iso", timestamp=time.time(), state={}, trajectory_snapshot=steps)
    await store.save(cp)

    # Mutate original list after saving
    steps.append(make_step(1))

    loaded = await store.load("iso")
    assert len(loaded.trajectory_snapshot) == 1  # store copy unaffected
