"""
triage.checkpoint
~~~~~~~~~~~~~~~~~
Snapshot agent state + trajectory at a point in time.
Protocol-based so Redis, SQLite, or other backends can be swapped in.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from triage.taxonomy import Step


@dataclass
class Checkpoint:
    """Snapshot of agent state at a point in time."""

    id: str
    timestamp: float
    state: dict[str, Any]
    trajectory_snapshot: list[Step]


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def load(self, id: str) -> Checkpoint: ...
    async def latest(self) -> Checkpoint | None: ...


class InMemoryCheckpointStore:
    """Default in-memory implementation. Not concurrency-safe."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        # Copy the snapshot so external mutations don't corrupt the stored checkpoint
        safe = Checkpoint(
            id=checkpoint.id,
            timestamp=checkpoint.timestamp,
            state=dict(checkpoint.state),
            trajectory_snapshot=list(checkpoint.trajectory_snapshot),
        )
        self._store[safe.id] = safe

    async def load(self, id: str) -> Checkpoint:
        if id not in self._store:
            raise KeyError(f"No checkpoint with id {id!r}")
        return self._store[id]

    async def latest(self) -> Checkpoint | None:
        if not self._store:
            return None
        return max(self._store.values(), key=lambda c: c.timestamp)


def make_checkpoint(
    state: dict[str, Any],
    trajectory_steps: list[Step],
    id: str | None = None,
) -> Checkpoint:
    """Convenience constructor. Generates a UUID id if not supplied."""
    return Checkpoint(
        id=id or str(uuid.uuid4()),
        timestamp=time.time(),
        state=dict(state),
        trajectory_snapshot=list(trajectory_steps),
    )
