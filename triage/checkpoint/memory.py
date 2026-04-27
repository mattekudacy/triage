"""
triage.checkpoint.memory
~~~~~~~~~~~~~~~~~~~~~~~~
In-memory CheckpointStore implementation. Default for development and testing.
Not concurrency-safe.
"""

from __future__ import annotations

from triage.checkpoint.base import Checkpoint


class InMemoryCheckpointStore:
    """Default in-memory implementation. Not concurrency-safe."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        # Copy snapshot to prevent external mutations corrupting stored data
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
