"""
triage.checkpoint.memory
~~~~~~~~~~~~~~~~~~~~~~~~
In-memory CheckpointStore implementation. Default for development and testing.
"""

from __future__ import annotations

import anyio

from triage.checkpoint.base import Checkpoint


class InMemoryCheckpointStore:
    """Default in-memory implementation.

    Guards its dict with an ``anyio.Lock`` so concurrent ``save``/``load``/
    ``latest`` calls — e.g. from multiple concurrent ``Agent.run()`` calls
    sharing this store — don't race on the underlying dict. This only
    serializes access to *this* store's dict; it does not make checkpoint
    semantics (like "rollback to the latest checkpoint") safe under
    concurrent writers racing to decide what "latest" means for their own
    recovery.
    """

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._lock = anyio.Lock()

    async def save(self, checkpoint: Checkpoint) -> None:
        # Copy snapshot to prevent external mutations corrupting stored data
        safe = Checkpoint(
            id=checkpoint.id,
            timestamp=checkpoint.timestamp,
            state=dict(checkpoint.state),
            trajectory_snapshot=list(checkpoint.trajectory_snapshot),
            run_id=checkpoint.run_id,
        )
        async with self._lock:
            self._store[safe.id] = safe

    async def load(self, id: str) -> Checkpoint:
        async with self._lock:
            if id not in self._store:
                raise KeyError(f"No checkpoint with id {id!r}")
            return self._store[id]

    async def latest(self, run_id: str | None = None) -> Checkpoint | None:
        async with self._lock:
            candidates = (
                [c for c in self._store.values() if c.run_id == run_id]
                if run_id is not None
                else list(self._store.values())
            )
            if not candidates:
                return None
            return max(candidates, key=lambda c: c.timestamp)
