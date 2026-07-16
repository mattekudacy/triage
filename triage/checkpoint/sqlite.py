"""
triage.checkpoint.sqlite
~~~~~~~~~~~~~~~~~~~~~~~~
File-based persistent CheckpointStore backed by SQLite via aiosqlite.

Install: pip install triage-agent[sqlite]
"""

from __future__ import annotations

import json

try:
    import aiosqlite
except ImportError as exc:
    raise ImportError(
        "SQLiteCheckpointStore requires 'aiosqlite'. "
        "Install it with: pip install triage-agent[sqlite]"
    ) from exc

from triage.checkpoint.base import (
    Checkpoint,
    _dict_to_step,
    _safe_json,
    _step_to_dict,
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id        TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    state     TEXT NOT NULL,
    trajectory TEXT NOT NULL,
    run_id    TEXT
)
"""

_MIGRATE_ADD_RUN_ID = "ALTER TABLE checkpoints ADD COLUMN run_id TEXT"


class SQLiteCheckpointStore:
    """Persistent CheckpointStore backed by SQLite.

    Pass a file path for durable storage, or use a shared-memory URI for testing::

        store = SQLiteCheckpointStore("runs/checkpoints.db")

    Not safe for concurrent writes from multiple processes.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._table_created = False

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        if not self._table_created:
            await db.execute(_CREATE_TABLE)
            # Migrate existing tables that predate the run_id column
            try:
                await db.execute(_MIGRATE_ADD_RUN_ID)
            except Exception:
                pass  # column already exists
            await db.commit()
            self._table_created = True

    async def save(self, checkpoint: Checkpoint) -> None:
        state_json = json.dumps(_safe_json(checkpoint.state))
        traj_json = json.dumps([_step_to_dict(s) for s in checkpoint.trajectory_snapshot])
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            await db.execute(
                "INSERT OR REPLACE INTO checkpoints (id, timestamp, state, trajectory, run_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (checkpoint.id, checkpoint.timestamp, state_json, traj_json, checkpoint.run_id),
            )
            await db.commit()

    async def load(self, id: str) -> Checkpoint:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            async with db.execute(
                "SELECT id, timestamp, state, trajectory, run_id FROM checkpoints WHERE id = ?",
                (id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"No checkpoint with id {id!r}")
        return Checkpoint(
            id=row[0],
            timestamp=row[1],
            state=json.loads(row[2]),
            trajectory_snapshot=[_dict_to_step(d) for d in json.loads(row[3])],
            run_id=row[4],
        )

    async def latest(self, run_id: str | None = None) -> Checkpoint | None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            if run_id is not None:
                async with db.execute(
                    "SELECT id, timestamp, state, trajectory, run_id "
                    "FROM checkpoints WHERE run_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (run_id,),
                ) as cursor:
                    row = await cursor.fetchone()
            else:
                async with db.execute(
                    "SELECT id, timestamp, state, trajectory, run_id "
                    "FROM checkpoints ORDER BY timestamp DESC LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
        if row is None:
            return None
        return Checkpoint(
            id=row[0],
            timestamp=row[1],
            state=json.loads(row[2]),
            trajectory_snapshot=[_dict_to_step(d) for d in json.loads(row[3])],
            run_id=row[4],
        )
