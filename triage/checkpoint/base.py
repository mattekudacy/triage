"""
triage.checkpoint.base
~~~~~~~~~~~~~~~~~~~~~~
Core checkpoint types: Checkpoint dataclass, CheckpointStore protocol,
and the make_checkpoint convenience constructor.

Serialization helpers (_step_to_dict, _dict_to_step, _safe_json) are also
defined here so SQLite and Redis backends can import them without circular deps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from triage.taxonomy import Step


@dataclass
class Checkpoint:
    """Snapshot of agent state at a point in time."""

    id: str
    timestamp: float
    state: dict[str, Any]
    trajectory_snapshot: list[Step]
    run_id: str | None = None


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None: ...
    async def load(self, id: str) -> Checkpoint: ...
    async def latest(self, run_id: str | None = None) -> Checkpoint | None: ...


def make_checkpoint(
    state: dict[str, Any],
    trajectory_steps: list[Step],
    id: str | None = None,
    run_id: str | None = None,
) -> Checkpoint:
    """Convenience constructor. Generates a UUID id if not supplied."""
    return Checkpoint(
        id=id or str(uuid.uuid4()),
        timestamp=time.time(),
        state=dict(state),
        trajectory_snapshot=list(trajectory_steps),
        run_id=run_id,
    )


# ── Serialization helpers (used by SQLite and Redis backends) ─────────────────

def _safe_json(val: Any) -> Any:
    """Convert a value to something JSON-serializable. Falls back to repr()."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, dict):
        return {str(k): _safe_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_safe_json(v) for v in val]
    return repr(val)


def _step_to_dict(step: Step) -> dict[str, Any]:
    return {
        "index": step.index,
        "action": step.action,
        "tool_called": step.tool_called,
        "tool_input": _safe_json(step.tool_input),
        "tool_output": _safe_json(step.tool_output),
        "llm_output": step.llm_output,
        "error": step.error,
        "timestamp": step.timestamp,
        "state_hash": step.state_hash,
        "metadata": {k: _safe_json(v) for k, v in step.metadata.items()},
    }


def _dict_to_step(d: dict[str, Any]) -> Step:
    return Step(
        index=d["index"],
        action=d["action"],
        tool_called=d.get("tool_called"),
        tool_input=d.get("tool_input"),
        tool_output=d.get("tool_output"),
        llm_output=d.get("llm_output"),
        error=d.get("error"),
        timestamp=d.get("timestamp", 0.0),
        state_hash=d.get("state_hash"),
        metadata=d.get("metadata", {}),
    )
