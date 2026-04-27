"""
triage
~~~~~~
Classify why your agent failed. Recover intelligently.
"""

from __future__ import annotations

from triage.agent import Agent, TriageAbortError, TriageEscalationError, agent
from triage.checkpoint import Checkpoint, CheckpointStore, InMemoryCheckpointStore
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureContext, FailureType, Step

__all__ = [
    "Agent",
    "Checkpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FailureContext",
    "FailurePolicy",
    "FailureType",
    "RecoveryAction",
    "Step",
    "TriageAbortError",
    "TriageEscalationError",
    "agent",
    # v0.2 extras (require optional dependencies)
    "LLMClassifier",
    "SQLiteCheckpointStore",
    "RedisCheckpointStore",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> object:
    if name == "LLMClassifier":
        from triage.classifier.llm import LLMClassifier
        return LLMClassifier
    if name == "SQLiteCheckpointStore":
        from triage.checkpoint.sqlite import SQLiteCheckpointStore
        return SQLiteCheckpointStore
    if name == "RedisCheckpointStore":
        from triage.checkpoint.redis import RedisCheckpointStore
        return RedisCheckpointStore
    raise AttributeError(f"module 'triage' has no attribute {name!r}")
