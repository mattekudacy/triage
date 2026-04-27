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
]

__version__ = "0.1.0"
