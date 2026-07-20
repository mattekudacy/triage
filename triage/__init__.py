"""
triage
~~~~~~
Classify why your agent failed. Recover intelligently.
"""

from __future__ import annotations

from triage.agent import (
    Agent,
    TriageAbortError,
    TriageEscalationError,
    agent,
    get_recorder,
    get_state_updater,
    get_usage_recorder,
)
from triage.bench import BenchReport, BenchResult, run_benchmark
from triage.breaker import BreakerState, CircuitBreaker
from triage.checkpoint import Checkpoint, CheckpointStore, InMemoryCheckpointStore
from triage.feedback import Correction, coverage_report, record_correction
from triage.policy import FailurePolicy, RecoveryAction
from triage.scorer.base import RiskScore, StepRiskScorer
from triage.scorer.rules import RulesRiskScorer
from triage.taxonomy import FailureContext, FailureType, Step, TriageContext
from triage.testing import RecordingAgent, assert_classifies_as, make_step
from triage.usage import Usage, UsageMeter

__all__ = [
    "Agent",
    "BenchReport",
    "BenchResult",
    "Checkpoint",
    "CheckpointStore",
    "Correction",
    "coverage_report",
    "InMemoryCheckpointStore",
    "FailureContext",
    "TriageContext",
    "FailurePolicy",
    "FailureType",
    "RecoveryAction",
    "Step",
    "TriageAbortError",
    "TriageEscalationError",
    "agent",
    "get_recorder",
    "get_state_updater",
    "get_usage_recorder",
    "record_correction",
    "run_benchmark",
    "RiskScore",
    "RulesRiskScorer",
    "StepRiskScorer",
    # testing utilities
    "RecordingAgent",
    "assert_classifies_as",
    "make_step",
    # circuit breaker
    "BreakerState",
    "CircuitBreaker",
    # usage accounting
    "Usage",
    "UsageMeter",
    # v0.2+ extras (require optional dependencies)
    "LLMClassifier",
    "HybridClassifier",
    "SQLiteCheckpointStore",
    "RedisCheckpointStore",
]

__version__ = "0.16.0"


def __getattr__(name: str) -> object:
    if name == "LLMClassifier":
        from triage.classifier.llm import LLMClassifier
        return LLMClassifier
    if name == "HybridClassifier":
        from triage.classifier.hybrid import HybridClassifier
        return HybridClassifier
    if name == "SQLiteCheckpointStore":
        from triage.checkpoint.sqlite import SQLiteCheckpointStore
        return SQLiteCheckpointStore
    if name == "RedisCheckpointStore":
        from triage.checkpoint.redis import RedisCheckpointStore
        return RedisCheckpointStore
    raise AttributeError(f"module 'triage' has no attribute {name!r}")
