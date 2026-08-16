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
    TriageSuspendedError,
    agent,
    get_compensator_recorder,
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
from triage.streaming import StreamRetryEvent
from triage.suspension import (
    InMemorySuspensionStore,
    SuspendedRun,
    SuspensionStore,
    deserialize_run,
    serialize_run,
)
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
    "TriageSuspendedError",
    "InMemorySuspensionStore",
    "SuspendedRun",
    "SuspensionStore",
    "serialize_run",
    "deserialize_run",
    "agent",
    "get_compensator_recorder",
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
    # streaming
    "StreamRetryEvent",
    # circuit breaker
    "BreakerState",
    "CircuitBreaker",
    "BreakerStore",
    "RedisBreakerStore",
    # usage accounting
    "Usage",
    "UsageMeter",
    # v0.2+ extras (require optional dependencies)
    "LLMClassifier",
    "HybridClassifier",
    "SQLiteCheckpointStore",
    "RedisCheckpointStore",
    "RedisSuspensionStore",
]

__version__ = "1.0.1"


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
    if name == "BreakerStore":
        from triage.breaker_store import BreakerStore

        return BreakerStore
    if name == "RedisBreakerStore":
        from triage.breaker_store import RedisBreakerStore

        return RedisBreakerStore
    if name == "RedisSuspensionStore":
        from triage.suspension_redis import RedisSuspensionStore

        return RedisSuspensionStore
    raise AttributeError(f"module 'triage' has no attribute {name!r}")
