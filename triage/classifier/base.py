"""
triage.classifier.base
~~~~~~~~~~~~~~~~~~~~~~
Structural protocol that all classifiers must implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from triage.taxonomy import FailureType

if TYPE_CHECKING:
    from triage.trajectory import Trajectory


@runtime_checkable
class Classifier(Protocol):
    """Synchronous failure classifier. Must not make any API calls.

    ``classify()`` remains the required, synchronous contract (``agent.py``
    calls it via ``anyio.to_thread.run_sync`` on the failure path). Classifiers
    that talk to an LLM API may additionally define an optional
    ``async def aclassify(self, trajectory, task) -> FailureType`` method — when
    present, ``agent.py`` awaits it directly instead of running ``classify()``
    in a thread, avoiding that hop. This is not part of the ``Classifier``
    protocol itself (kept structural/duck-typed) since most classifiers, like
    ``RulesClassifier``, have no I/O to make async.
    """

    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...


@dataclass(frozen=True)
class ClassificationResult:
    """A ``FailureType`` plus the classifier's confidence in it (0.0-1.0).

    Returned by the optional, duck-typed ``classify_with_confidence()`` /
    ``aclassify_with_confidence()`` methods a classifier may define — not part
    of the ``Classifier`` protocol itself, same reasoning as ``aclassify()``:
    ``RulesClassifier`` has no meaningful confidence to report (its answers are
    100%-precision-by-construction or ``UNKNOWN``), so this isn't forced on
    every classifier. ``HybridClassifier(confidence_threshold=...)`` checks for
    these methods via ``getattr`` and, when present, only trusts a non-``UNKNOWN``
    answer whose confidence meets the threshold — see
    ``docs/known-limitations.md``'s "close the recall gap, but not the precision
    gap" section for the measurement this exists to address, and
    ``LLMClassifier.classify_with_confidence()`` for the reference implementation.
    """

    failure_type: FailureType
    confidence: float
