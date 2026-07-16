"""
triage.classifier.base
~~~~~~~~~~~~~~~~~~~~~~
Structural protocol that all classifiers must implement.
"""

from __future__ import annotations

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

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...
