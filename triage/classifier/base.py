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
    """Synchronous failure classifier. Must not make any API calls."""

    def classify(self, trajectory: "Trajectory", task: str) -> FailureType:
        ...
