from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from triage.taxonomy import Step
    from triage.trajectory import Trajectory


@dataclass
class RiskScore:
    """Risk assessment for a single recorded step."""

    score: float
    reason: str | None = None

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))


@runtime_checkable
class StepRiskScorer(Protocol):
    """Synchronous per-step risk scorer. Must not make API calls."""

    def __call__(self, step: "Step", trajectory: "Trajectory") -> RiskScore:
        ...
