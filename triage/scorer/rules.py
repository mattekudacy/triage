from __future__ import annotations

import re
from typing import TYPE_CHECKING

from triage.scorer.base import RiskScore

if TYPE_CHECKING:
    from triage.taxonomy import Step
    from triage.trajectory import Trajectory

_HIGH_RISK_RE = re.compile(
    r"\b(send_email|send_message|charge_card|payment|delete\b|drop\s+table"
    r"|rm\s+-rf|truncate|destroy|nuke|wipe|purge)\b",
    re.IGNORECASE,
)
_MEDIUM_RISK_RE = re.compile(
    r"\b(write|upload|post|put|patch|update|insert|create|deploy|publish)\b",
    re.IGNORECASE,
)


class RulesRiskScorer:
    """Pattern-based step risk scorer. Zero API calls, synchronous.

    Parameters
    ----------
    high_risk_patterns:
        Additional patterns to treat as high-risk (score 0.95). ORed with
        built-in patterns.
    medium_risk_patterns:
        Additional patterns to treat as medium-risk (score 0.65).
    """

    def __init__(
        self,
        high_risk_patterns: list[str] | None = None,
        medium_risk_patterns: list[str] | None = None,
    ) -> None:
        extra_high = "|".join(re.escape(p) for p in (high_risk_patterns or []))
        extra_med = "|".join(re.escape(p) for p in (medium_risk_patterns or []))
        self._high_re = (
            re.compile(f"{_HIGH_RISK_RE.pattern}|{extra_high}", re.IGNORECASE)
            if extra_high
            else _HIGH_RISK_RE
        )
        self._med_re = (
            re.compile(f"{_MEDIUM_RISK_RE.pattern}|{extra_med}", re.IGNORECASE)
            if extra_med
            else _MEDIUM_RISK_RE
        )

    def __call__(self, step: Step, trajectory: Trajectory) -> RiskScore:
        text = " ".join(filter(None, [step.action, step.tool_called]))

        if self._high_re.search(text):
            return RiskScore(score=0.95, reason=f"high-risk pattern in: {text!r}")

        if self._med_re.search(text):
            base = 0.65
            if not step.idempotent:
                base = min(1.0, base + 0.1)
            return RiskScore(score=base, reason=f"medium-risk pattern in: {text!r}")

        if not step.idempotent:
            return RiskScore(score=0.2, reason="step marked non-idempotent")

        return RiskScore(score=0.0)
