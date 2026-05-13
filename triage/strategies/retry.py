"""
triage.strategies.retry
~~~~~~~~~~~~~~~~~~~~~~~
Factory functions for retry-flavored recovery actions.
"""

from __future__ import annotations

from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def retry_with_tool_manifest(max_attempts: int = 3) -> StrategyFn:
    """Retry with a hint to use only tools in the current manifest.

    Escalates after ``max_attempts`` retries rather than deferring entirely
    to ``Agent(max_recovery_attempts=...)``.
    """
    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        prior = sum(1 for _, kind in ctx.attempt_history if kind == "retry")
        if prior >= max_attempts:
            return RecoveryAction.ESCALATE(
                f"retry_with_tool_manifest: max_attempts ({max_attempts}) reached."
            )
        return RecoveryAction.RETRY(
            hint="Re-run using only tools in the current manifest.",
        )
    return _strategy


def backoff_and_retry(max_attempts: int = 5) -> StrategyFn:
    """Retry with exponential backoff delay. agent.py calls anyio.sleep(delay).

    Escalates after ``max_attempts`` retries rather than deferring entirely
    to ``Agent(max_recovery_attempts=...)``.
    """
    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        prior = sum(1 for _, kind in ctx.attempt_history if kind == "retry")
        if prior >= max_attempts:
            return RecoveryAction.ESCALATE(
                f"backoff_and_retry: max_attempts ({max_attempts}) reached."
            )
        attempt = ctx.metadata.get("attempt_number", 0)
        return RecoveryAction.RETRY(
            hint="External fault. Retry with exponential backoff.",
            delay=float(2 ** attempt),
        )
    return _strategy
