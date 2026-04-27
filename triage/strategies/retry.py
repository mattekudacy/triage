"""
triage.strategies.retry
~~~~~~~~~~~~~~~~~~~~~~~
Factory functions for retry-flavored recovery actions.
"""

from __future__ import annotations

from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def retry_with_tool_manifest(max_attempts: int = 3) -> StrategyFn:
    """Retry with a hint to use only tools in the current manifest."""
    async def _strategy(ctx: FailureContext) -> RecoveryAction:  # noqa: ARG001
        return RecoveryAction.RETRY(
            hint="Re-run using only tools in the current manifest.",
            inject={"max_attempts": max_attempts},
        )
    return _strategy


def backoff_and_retry(max_attempts: int = 5) -> StrategyFn:
    """Retry with exponential backoff delay. agent.py calls anyio.sleep(delay)."""
    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        attempt = ctx.metadata.get("attempt_number", 0)
        return RecoveryAction.RETRY(
            hint="External fault. Retry with exponential backoff.",
            inject={"max_attempts": max_attempts},
            delay=float(2 ** attempt),
        )
    return _strategy
