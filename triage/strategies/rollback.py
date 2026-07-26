"""
triage.strategies.rollback
~~~~~~~~~~~~~~~~~~~~~~~~~~
Factory function for checkpoint-restore recovery actions.
"""

from __future__ import annotations

from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def rollback_to_checkpoint(checkpoint_id: str | None = None) -> StrategyFn:
    """Restore state to a named checkpoint (or latest if not specified)."""

    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        return RecoveryAction.ROLLBACK(
            checkpoint_id=checkpoint_id or ctx.last_checkpoint_id,
        )

    return _strategy
