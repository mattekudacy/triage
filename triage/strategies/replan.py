"""
triage.strategies.replan
~~~~~~~~~~~~~~~~~~~~~~~~
Factory functions for plan-mutation recovery actions.
"""

from __future__ import annotations

from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def replan(hint: str | None = None, max_replans: int = 3) -> StrategyFn:
    """Abort the current plan and generate a new one.

    Escalates after ``max_replans`` replan attempts rather than deferring
    entirely to ``Agent(max_recovery_attempts=...)``.
    """
    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        prior = sum(1 for _, kind in ctx.attempt_history if kind == "replan")
        if prior >= max_replans:
            return RecoveryAction.ESCALATE(
                f"replan: max_replans ({max_replans}) reached."
            )
        return RecoveryAction.REPLAN(
            hint=hint or "Generate a new plan. The previous approach failed.",
        )
    return _strategy


def resume_from_subgoal() -> StrategyFn:
    """Continue execution from the first incomplete sub-goal.

    Requires ``ctx.metadata["incomplete_subgoal"]`` to be set by the agent
    before raising. If the key is missing, escalates rather than silently
    performing a no-op retry.
    """
    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        subgoal = ctx.metadata.get("incomplete_subgoal")
        if subgoal is None:
            return RecoveryAction.ESCALATE(
                "resume_from_subgoal: 'incomplete_subgoal' not found in "
                "ctx.metadata. Set it before raising to enable resume recovery."
            )
        return RecoveryAction.RESUME(from_subgoal=subgoal)
    return _strategy
