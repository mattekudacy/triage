"""
triage.policy
~~~~~~~~~~~~~
Declarative failure policy. Maps FailureType values to recovery
strategy callables. The policy is the user-facing configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from triage.taxonomy import FailureContext, FailureType

# A strategy is any async callable that receives a FailureContext
# and returns a RecoveryAction.
StrategyFn = Callable[[FailureContext], Awaitable["RecoveryAction"]]


class RecoveryAction:
    """Returned by every strategy to tell the Agent wrapper what to do next."""

    @classmethod
    def RETRY(
        cls,
        hint: str | None = None,
        inject: dict[str, Any] | None = None,
        delay: float = 0.0,
    ) -> "RecoveryAction":
        """Re-run from the critical step, optionally injecting context."""
        return cls("retry", hint=hint, inject=inject, delay=delay if delay else None)

    @classmethod
    def REPLAN(cls, hint: str | None = None) -> "RecoveryAction":
        """Abort the current plan branch and generate a new plan from scratch."""
        return cls("replan", hint=hint)

    @classmethod
    def ROLLBACK(cls, checkpoint_id: str | None = None) -> "RecoveryAction":
        """Restore state to a named checkpoint and re-run from there."""
        return cls("rollback", checkpoint_id=checkpoint_id)

    @classmethod
    def RESUME(cls, from_subgoal: str | None = None) -> "RecoveryAction":
        """Continue execution from an incomplete sub-goal."""
        return cls("resume", from_subgoal=from_subgoal)

    @classmethod
    def ESCALATE(cls, message: str | None = None) -> "RecoveryAction":
        """Surface to a human. Halt autonomous execution."""
        return cls("escalate", message=message)

    @classmethod
    def ABORT(cls, reason: str | None = None) -> "RecoveryAction":
        """Hard stop. No recovery attempted."""
        return cls("abort", reason=reason)

    def __init__(self, kind: str, **kwargs: Any) -> None:
        self.kind = kind
        self.params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"RecoveryAction.{self.kind.upper()}({params})"


@dataclass
class FailurePolicy:
    """Maps each FailureType to a recovery strategy callable.

    Usage::

        from triage import FailurePolicy
        from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
        from triage.strategies.replan import replan, resume_from_subgoal
        from triage.strategies.rollback import rollback_to_checkpoint

        policy = FailurePolicy(
            WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
            CONSTRAINT_IGNORED = replan(hint="re-read the constraints carefully"),
            LOOP_DETECTED      = replan(max_replans=2),
            HALLUCINATED_STATE = rollback_to_checkpoint(),
            PLAN_INCOMPLETE    = resume_from_subgoal(),
            EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
            default            = FailurePolicy.escalate_by_default(),
        )

    Any FailureType not explicitly listed falls through to ``default``.
    """

    WRONG_TOOL_CALLED: StrategyFn | None = None
    CONSTRAINT_IGNORED: StrategyFn | None = None
    LOOP_DETECTED: StrategyFn | None = None
    HALLUCINATED_STATE: StrategyFn | None = None
    PLAN_INCOMPLETE: StrategyFn | None = None
    SCHEMA_MISMATCH: StrategyFn | None = None
    CONTEXT_OVERFLOW: StrategyFn | None = None
    GOAL_DRIFT: StrategyFn | None = None
    EXTERNAL_FAULT: StrategyFn | None = None
    UNKNOWN: StrategyFn | None = None
    default: StrategyFn | None = None

    _FIELD_MAP: dict[FailureType, str] = field(init=False, repr=False, default_factory=lambda: {
        FailureType.WRONG_TOOL_CALLED:  "WRONG_TOOL_CALLED",
        FailureType.CONSTRAINT_IGNORED: "CONSTRAINT_IGNORED",
        FailureType.LOOP_DETECTED:      "LOOP_DETECTED",
        FailureType.HALLUCINATED_STATE: "HALLUCINATED_STATE",
        FailureType.PLAN_INCOMPLETE:    "PLAN_INCOMPLETE",
        FailureType.SCHEMA_MISMATCH:    "SCHEMA_MISMATCH",
        FailureType.CONTEXT_OVERFLOW:   "CONTEXT_OVERFLOW",
        FailureType.GOAL_DRIFT:         "GOAL_DRIFT",
        FailureType.EXTERNAL_FAULT:     "EXTERNAL_FAULT",
        FailureType.UNKNOWN:            "UNKNOWN",
    })

    def resolve(self, failure_type: FailureType) -> StrategyFn | None:
        """Return the strategy for a given failure type, falling back to default."""
        field_name = self._FIELD_MAP.get(failure_type)
        if field_name:
            strategy = getattr(self, field_name, None)
            if strategy is not None:
                return strategy
        return self.default

    async def dispatch(self, ctx: FailureContext) -> RecoveryAction:
        """Dispatch to the appropriate strategy for ctx.failure_type.

        Falls back to ESCALATE if no strategy is registered.
        """
        strategy = self.resolve(ctx.failure_type)
        if strategy is None:
            return RecoveryAction.ESCALATE(
                message=f"No strategy registered for {ctx.failure_type.value}. Manual review required."
            )
        return await strategy(ctx)

    @staticmethod
    def chain(primary: StrategyFn, fallback: StrategyFn, after_kinds: tuple[str, ...] = ("escalate",)) -> StrategyFn:
        """Return a strategy that tries ``primary`` and falls through to ``fallback``
        when ``primary`` returns an action whose kind is in ``after_kinds``.

        The default ``after_kinds=("escalate",)`` means: use primary normally, but
        if primary decides to escalate, try fallback first instead.

        Example — replan first, rollback if replan has already been tried::

            from triage.strategies.replan import replan
            from triage.strategies.rollback import rollback_to_checkpoint

            policy = FailurePolicy(
                LOOP_DETECTED=FailurePolicy.chain(
                    replan(hint="Try a different approach."),
                    rollback_to_checkpoint(),
                    after_kinds=("escalate",),
                ),
            )

        Example — retry up to 2 times, then replan::

            policy = FailurePolicy(
                EXTERNAL_FAULT=FailurePolicy.chain(
                    backoff_and_retry(max_attempts=2),
                    replan(hint="External service is down, try a different approach."),
                    after_kinds=("escalate",),
                ),
            )
        """
        async def _chained(ctx: FailureContext) -> RecoveryAction:
            action = await primary(ctx)
            if action.kind in after_kinds:
                return await fallback(ctx)
            return action
        return _chained

    @staticmethod
    def escalate_by_default() -> StrategyFn:
        """Default strategy: always escalate to human."""
        async def _escalate(ctx: FailureContext) -> RecoveryAction:
            return RecoveryAction.ESCALATE(
                message=f"Unhandled failure: {ctx.failure_type.value} at step {ctx.critical_step_index}"
            )
        return _escalate

    @staticmethod
    def abort_by_default() -> StrategyFn:
        """Default strategy: hard abort on any unhandled failure."""
        async def _abort(ctx: FailureContext) -> RecoveryAction:
            return RecoveryAction.ABORT(reason=ctx.failure_type.value)
        return _abort
