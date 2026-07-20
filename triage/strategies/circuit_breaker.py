"""
triage.strategies.circuit_breaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Strategy factory that wraps another strategy with a CircuitBreaker guard.

Usage::

    from triage.breaker import CircuitBreaker
    from triage.strategies.circuit_breaker import circuit_breaker
    from triage.strategies.retry import backoff_and_retry

    breaker = CircuitBreaker(failure_threshold=5, window_seconds=60, cooldown_seconds=30)

    policy = FailurePolicy(
        EXTERNAL_FAULT=circuit_breaker(breaker, backoff_and_retry(max_attempts=3)),
    )

When the breaker is OPEN, the inner strategy is never called — the run is
immediately escalated with a "circuit open" message. When CLOSED or HALF_OPEN,
the inner strategy is called normally.

The breaker records failures and successes automatically:
- ``record_failure()`` is called whenever the inner strategy is invoked (i.e.
  whenever ``Agent`` dispatches a recovery action for this failure type).
- ``record_success()`` is called via the optional ``on_recovery`` hook wiring
  described below — or you can call it manually when your agent completes a
  successful run without failures.

Because a strategy is only called *on failure*, successes must be signalled
separately. The recommended approach::

    from triage.agent import Agent

    agent = Agent(
        my_fn,
        policy=policy,
        on_recovery=lambda ctx, action: None,   # handled by circuit_breaker wrapper
    )

    # After a successful run, close the breaker:
    breaker.record_success()

Or wire it into ``on_recovery`` to reset on any non-failure action::

    def on_recovery(ctx, action):
        if action.kind not in ("escalate", "abort"):
            breaker.record_success()

    agent = Agent(my_fn, policy=policy, on_recovery=on_recovery)
"""

from __future__ import annotations

from triage.breaker import CircuitBreaker
from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def circuit_breaker(
    breaker: CircuitBreaker,
    inner: StrategyFn,
    *,
    open_action: str = "escalate",
) -> StrategyFn:
    """Wrap ``inner`` with a circuit breaker guard.

    Parameters
    ----------
    breaker:
        Shared ``CircuitBreaker`` instance. Must be the same object across all
        agents and runs that should share the failure-rate window.
    inner:
        The strategy to call when the breaker is CLOSED or HALF_OPEN.
    open_action:
        What to do when the breaker is OPEN. ``"escalate"`` (default) raises
        ``TriageEscalationError``; ``"abort"`` raises ``TriageAbortError``.
    """
    if open_action not in ("escalate", "abort"):
        raise ValueError("open_action must be 'escalate' or 'abort'")

    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        if not breaker.allow_request():
            msg = (
                f"Circuit breaker is OPEN for {ctx.failure_type.value}. "
                f"Blocking recovery until cooldown expires "
                f"({breaker.cooldown_seconds}s)."
            )
            if open_action == "abort":
                return RecoveryAction.ABORT(reason=msg)
            return RecoveryAction.ESCALATE(message=msg)

        # Breaker is CLOSED or HALF_OPEN — record this failure and let the
        # inner strategy decide the recovery action.
        breaker.record_failure()
        return await inner(ctx)

    return _strategy
