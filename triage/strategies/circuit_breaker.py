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
the inner strategy is called and the result determines the breaker outcome:

- Inner returns a recoverable action (retry, replan, rollback, resume):
  ``record_failure()`` is called — the failure is counted toward the threshold.
  In HALF_OPEN this re-opens the breaker (probe attempt was not enough to recover).
- Inner returns escalate or abort:
  ``record_failure()`` is called — same as above.

To close the breaker after a fully successful run (no failures at all), call
``breaker.record_success()`` after ``agent.run()`` returns. The recommended
pattern via ``on_recovery``::

    def on_recovery(ctx, action):
        if action.kind not in ("escalate", "abort"):
            breaker.record_success()   # recovery action dispatched — probe succeeded

    agent = Agent(my_fn, policy=policy, on_recovery=on_recovery)

Or call it unconditionally after a clean ``agent.run()`` if you manage
success signalling at the call site.

HALF_OPEN single-probe guarantee
---------------------------------
``CircuitBreaker.allow_request()`` sets a ``_probe_in_flight`` flag atomically
when transitioning to HALF_OPEN. Concurrent callers that call ``allow_request()``
while a probe is in flight receive ``False`` and are blocked, ensuring exactly
one probe attempt reaches ``inner`` at a time. The flag is cleared by either
``record_failure()`` or ``record_success()``.
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

        # Breaker is CLOSED or HALF_OPEN — call inner and record the outcome.
        # Every failure that reaches the inner strategy counts against the
        # threshold (strategies are only ever called after a failure).
        # record_failure() also clears the _probe_in_flight flag set by
        # allow_request() in HALF_OPEN, so the next probe slot opens once
        # the cooldown elapses again.
        action = await inner(ctx)
        breaker.record_failure()
        return action

    return _strategy
