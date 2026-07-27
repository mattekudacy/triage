"""
triage.strategies.saga
~~~~~~~~~~~~~~~~~~~~~~
Compensating rollback strategy: runs registered compensators in reverse
step-index order before restoring a checkpoint.

Usage::

    from triage.strategies.saga import compensating_rollback

    async def my_agent(task, *, record_step, record_compensator, **kwargs):
        charge_id = await charge_card(amount)
        record_step(Step(index=1, action="charge_card"))
        record_compensator(1, lambda: refund_card(charge_id))

        await send_email(user)
        record_step(Step(index=2, action="send_email"))
        record_compensator(2, lambda: cancel_email(msg_id))
        ...

    policy = FailurePolicy(
        EXTERNAL_FAULT=compensating_rollback(),
    )

When the agent fails and ROLLBACK fires, triage calls ``cancel_email`` then
``refund_card`` (step 2 first, then step 1) before restoring the checkpoint.
"""

from __future__ import annotations

from triage.policy import RecoveryAction, StrategyFn
from triage.taxonomy import FailureContext


def compensating_rollback(checkpoint_id: str | None = None) -> StrategyFn:
    """Roll back to a checkpoint, running registered compensators first.

    Compensators are registered during the run via the ``record_compensator``
    injected kwarg or ``triage.agent.get_compensator_recorder()``::

        async def my_agent(task, *, record_step, record_compensator, **kw):
            charge_id = await charge_card(amount)
            record_step(Step(index=1, action="charge_card"))
            record_compensator(1, lambda: refund_card(charge_id))

    When a ROLLBACK action fires, triage calls each compensator in **reverse
    step-index order** before restoring the checkpoint.  Compensator errors
    are logged (``triage_event: "compensator_error"``) but never abort recovery.

    Parameters
    ----------
    checkpoint_id:
        Specific checkpoint to restore.  ``None`` (default) uses the latest
        checkpoint for the current run — the same default as ``ROLLBACK()``.
    """

    async def _strategy(ctx: FailureContext) -> RecoveryAction:
        return RecoveryAction.ROLLBACK(
            checkpoint_id=checkpoint_id or ctx.last_checkpoint_id,
        )

    return _strategy
