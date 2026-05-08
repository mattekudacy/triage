"""
triage.agent
~~~~~~~~~~~~
Agent wrapper. Runs the async agent loop, records steps, classifies
failures, dispatches recovery actions, and executes them.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Coroutine

import anyio

from triage.checkpoint import CheckpointStore, InMemoryCheckpointStore, make_checkpoint
from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureContext, Step
from triage.trajectory import Trajectory

logger = logging.getLogger("triage")


class TriageEscalationError(Exception):
    """Raised when a strategy returns RecoveryAction.ESCALATE or max attempts exceeded."""

    def __init__(self, message: str, context: FailureContext) -> None:
        super().__init__(message)
        self.context = context


class TriageAbortError(Exception):
    """Raised when a strategy returns RecoveryAction.ABORT."""

    def __init__(self, reason: str, context: FailureContext) -> None:
        super().__init__(reason)
        self.context = context


class Agent:
    """Wraps any async callable and adds failure classification + recovery.

    The wrapped function must accept ``record_step`` and optionally
    ``update_state`` keyword arguments::

        async def my_agent(task: str, *, record_step, update_state, **kwargs) -> Any:
            data = fetch_something()
            record_step(Step(index=0, action="fetch", tool_output=data))
            update_state({"data": data})   # persisted into checkpoints
            return result

    ``update_state`` is optional — agents that don't call it get ``state={}``
    in their checkpoints, and rollback will not inject ``_triage_state``.

    If ``auto_checkpoint=True``, a checkpoint (including current state) is
    saved after each ``record_step`` call.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        policy: FailurePolicy,
        classifier: Classifier | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_recovery_attempts: int = 3,
        auto_checkpoint: bool = False,
    ) -> None:
        self._fn = fn
        self._policy = policy
        self._classifier: Classifier = classifier or RulesClassifier()
        self._checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._max_recovery_attempts = max_recovery_attempts
        self._auto_checkpoint = auto_checkpoint

        # mutable run-state (reset on each run() call)
        self._trajectory: Trajectory = Trajectory()
        self._current_state: dict[str, Any] = {}
        self._last_checkpoint_id: str | None = None
        self._pending_checkpoints: list[Coroutine[Any, Any, None]] = []

    async def run(self, task: str, **kwargs: Any) -> Any:
        """Run the wrapped agent, recovering from failures per the policy."""
        attempt = 0
        attempt_history: list[tuple[Any, str]] = []

        while True:
            self._trajectory = Trajectory()
            self._current_state = {}
            self._pending_checkpoints = []

            try:
                result = await self._fn(
                    task,
                    record_step=self._record_step,
                    update_state=self._update_state,
                    **kwargs,
                )
                await self._drain_checkpoints()
                return result

            except (TriageEscalationError, TriageAbortError):
                await self._drain_checkpoints()
                raise

            except Exception as exc:
                await self._drain_checkpoints()
                steps = self._trajectory.steps

                # Run classify() in a thread — LLMClassifier blocks ~100-400ms.
                # RulesClassifier is fast but the thread overhead is negligible on
                # the failure path. Keeps the event loop unblocked for both.
                failure_type = await anyio.to_thread.run_sync(
                    self._classifier.classify, self._trajectory, task
                )

                ctx = FailureContext(
                    failure_type=failure_type,
                    trajectory=steps,
                    critical_step_index=max(len(steps) - 1, 0),
                    original_task=task,
                    last_checkpoint_id=self._last_checkpoint_id,
                    raw_error=exc,
                    metadata={"attempt_number": attempt},
                    attempt_history=list(attempt_history),
                )

                logger.info("[triage] %s detected at step %d", failure_type.value, ctx.critical_step_index)

                if attempt >= self._max_recovery_attempts:
                    raise TriageEscalationError(
                        f"Max recovery attempts ({self._max_recovery_attempts}) exceeded "
                        f"after {failure_type.value}.",
                        ctx,
                    ) from exc

                action = await self._policy.dispatch(ctx)
                logger.info("[triage] Dispatching: %r", action)
                attempt_history.append((failure_type, action.kind))
                attempt += 1

                kwargs = await self._execute_action(action, ctx, task, kwargs)

    async def _execute_action(
        self,
        action: RecoveryAction,
        ctx: FailureContext,
        task: str,  # noqa: ARG002
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Mutate kwargs and/or state based on the recovery action. Returns new kwargs."""
        new_kwargs = dict(kwargs)

        if action.kind == "retry":
            delay = action.params.get("delay", 0.0)
            if delay:
                logger.info("[triage] Backing off %.1fs before retry (attempt %d)", delay, ctx.metadata["attempt_number"] + 1)
                await anyio.sleep(delay)
            if "hint" in action.params:
                new_kwargs["_triage_hint"] = action.params["hint"]

        elif action.kind == "replan":
            new_kwargs["_triage_hint"] = action.params.get("hint", "Generate a new plan.")

        elif action.kind == "rollback":
            checkpoint_id = action.params.get("checkpoint_id")
            if checkpoint_id:
                checkpoint = await self._checkpoint_store.load(checkpoint_id)
            else:
                checkpoint = await self._checkpoint_store.latest()
            if checkpoint is None:
                raise TriageEscalationError(
                    "ROLLBACK requested but no checkpoint is available.", ctx
                )
            self._trajectory = Trajectory.from_steps(checkpoint.trajectory_snapshot)
            self._last_checkpoint_id = checkpoint.id
            self._current_state = dict(checkpoint.state)
            new_kwargs["_triage_hint"] = f"Rolled back to checkpoint {checkpoint.id!r}."
            if checkpoint.state:
                new_kwargs["_triage_state"] = checkpoint.state

        elif action.kind == "resume":
            subgoal = action.params.get("from_subgoal")
            if subgoal:
                new_kwargs["_triage_subgoal"] = subgoal

        elif action.kind == "escalate":
            raise TriageEscalationError(
                action.params.get("message", "Escalated by policy."), ctx
            )

        elif action.kind == "abort":
            raise TriageAbortError(
                action.params.get("reason", "Aborted by policy."), ctx
            )

        logger.info("[triage] Attempt %d...", ctx.metadata["attempt_number"] + 1)
        return new_kwargs

    def _record_step(self, step: Step) -> None:
        self._trajectory.append(step)
        if self._auto_checkpoint:
            self._pending_checkpoints.append(self._save_auto_checkpoint())

    def _update_state(self, state: dict[str, Any]) -> None:
        self._current_state = dict(state)

    async def _drain_checkpoints(self) -> None:
        for coro in self._pending_checkpoints:
            try:
                await coro
            except Exception as exc:
                logger.warning("[triage] auto_checkpoint failed: %s", exc)
        self._pending_checkpoints = []

    async def _save_auto_checkpoint(self) -> None:
        checkpoint = make_checkpoint(
            state=self._current_state,
            trajectory_steps=self._trajectory.steps,
        )
        await self._checkpoint_store.save(checkpoint)
        self._last_checkpoint_id = checkpoint.id

    async def __call__(self, task: str, **kwargs: Any) -> Any:
        return await self.run(task, **kwargs)


def agent(policy: FailurePolicy, **kwargs: Any) -> Callable[[Callable[..., Awaitable[Any]], Agent]]:
    """Decorator factory. Wraps an async function with triage recovery.

    Usage::

        @triage.agent(policy=my_policy)
        async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
            ...
    """
    def decorator(fn: Callable[..., Awaitable[Any]]) -> Agent:
        return Agent(fn, policy=policy, **kwargs)
    return decorator
