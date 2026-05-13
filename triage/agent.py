"""
triage.agent
~~~~~~~~~~~~
Agent wrapper. Runs the async agent loop, records steps, classifies
failures, dispatches recovery actions, and executes them.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Coroutine

import anyio

from triage.checkpoint import CheckpointStore, InMemoryCheckpointStore, make_checkpoint
from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureContext, Step, TriageContext
from triage.trajectory import Trajectory


def _safe_hook(fn: Callable[..., None], *args: Any) -> None:
    """Call a lifecycle hook, swallowing exceptions so hooks never break a run."""
    try:
        fn(*args)
    except Exception as exc:
        logger.warning("[triage] hook raised: %s", exc)

logger = logging.getLogger("triage")

# ── contextvars for zero-signature-change injection ───────────────────────────

_record_step_var: contextvars.ContextVar[Callable[[Step], None] | None] = \
    contextvars.ContextVar("triage_record_step", default=None)

_update_state_var: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = \
    contextvars.ContextVar("triage_update_state", default=None)


def get_recorder() -> Callable[[Step], None]:
    """Return the ``record_step`` callback for the current triage run.

    For use by agents that cannot or do not want to accept ``record_step``
    as a keyword argument::

        from triage.agent import get_recorder
        from triage.taxonomy import Step

        async def my_agent(task: str, **kwargs) -> str:
            record = get_recorder()
            record(Step(index=0, action="fetch", tool_output=data))
            return result

    Raises ``RuntimeError`` if called outside a triage ``Agent.run()`` context.
    """
    fn = _record_step_var.get()
    if fn is None:
        raise RuntimeError(
            "get_recorder() called outside a triage Agent.run() context. "
            "Either call it from within a wrapped agent, or accept record_step "
            "as a keyword argument instead."
        )
    return fn


def get_state_updater() -> Callable[[dict[str, Any]], None]:
    """Return the ``update_state`` callback for the current triage run.

    Raises ``RuntimeError`` if called outside a triage ``Agent.run()`` context.
    """
    fn = _update_state_var.get()
    if fn is None:
        raise RuntimeError(
            "get_state_updater() called outside a triage Agent.run() context."
        )
    return fn


# ── Exceptions ────────────────────────────────────────────────────────────────

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


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """Wraps any async callable and adds failure classification + recovery.

    The wrapped function must accept ``record_step`` and optionally
    ``update_state`` keyword arguments::

        async def my_agent(task: str, *, record_step, update_state, **kwargs) -> Any:
            data = fetch_something()
            record_step(Step(index=0, action="fetch", tool_output=data))
            update_state({"data": data})   # persisted into checkpoints
            return result

    Alternatively, use ``triage.agent.get_recorder()`` inside the agent body
    to avoid changing the function signature.

    Parameters
    ----------
    fn:
        The async agent callable to wrap.
    policy:
        Maps each ``FailureType`` to a recovery strategy.
    classifier:
        Classifies failures from the trajectory. Defaults to ``RulesClassifier``.
    checkpoint_store:
        Stores and loads checkpoints. Defaults to ``InMemoryCheckpointStore``.
    max_recovery_attempts:
        Maximum number of recovery loop iterations per ``run()`` call (default 3).
        Each failed attempt + dispatch counts as one iteration.
    max_total_attempts:
        Hard cap on ``len(attempt_history)`` across all failure types. When
        reached, triage escalates regardless of the active strategy. ``None``
        (default) disables this cap and defers entirely to ``max_recovery_attempts``.
        Use this to bound total retries when your policy has multiple failure types
        that could alternate and extend the loop beyond intent.
    auto_checkpoint:
        If ``True``, saves a checkpoint after every ``record_step()`` call.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        policy: FailurePolicy,
        classifier: Classifier | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_recovery_attempts: int = 3,
        max_total_attempts: int | None = None,
        auto_checkpoint: bool = False,
        on_step: Callable[[Step], None] | None = None,
        on_failure: Callable[[FailureContext], None] | None = None,
        on_recovery: Callable[[FailureContext, RecoveryAction], None] | None = None,
    ) -> None:
        self._fn = fn
        self._policy = policy
        self._classifier: Classifier = classifier or RulesClassifier()
        self._checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._max_recovery_attempts = max_recovery_attempts
        self._max_total_attempts = max_total_attempts
        self._auto_checkpoint = auto_checkpoint
        self._on_step = on_step
        self._on_failure = on_failure
        self._on_recovery = on_recovery

        # mutable run-state (reset on each run() call)
        self._trajectory: Trajectory = Trajectory()
        self._current_state: dict[str, Any] = {}
        self._last_checkpoint_id: str | None = None
        self._pending_checkpoints: list[Coroutine[Any, Any, None]] = []

    def clone(self) -> "Agent":
        """Return a new Agent sharing the same policy, classifier, and checkpoint
        store but with fresh per-run state.

        Use this to run multiple tasks concurrently — a single Agent instance is
        not safe for concurrent ``run()`` calls::

            agents = [agent.clone() for _ in tasks]
            results = await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks)])
        """
        return Agent(
            fn=self._fn,
            policy=self._policy,
            classifier=self._classifier,
            checkpoint_store=self._checkpoint_store,
            max_recovery_attempts=self._max_recovery_attempts,
            max_total_attempts=self._max_total_attempts,
            auto_checkpoint=self._auto_checkpoint,
            on_step=self._on_step,
            on_failure=self._on_failure,
            on_recovery=self._on_recovery,
        )

    async def run(self, task: str, **kwargs: Any) -> Any:
        """Run the wrapped agent, recovering from failures per the policy."""
        attempt = 0
        attempt_history: list[tuple[Any, str]] = []

        # Set contextvars so get_recorder() / get_state_updater() work inside fn
        rec_token = _record_step_var.set(self._record_step)
        upd_token = _update_state_var.set(self._update_state)
        try:
            return await self._run_loop(task, attempt, attempt_history, kwargs)
        finally:
            _record_step_var.reset(rec_token)
            _update_state_var.reset(upd_token)

    async def _run_loop(
        self,
        task: str,
        attempt: int,
        attempt_history: list[tuple[Any, str]],
        kwargs: dict[str, Any],
    ) -> Any:
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

            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise  # CancelledError, KeyboardInterrupt, GeneratorExit — never recover
                await self._drain_checkpoints()

                # If the agent raised before calling record_step(), synthesize a
                # sentinel step from the exception so the classifier has context.
                if not self._trajectory.steps:
                    self._trajectory.append(Step(
                        index=0,
                        action="<no steps recorded>",
                        error=str(exc),
                    ))

                steps = self._trajectory.steps

                # Run classify() in a thread — LLMClassifier blocks ~100-400ms.
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
                if self._on_failure:
                    _safe_hook(self._on_failure, ctx)

                # Check both per-loop cap and cross-type global cap
                total_exceeded = (
                    self._max_total_attempts is not None
                    and len(attempt_history) >= self._max_total_attempts
                )
                if attempt >= self._max_recovery_attempts or total_exceeded:
                    cap = (
                        f"max_total_attempts ({self._max_total_attempts})"
                        if total_exceeded
                        else f"max_recovery_attempts ({self._max_recovery_attempts})"
                    )
                    raise TriageEscalationError(
                        f"{cap} exceeded after {failure_type.value}.",
                        ctx,
                    ) from exc

                action = await self._policy.dispatch(ctx)
                logger.info("[triage] Dispatching: %r", action)
                if self._on_recovery:
                    _safe_hook(self._on_recovery, ctx, action)
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

        # Inject structured TriageContext alongside the legacy scalar kwargs.
        # Agents can use either form; both are always present after a failure.
        new_kwargs["_triage_context"] = TriageContext(
            failure_type=ctx.failure_type,
            attempt_number=ctx.metadata["attempt_number"],
            hint=new_kwargs.get("_triage_hint"),
            subgoal=new_kwargs.get("_triage_subgoal"),
            state=dict(new_kwargs.get("_triage_state", {})),
        )

        logger.info("[triage] Attempt %d...", ctx.metadata["attempt_number"] + 1)
        return new_kwargs

    def _record_step(self, step: Step) -> None:
        self._trajectory.append(step)
        if self._on_step:
            _safe_hook(self._on_step, step)
        if self._auto_checkpoint:
            self._pending_checkpoints.append(self._save_auto_checkpoint())

    def _update_state(self, state: dict[str, Any]) -> None:
        self._current_state = dict(state)

    async def _drain_checkpoints(self) -> None:
        pending, self._pending_checkpoints = self._pending_checkpoints, []
        for coro in pending:
            try:
                await coro
            except Exception as exc:
                logger.warning("[triage] auto_checkpoint failed: %s", exc)

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
