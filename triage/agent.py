"""
triage.agent
~~~~~~~~~~~~
Agent wrapper. Runs the async agent loop, records steps, classifies
failures, dispatches recovery actions, and executes them.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

if TYPE_CHECKING:
    from triage.breaker import CircuitBreaker
    from triage.scorer.base import StepRiskScorer

from triage.checkpoint import Checkpoint, CheckpointStore, InMemoryCheckpointStore, make_checkpoint
from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier
from triage.observability.metrics import record_failure as metrics_record_failure
from triage.observability.metrics import record_recovery as metrics_record_recovery
from triage.observability.metrics import record_recovery_end as metrics_record_recovery_end
from triage.observability.metrics import record_run_end as metrics_record_run_end
from triage.observability.metrics import resolve_meter
from triage.observability.otel import (
    classify_span,
    dispatch_span,
    resolve_tracer,
    run_span,
    set_span_classify_result,
    set_span_dispatch_result,
    set_span_run_outcome,
)
from triage.policy import FailurePolicy, RecoveryAction
from triage.suspension import InMemorySuspensionStore, SuspendedRun, SuspensionStore, make_token
from triage.taxonomy import FailureContext, FailureType, Step, TriageContext
from triage.trajectory import Trajectory
from triage.usage import Usage, UsageMeter


def _safe_hook(fn: Callable[..., None], *args: Any) -> None:
    """Call a lifecycle hook, swallowing exceptions so hooks never break a run."""
    try:
        fn(*args)
    except Exception as exc:
        logger.warning(
            "[triage] hook error",
            extra={"triage_event": "hook_error", "error": str(exc)},
        )

logger = logging.getLogger("triage")

# ── contextvars for zero-signature-change injection ───────────────────────────

_record_step_var: contextvars.ContextVar[Callable[[Step], None] | None] = \
    contextvars.ContextVar("triage_record_step", default=None)

_update_state_var: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = \
    contextvars.ContextVar("triage_update_state", default=None)

_record_usage_var: contextvars.ContextVar[Callable[[Usage], None] | None] = \
    contextvars.ContextVar("triage_record_usage", default=None)


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


def get_usage_recorder() -> Callable[[Usage], None]:
    """Return the ``record_usage`` callback for the current triage run.

    Call this inside a wrapped agent to report token and cost usage::

        from triage.agent import get_usage_recorder
        from triage.usage import Usage

        async def my_agent(task: str, *, record_step, **kwargs) -> str:
            result = await call_llm(prompt)
            get_usage_recorder()(Usage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ))
            return result.content

    Raises ``RuntimeError`` if called outside a triage ``Agent.run()`` context.
    """
    fn = _record_usage_var.get()
    if fn is None:
        raise RuntimeError(
            "get_usage_recorder() called outside a triage Agent.run() context."
        )
    return fn


@dataclass
class _RunState:
    """Per-task run-state for one Agent instance.

    Held behind a ContextVar (not a plain instance attribute) so that
    concurrent ``run()`` calls on the same Agent — each executing in its own
    asyncio Task, hence its own copy-on-write context — never see each
    other's trajectory, state, or checkpoint bookkeeping.
    """

    trajectory: Trajectory = field(default_factory=Trajectory)
    current_state: dict[str, Any] = field(default_factory=dict)
    last_checkpoint_id: str | None = None
    pending_checkpoints: list[Checkpoint] = field(default_factory=list)
    last_ctx: FailureContext | None = None
    run_id: str | None = None
    meter: UsageMeter = field(default_factory=UsageMeter)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TriageEscalationError(Exception):
    """Raised when a strategy returns RecoveryAction.ESCALATE or max attempts exceeded."""

    def __init__(self, message: str, context: FailureContext) -> None:
        super().__init__(message)
        self.context = context


class TriageAbortError(Exception):
    """Raised when a strategy returns RecoveryAction.ABORT."""

    def __init__(self, reason: str, context: FailureContext | None) -> None:
        super().__init__(reason)
        self.context = context


class TriageSuspendedError(Exception):
    """Raised when a strategy returns RecoveryAction.SUSPEND().

    The run is paused; call ``agent.resume(token, action=...)`` to continue.
    """

    def __init__(self, token: str, run: SuspendedRun) -> None:
        super().__init__(
            f"Run suspended (token={token!r}). "
            f"Call agent.resume({token!r}, action=...) to continue."
        )
        self.token = token
        self.run = run


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

    Synchronous callables are also accepted — triage runs them via
    ``anyio.to_thread.run_sync()`` so the event loop is never blocked::

        def my_sync_agent(task: str, *, record_step, **kwargs) -> Any:
            ...

    Alternatively, use ``triage.agent.get_recorder()`` inside the agent body
    to avoid changing the function signature.

    Parameters
    ----------
    fn:
        The agent callable to wrap — either ``async def`` or plain ``def``.
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
    max_recovery_seconds:
        Wall-clock budget for the entire recovery process. If recovery has been
        ongoing for more than this many seconds, triage escalates. ``None``
        (default) disables this cap. The timer starts after the first failure.
    auto_checkpoint:
        If ``True``, saves a checkpoint after every ``record_step()`` call.
    strict_idempotency:
        If ``True``, triage will escalate instead of retrying whenever any step
        in the trajectory has ``idempotent=False``. Default ``False``.
        Use this when your agent has steps that send emails, charge cards, or
        perform other non-reversible side effects.
    tracer:
        Optional OpenTelemetry ``Tracer`` instance. When provided, triage emits
        ``triage.run``, ``triage.classify``, and ``triage.dispatch`` spans for
        every ``run()`` call. When ``None`` (default), triage auto-detects: if
        ``opentelemetry-sdk`` is installed *and* a real tracer provider has been
        configured via ``trace.set_tracer_provider(...)``, the global tracer is
        used automatically. If OTel is not installed or no provider is configured,
        tracing is a silent no-op with zero overhead.
    meter:
        Optional OpenTelemetry ``Meter`` instance for recording metrics
        (``triage.runs``, ``triage.failures``, ``triage.recoveries``,
        ``triage.run.duration``, ``triage.recovery.attempts``). When ``None``
        (default), triage auto-detects a configured ``MeterProvider`` the same
        way ``tracer`` auto-detects a ``TracerProvider``. If OTel metrics is not
        installed or no provider is configured, metrics are a silent no-op.
    circuit_breakers:
        Optional list of ``CircuitBreaker`` instances to notify on every
        successful ``run()`` completion. After a clean return (no failures),
        triage calls ``breaker.record_success()`` on each breaker in order.
        This closes any breaker that was in HALF_OPEN state after a probe,
        without requiring manual success-signalling at the call site. Pass the
        same breaker instances that are wired into ``circuit_breaker()``
        strategy wrappers in the policy.
    suspension_store:
        Store for serializing paused runs when ``RecoveryAction.SUSPEND()``
        is returned.  Defaults to ``InMemorySuspensionStore`` (not durable
        across restarts).  Swap for a Redis-backed store in production so
        tokens survive process restarts.
    on_escalate:
        Optional async callback invoked just before ``TriageEscalationError``
        is raised. Signature::

            async def on_escalate(ctx: FailureContext) -> RecoveryAction | None:
                ...

        Return a ``RecoveryAction`` to override the escalation — triage will
        execute that action instead of raising. Return ``None`` (or omit a
        return) to proceed with the escalation as normal. This is the
        human-in-the-loop hook: pause autonomous execution, notify a human,
        wait for a decision, and return an action (or ``None`` to surface the
        error). Exceptions raised inside ``on_escalate`` are propagated directly
        (unlike lifecycle hooks such as ``on_failure``, which swallow errors).
    max_tokens:
        Maximum total tokens (input + output) allowed per ``run()`` call across
        all LLM calls recorded via ``record_usage()``. When exceeded, triage
        raises ``TriageEscalationError`` before the next recovery attempt.
        ``None`` (default) disables this cap.

        **Important:** enforcement is checked at each failure point, not
        preemptively. An agent that burns tokens but never raises will complete
        regardless of this cap. The check fires *after* the first failure —
        meaning up to one failure's worth of overage is always possible before
        triage intervenes. This is intentional: triage only intercepts at the
        failure boundary, so the cap is a "do not keep retrying after I'm
        already over budget" guard, not a hard token ceiling.
    max_cost_usd:
        Maximum total cost in USD allowed per ``run()`` call. Subject to the
        same failure-point enforcement as ``max_tokens`` above — not a
        preemptive cap. When exceeded, triage raises ``TriageEscalationError``.
        ``None`` (default) disables this cap. Use alongside
        ``record_usage(Usage(cost_usd=...))`` calls in the agent body or via
        the ``LLMClassifier`` auto-reporting.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        policy: FailurePolicy,
        classifier: Classifier | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_recovery_attempts: int = 3,
        max_total_attempts: int | None = None,
        max_recovery_seconds: float | None = None,
        auto_checkpoint: bool = False,
        on_step: Callable[[Step], None] | None = None,
        on_failure: Callable[[FailureContext], None] | None = None,
        on_recovery: Callable[[FailureContext, RecoveryAction], None] | None = None,
        strict_idempotency: bool = False,
        risk_scorer: StepRiskScorer | None = None,
        risk_threshold: float = 0.9,
        tracer: Any = None,
        meter: Any = None,
        circuit_breakers: list[CircuitBreaker] | None = None,
        on_escalate: Callable[[FailureContext], Awaitable[RecoveryAction | None]] | None = None,
        suspension_store: SuspensionStore | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self._fn = fn
        self._fn_is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        )
        self._policy = policy
        self._classifier: Classifier = classifier or RulesClassifier()
        self._checkpoint_store: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self._max_recovery_attempts = max_recovery_attempts
        self._max_total_attempts = max_total_attempts
        self._max_recovery_seconds = max_recovery_seconds
        self._auto_checkpoint = auto_checkpoint
        self._on_step = on_step
        self._on_failure = on_failure
        self._on_recovery = on_recovery
        self._strict_idempotency = strict_idempotency
        self._risk_scorer = risk_scorer
        self._risk_threshold = risk_threshold
        # resolve_tracer() returns explicit tracer if provided, else auto-detects
        # an active OTel provider, else None (no-op path).
        self._tracer = resolve_tracer(tracer)
        self._otel_meter = resolve_meter(meter)
        self._circuit_breakers: list[CircuitBreaker] = list(circuit_breakers or [])
        self._on_escalate = on_escalate
        self._suspension_store: SuspensionStore = (
            suspension_store or InMemorySuspensionStore()
        )
        self._max_tokens = max_tokens
        self._max_cost_usd = max_cost_usd

        # Per-task run-state, isolated via a ContextVar. asyncio/anyio copy the
        # current Context when spawning a Task, so concurrent run() calls on
        # this same Agent instance — each in its own Task — get independent
        # trajectory/state/checkpoint bookkeeping instead of clobbering a
        # shared instance attribute. See _RunState and the _run_state property.
        self._run_state_var: contextvars.ContextVar[_RunState] = contextvars.ContextVar(
            f"triage_run_state_{id(self)}"
        )

    @property
    def _run_state(self) -> _RunState:
        try:
            return self._run_state_var.get()
        except LookupError:
            state = _RunState()
            self._run_state_var.set(state)
            return state

    @property
    def _trajectory(self) -> Trajectory:
        return self._run_state.trajectory

    @_trajectory.setter
    def _trajectory(self, value: Trajectory) -> None:
        self._run_state.trajectory = value

    @property
    def _current_state(self) -> dict[str, Any]:
        return self._run_state.current_state

    @_current_state.setter
    def _current_state(self, value: dict[str, Any]) -> None:
        self._run_state.current_state = value

    @property
    def _last_checkpoint_id(self) -> str | None:
        return self._run_state.last_checkpoint_id

    @_last_checkpoint_id.setter
    def _last_checkpoint_id(self, value: str | None) -> None:
        self._run_state.last_checkpoint_id = value

    @property
    def _pending_checkpoints(self) -> list[Checkpoint]:
        return self._run_state.pending_checkpoints

    @_pending_checkpoints.setter
    def _pending_checkpoints(self, value: list[Checkpoint]) -> None:
        self._run_state.pending_checkpoints = value

    @property
    def _last_ctx(self) -> FailureContext | None:
        return self._run_state.last_ctx

    @_last_ctx.setter
    def _last_ctx(self, value: FailureContext | None) -> None:
        self._run_state.last_ctx = value

    @property
    def _run_id(self) -> str | None:
        return self._run_state.run_id

    @_run_id.setter
    def _run_id(self, value: str | None) -> None:
        self._run_state.run_id = value

    @property
    def _meter(self) -> UsageMeter:
        return self._run_state.meter

    def clone(self) -> Agent:
        """Return a new Agent sharing the same policy, classifier, and checkpoint
        store but with fresh per-run state, independent lifecycle hooks, and its
        own ``_last_ctx``.

        Concurrent ``run()`` calls on a single Agent instance are already safe
        (per-run state is isolated via contextvars) — use ``clone()`` when you
        want a task to have independent hooks or a dedicated checkpoint store
        instead::

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
            max_recovery_seconds=self._max_recovery_seconds,
            auto_checkpoint=self._auto_checkpoint,
            on_step=self._on_step,
            on_failure=self._on_failure,
            on_recovery=self._on_recovery,
            strict_idempotency=self._strict_idempotency,
            risk_scorer=self._risk_scorer,
            risk_threshold=self._risk_threshold,
            tracer=self._tracer,
            meter=self._otel_meter,
            circuit_breakers=list(self._circuit_breakers),
            on_escalate=self._on_escalate,
            suspension_store=self._suspension_store,
            max_tokens=self._max_tokens,
            max_cost_usd=self._max_cost_usd,
        )

    async def run(self, task: str, **kwargs: Any) -> Any:
        """Run the wrapped agent, recovering from failures per the policy."""
        attempt = 0
        attempt_history: list[tuple[Any, str]] = []

        # Assign a stable run ID once per run() call so all auto-checkpoints
        # from this run (including across recovery retries) share the same ID.
        # Stored in _RunState so concurrent run() calls get independent IDs.
        self._run_id = str(uuid.uuid4())

        # Reset usage meter so each run() starts with a clean slate.
        self._meter.reset()

        # Reset any per-run budget the classifier tracks (e.g. HybridClassifier's
        # max_llm_calls_per_run). Duck-typed — most classifiers don't define this.
        reset_call_count = getattr(self._classifier, "reset_call_count", None)
        if reset_call_count is not None:
            reset_call_count()

        # Set contextvars so get_recorder() / get_state_updater() / get_usage_recorder()
        # work inside fn without requiring signature changes.
        rec_token = _record_step_var.set(self._record_step)
        upd_token = _update_state_var.set(self._update_state)
        usg_token = _record_usage_var.set(self._record_usage)
        _run_start = time.monotonic()
        try:
            async with run_span(self._tracer, self._run_id, task) as _root_span:
                try:
                    result = await self._run_loop(task, attempt, attempt_history, kwargs)
                    set_span_run_outcome(_root_span)
                    # Notify circuit breakers that this run completed cleanly.
                    for breaker in self._circuit_breakers:
                        breaker.record_success()
                    metrics_record_run_end(
                        self._otel_meter,
                        outcome="success",
                        duration_s=time.monotonic() - _run_start,
                    )
                    return result
                except Exception as exc:
                    set_span_run_outcome(_root_span, error=exc)
                    metrics_record_run_end(
                        self._otel_meter,
                        outcome="error",
                        duration_s=time.monotonic() - _run_start,
                    )
                    raise
        finally:
            _record_step_var.reset(rec_token)
            _update_state_var.reset(upd_token)
            _record_usage_var.reset(usg_token)

    async def resume(self, token: str, *, action: RecoveryAction) -> Any:
        """Resume a previously suspended run.

        Loads the ``SuspendedRun`` from the suspension store, executes
        ``action`` as the human's decision, then continues the recovery loop
        from where it left off.

        Parameters
        ----------
        token:
            The token from ``TriageSuspendedError.token``.
        action:
            The ``RecoveryAction`` chosen by the human.  Common choices:

            - ``RecoveryAction.RETRY()`` — try again, possibly with a hint
            - ``RecoveryAction.REPLAN(hint="...")`` — generate a new plan
            - ``RecoveryAction.ABORT(reason="...")`` — give up permanently

        The suspended run is deleted from the store after a successful load,
        so tokens are single-use.

        Raises
        ------
        KeyError
            If ``token`` is not found in the suspension store.
        TriageSuspendedError
            If the resumed run is immediately suspended again by the policy.
        TriageEscalationError / TriageAbortError
            If the recovery loop exhausts attempts or the action is ABORT.
        """
        suspended = await self._suspension_store.load(token)
        await self._suspension_store.delete(token)

        ctx = suspended.context
        task = suspended.task
        attempt = suspended.attempt + 1
        attempt_history = list(suspended.attempt_history)

        # Re-establish contextvars so get_recorder() etc. work inside fn.
        rec_token = _record_step_var.set(self._record_step)
        upd_token = _update_state_var.set(self._update_state)
        usg_token = _record_usage_var.set(self._record_usage)

        # Restore run_id so any new checkpoints join the same scoped run.
        self._run_id = ctx.metadata.get("run_id") or str(uuid.uuid4())
        self._meter.reset()

        _run_start = time.monotonic()
        try:
            async with run_span(self._tracer, self._run_id, task) as _root_span:
                try:
                    # Execute the human-supplied action first, then continue
                    # the loop exactly as if this were a normal recovery attempt.
                    kwargs = await self._execute_action(
                        action, ctx, task, suspended.kwargs
                    )
                    result = await self._run_loop(
                        task, attempt, attempt_history, kwargs
                    )
                    set_span_run_outcome(_root_span)
                    for breaker in self._circuit_breakers:
                        breaker.record_success()
                    metrics_record_run_end(
                        self._otel_meter,
                        outcome="success",
                        duration_s=time.monotonic() - _run_start,
                    )
                    return result
                except Exception as exc:
                    set_span_run_outcome(_root_span, error=exc)
                    metrics_record_run_end(
                        self._otel_meter,
                        outcome="error",
                        duration_s=time.monotonic() - _run_start,
                    )
                    raise
        finally:
            _record_step_var.reset(rec_token)
            _update_state_var.reset(upd_token)
            _record_usage_var.reset(usg_token)

    async def _run_loop(
        self,
        task: str,
        attempt: int,
        attempt_history: list[tuple[Any, str]],
        kwargs: dict[str, Any],
    ) -> Any:
        _recovery_start: float | None = None

        while True:
            self._trajectory = Trajectory()
            self._current_state = {}
            self._pending_checkpoints = []

            try:
                if self._fn_is_async:
                    result = await self._fn(
                        task,
                        record_step=self._record_step,
                        update_state=self._update_state,
                        record_usage=self._record_usage,
                        **kwargs,
                    )
                else:
                    result = await anyio.to_thread.run_sync(
                        functools.partial(
                            self._fn,
                            task,
                            record_step=self._record_step,
                            update_state=self._update_state,
                            record_usage=self._record_usage,
                            **kwargs,
                        )
                    )
                await self._drain_checkpoints()
                return result

            except (TriageEscalationError, TriageAbortError, TriageSuspendedError):
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

                # If a child triage agent already classified this failure, reuse
                # its context type rather than re-classifying.
                _triage_exc = (TriageEscalationError, TriageAbortError)
                child_escalation = (
                    exc.__cause__ if isinstance(exc.__cause__, _triage_exc)
                    else exc.__context__ if isinstance(exc.__context__, _triage_exc)
                    else None
                )

                if child_escalation is not None and child_escalation.context is not None:
                    failure_type = child_escalation.context.failure_type
                else:
                    with classify_span(self._tracer, self._run_id) as _classify_span:
                        aclassify = getattr(self._classifier, "aclassify", None)
                        if aclassify is not None:
                            # Native async path — e.g. LLMClassifier/HybridClassifier
                            # awaiting an async SDK client directly, no thread hop.
                            failure_type = await aclassify(self._trajectory, task)
                        else:
                            # Run classify() in a thread — sync clients can block
                            # ~100-400ms and must never freeze the event loop.
                            failure_type = await anyio.to_thread.run_sync(
                                self._classifier.classify, self._trajectory, task
                            )
                        set_span_classify_result(_classify_span, failure_type.value)

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
                self._last_ctx = ctx

                logger.info(
                    "[triage] failure classified",
                    extra={
                        "triage_event": "failure_classified",
                        "failure_type": failure_type.value,
                        "step_index": ctx.critical_step_index,
                        "attempt": attempt,
                        "task": task,
                    },
                )
                metrics_record_failure(self._otel_meter, failure_type=failure_type.value)
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

                # Wall-clock recovery budget: start timing on first failure,
                # escalate on subsequent failures if elapsed time exceeds cap.
                if self._max_recovery_seconds is not None:
                    if _recovery_start is None:
                        _recovery_start = time.monotonic()
                    elif (time.monotonic() - _recovery_start) >= self._max_recovery_seconds:
                        raise TriageEscalationError(
                            f"max_recovery_seconds ({self._max_recovery_seconds}s) exceeded.",
                            ctx,
                        ) from exc

                # Token and cost budgets — checked once per failure, after the
                # first failure so the agent gets at least one attempt.
                if self._max_tokens is not None:
                    used = self._meter.total_tokens
                    if used >= self._max_tokens:
                        raise TriageEscalationError(
                            f"max_tokens ({self._max_tokens}) exceeded "
                            f"(used {used} tokens).",
                            ctx,
                        ) from exc

                if self._max_cost_usd is not None:
                    spent = self._meter.cost_usd
                    if spent >= self._max_cost_usd:
                        raise TriageEscalationError(
                            f"max_cost_usd ({self._max_cost_usd:.6f}) exceeded "
                            f"(spent ${spent:.6f}).",
                            ctx,
                        ) from exc

                with dispatch_span(self._tracer, self._run_id, attempt) as _dispatch_span:
                    action = await self._policy.dispatch(ctx)
                    # on_escalate: human-in-the-loop hook — called before raising
                    # TriageEscalationError. May return an override RecoveryAction
                    # or None to proceed with escalation.
                    if action.kind == "escalate" and self._on_escalate is not None:
                        override = await self._on_escalate(ctx)
                        if override is not None:
                            action = override
                    set_span_dispatch_result(_dispatch_span, action.kind, failure_type.value)
                metrics_record_recovery(
                    self._otel_meter,
                    failure_type=failure_type.value,
                    action_kind=action.kind,
                )
                # Decrement the in-progress counter when the run exits via
                # escalate or abort (the loop won't iterate again).
                if action.kind in ("escalate", "abort"):
                    metrics_record_recovery_end(
                        self._otel_meter, failure_type=failure_type.value
                    )
                logger.info(
                    "[triage] action dispatched",
                    extra={
                        "triage_event": "action_dispatched",
                        "action_kind": action.kind,
                        "failure_type": failure_type.value,
                        "attempt": attempt,
                    },
                )
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
                logger.info(
                    "[triage] retry backoff",
                    extra={
                        "triage_event": "retry_backoff",
                        "delay_s": delay,
                        "attempt": ctx.metadata["attempt_number"] + 1,
                    },
                )
                await anyio.sleep(delay)
            if "hint" in action.params:
                new_kwargs["_triage_hint"] = action.params["hint"]

            # Idempotency enforcement: escalate instead of retrying if any
            # executed step was non-idempotent and strict_idempotency is set.
            if self._strict_idempotency:
                non_idempotent = [s for s in ctx.trajectory if not s.idempotent]
                if non_idempotent:
                    names = ", ".join(
                        f"step[{s.index}] {s.action!r}" for s in non_idempotent
                    )
                    raise TriageEscalationError(
                        f"strict_idempotency: cannot retry — non-idempotent steps "
                        f"in trajectory: {names}",
                        ctx,
                    )

        elif action.kind == "replan":
            new_kwargs["_triage_hint"] = action.params.get("hint", "Generate a new plan.")

        elif action.kind == "rollback":
            checkpoint_id = action.params.get("checkpoint_id")
            loaded: Checkpoint | None
            if checkpoint_id:
                loaded = await self._checkpoint_store.load(checkpoint_id)
            else:
                loaded = await self._checkpoint_store.latest(self._run_id)
            if loaded is None:
                raise TriageEscalationError(
                    "ROLLBACK requested but no checkpoint is available.", ctx
                )
            checkpoint = loaded
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

        elif action.kind == "suspend":
            token = make_token()
            suspended = SuspendedRun(
                token=token,
                context=ctx,
                task=task,
                kwargs=dict(new_kwargs),
                attempt=ctx.metadata["attempt_number"],
                attempt_history=list(ctx.attempt_history),
                message=action.params.get("message", ""),
                metadata=action.params.get("metadata") or {},
            )
            await self._suspension_store.save(suspended)
            logger.info(
                "[triage] run suspended",
                extra={
                    "triage_event": "run_suspended",
                    "token": token,
                    "failure_type": ctx.failure_type.value,
                    "message": suspended.message,
                },
            )
            raise TriageSuspendedError(token, suspended)

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

        logger.info(
            "[triage] attempt start",
            extra={
                "triage_event": "attempt_start",
                "attempt": ctx.metadata["attempt_number"] + 1,
                "action_kind": action.kind,
            },
        )
        return new_kwargs

    def report_misclassification(
        self,
        expected_type: FailureType,
        *,
        store_path: str = "corrections.jsonl",
    ) -> None:
        """Record that the last classification was wrong.

        Call this after ``run()`` raises, passing the correct failure type::

            try:
                await agent.run("task")
            except TriageEscalationError:
                agent.report_misclassification(
                    FailureType.EXTERNAL_FAULT,
                    store_path="corrections.jsonl",
                )

        Appends a labeled entry to ``corrections.jsonl``. Use
        ``RulesClassifier.fit(path)`` to review coverage.
        """
        if self._last_ctx is None:
            raise RuntimeError(
                "No failure context available — call report_misclassification() "
                "after a failed run()."
            )
        from triage.feedback import record_correction
        from triage.taxonomy import FailureType  # local import avoids any circularity
        if not isinstance(expected_type, FailureType):
            raise TypeError(f"expected_type must be a FailureType, got {type(expected_type)!r}")
        record_correction(self._last_ctx, expected_type, store_path=store_path)

    def _record_step(self, step: Step) -> None:
        self._trajectory.append(step)
        if self._on_step:
            _safe_hook(self._on_step, step)
        if self._risk_scorer is not None:
            try:
                risk = self._risk_scorer(step, self._trajectory)
            except Exception as exc:
                logger.warning(
                    "[triage] risk scorer raised — skipping score check",
                    extra={"triage_event": "scorer_error", "error": str(exc)},
                )
            else:
                if risk.score >= self._risk_threshold:
                    raise TriageAbortError(
                        f"step risk score {risk.score:.2f} >= threshold {self._risk_threshold}"
                        + (f": {risk.reason}" if risk.reason else ""),
                        None,
                    )
        if self._auto_checkpoint:
            # Snapshot trajectory eagerly so each queued checkpoint captures
            # the steps recorded so far, not the final trajectory at drain time.
            # State is seeded with the current (carried-forward) state so a step
            # with no following update_state() still records the state as it
            # stood; _update_state() then patches this checkpoint if an
            # update_state() call follows, preserving the documented
            # record_step(...); update_state(...) ordering. make_checkpoint()
            # shallow-copies state (dict(state)), so top-level rebinding of
            # _current_state can't retroactively alter an already-queued
            # checkpoint; nested mutable values are not protected.
            checkpoint = make_checkpoint(
                state=self._current_state,
                trajectory_steps=self._trajectory.steps,
                run_id=self._run_id,
            )
            self._pending_checkpoints.append(checkpoint)

    def _update_state(self, state: dict[str, Any]) -> None:
        self._current_state = dict(state)
        # Patch the most-recent pending checkpoint so its state reflects the
        # update_state() call that follows record_step() in the same step block.
        if self._pending_checkpoints:
            self._pending_checkpoints[-1].state = dict(state)

    def _record_usage(self, usage: Usage) -> None:
        self._meter.record(usage)

    async def _drain_checkpoints(self) -> None:
        pending, self._pending_checkpoints = self._pending_checkpoints, []
        for checkpoint in pending:
            try:
                await self._checkpoint_store.save(checkpoint)
                self._last_checkpoint_id = checkpoint.id
            except Exception as exc:
                logger.warning("[triage] auto_checkpoint failed: %s", exc)

    async def __call__(self, task: str, **kwargs: Any) -> Any:
        return await self.run(task, **kwargs)


def agent(policy: FailurePolicy, **kwargs: Any) -> Callable[[Callable[..., Any]], Agent]:
    """Decorator factory. Wraps an async or sync function with triage recovery.

    Usage::

        @triage.agent(policy=my_policy)
        async def my_agent(task: str, *, record_step, update_state, **kwargs) -> str:
            ...

        @triage.agent(policy=my_policy)
        def my_sync_agent(task: str, *, record_step, **kwargs) -> str:
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Agent:
        return Agent(fn, policy=policy, **kwargs)
    return decorator
