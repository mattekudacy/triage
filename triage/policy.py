"""
triage.policy
~~~~~~~~~~~~~
Declarative failure policy. Maps FailureType values to recovery
strategy callables. The policy is the user-facing configuration object.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

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
    ) -> RecoveryAction:
        """Re-run from the critical step, optionally injecting context."""
        return cls("retry", hint=hint, inject=inject, delay=delay if delay else None)

    @classmethod
    def REPLAN(cls, hint: str | None = None) -> RecoveryAction:
        """Abort the current plan branch and generate a new plan from scratch."""
        return cls("replan", hint=hint)

    @classmethod
    def ROLLBACK(cls, checkpoint_id: str | None = None) -> RecoveryAction:
        """Restore state to a named checkpoint and re-run from there."""
        return cls("rollback", checkpoint_id=checkpoint_id)

    @classmethod
    def RESUME(cls, from_subgoal: str | None = None) -> RecoveryAction:
        """Continue execution from an incomplete sub-goal."""
        return cls("resume", from_subgoal=from_subgoal)

    @classmethod
    def ESCALATE(cls, message: str | None = None) -> RecoveryAction:
        """Surface to a human. Halt autonomous execution."""
        return cls("escalate", message=message)

    @classmethod
    def SUSPEND(
        cls,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RecoveryAction:
        """Pause execution and await a human decision.

        Unlike ``ESCALATE`` (which raises immediately), ``SUSPEND`` serializes
        the current run state to the ``SuspensionStore`` and raises
        ``TriageSuspendedError`` carrying a token.  The caller routes the token
        to a human, waits for a decision, then calls ``agent.resume(token,
        action=...)``.

        Parameters
        ----------
        message:
            Human-readable description of why the run was suspended.
        metadata:
            Arbitrary key/value pairs to store alongside the suspended run
            (e.g. routing hints like ``{"channel": "#ops"}``).
        """
        return cls("suspend", message=message, metadata=metadata)

    @classmethod
    def ABORT(cls, reason: str | None = None) -> RecoveryAction:
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
            PLAN_INCOMPLETE    = resume_from_subgoal(),
            EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
            default            = FailurePolicy.escalate_by_default(),
        )

    Any FailureType not explicitly listed falls through to ``default``.
    """

    WRONG_TOOL_CALLED: StrategyFn | None = None
    CONSTRAINT_IGNORED: StrategyFn | None = None
    LOOP_DETECTED: StrategyFn | None = None
    PLAN_INCOMPLETE: StrategyFn | None = None
    SCHEMA_MISMATCH: StrategyFn | None = None
    CONTEXT_OVERFLOW: StrategyFn | None = None
    EXTERNAL_FAULT: StrategyFn | None = None
    TIMEOUT: StrategyFn | None = None
    UNKNOWN: StrategyFn | None = None
    default: StrategyFn | None = None

    _FIELD_MAP: dict[FailureType, str] = field(init=False, repr=False, default_factory=lambda: {
        FailureType.WRONG_TOOL_CALLED:  "WRONG_TOOL_CALLED",
        FailureType.CONSTRAINT_IGNORED: "CONSTRAINT_IGNORED",
        FailureType.LOOP_DETECTED:      "LOOP_DETECTED",
        FailureType.PLAN_INCOMPLETE:    "PLAN_INCOMPLETE",
        FailureType.SCHEMA_MISMATCH:    "SCHEMA_MISMATCH",
        FailureType.CONTEXT_OVERFLOW:   "CONTEXT_OVERFLOW",
        FailureType.EXTERNAL_FAULT:     "EXTERNAL_FAULT",
        FailureType.TIMEOUT:            "TIMEOUT",
        FailureType.UNKNOWN:            "UNKNOWN",
    })

    def resolve(self, failure_type: FailureType) -> StrategyFn | None:
        """Return the strategy for a given failure type, falling back to default."""
        field_name = self._FIELD_MAP.get(failure_type)
        if field_name:
            strategy: StrategyFn | None = getattr(self, field_name, None)
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
                message=(
                    f"No strategy registered for {ctx.failure_type.value}."
                    " Manual review required."
                )
            )
        return await strategy(ctx)

    @staticmethod
    def sequence(*strategies: StrategyFn) -> StrategyFn:
        """Return a strategy that steps through ``strategies`` in order across
        successive failures of the same type.

        On the first failure, ``strategies[0]`` is called. On the second
        failure of the same type, ``strategies[1]``, and so on. Once all
        strategies are exhausted, returns ``RecoveryAction.ESCALATE``.

        The current position is derived from ``ctx.attempt_history`` — the
        number of prior attempts that share ``ctx.failure_type`` — so no
        external state is needed and the sequence is safe to share across
        concurrent runs.

        Example — replan once, then rollback, then escalate::

            from triage.strategies.replan import replan
            from triage.strategies.rollback import rollback_to_checkpoint

            policy = FailurePolicy(
                LOOP_DETECTED=FailurePolicy.sequence(
                    replan(hint="Try a different approach."),
                    rollback_to_checkpoint(),
                ),
            )

        Example — retry twice, then replan, then escalate::

            policy = FailurePolicy(
                EXTERNAL_FAULT=FailurePolicy.sequence(
                    backoff_and_retry(),
                    backoff_and_retry(),
                    replan(hint="External service may be down."),
                ),
            )
        """
        if not strategies:
            raise ValueError("sequence() requires at least one strategy")

        async def _sequenced(ctx: FailureContext) -> RecoveryAction:
            prior = sum(1 for ft, _ in ctx.attempt_history if ft == ctx.failure_type)
            if prior < len(strategies):
                return await strategies[prior](ctx)
            return RecoveryAction.ESCALATE(
                message=(
                    f"All {len(strategies)} strategies in sequence exhausted"
                    f" for {ctx.failure_type.value}."
                )
            )
        return _sequenced

    @staticmethod
    def chain(
        primary: StrategyFn,
        fallback: StrategyFn,
        after_kinds: tuple[str, ...] = ("escalate",),
    ) -> StrategyFn:
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
                message=(
                    f"Unhandled failure: {ctx.failure_type.value}"
                    f" at step {ctx.critical_step_index}"
                )
            )
        return _escalate

    @staticmethod
    def abort_by_default() -> StrategyFn:
        """Default strategy: hard abort on any unhandled failure."""
        async def _abort(ctx: FailureContext) -> RecoveryAction:
            return RecoveryAction.ABORT(reason=ctx.failure_type.value)
        return _abort

    @classmethod
    def from_dict(
        cls,
        strategies: dict[FailureType, StrategyFn],
        *,
        default: StrategyFn | None = None,
    ) -> FailurePolicy:
        """Build a ``FailurePolicy`` from a ``{FailureType: strategy}`` mapping.

        Useful when constructing policies dynamically (e.g. from a config dict,
        per-tenant rules, or runtime composition)::

            policy = FailurePolicy.from_dict({
                FailureType.EXTERNAL_FAULT: backoff_and_retry(max_attempts=3),
                FailureType.TIMEOUT:        backoff_and_retry(max_attempts=2),
            }, default=FailurePolicy.escalate_by_default())

        ``FailureType`` member names match the dataclass field names exactly,
        so this is equivalent to::

            policy = FailurePolicy(
                EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
                TIMEOUT=backoff_and_retry(max_attempts=2),
                default=FailurePolicy.escalate_by_default(),
            )
        """
        kwargs: dict[str, StrategyFn | None] = {ft.name: s for ft, s in strategies.items()}
        if default is not None:
            kwargs["default"] = default
        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        strategy_registry: dict[str, Any] | None = None,
    ) -> FailurePolicy:
        """Load a ``FailurePolicy`` from a TOML or YAML config file.

        File format is determined by extension:
        - ``.toml`` — parsed with stdlib ``tomllib`` (Python 3.11+)
        - ``.yaml`` / ``.yml`` — parsed with ``pyyaml`` (install ``triage-agent[yaml]``)

        TOML example (``policy.toml``)::

            [EXTERNAL_FAULT]
            strategy = "backoff_and_retry"
            max_attempts = 5

            [TIMEOUT]
            strategy = "backoff_and_retry"
            max_attempts = 3

            [WRONG_TOOL_CALLED]
            strategy = "retry_with_tool_manifest"
            max_attempts = 2

            default = "escalate"

        YAML example (``policy.yaml``)::

            EXTERNAL_FAULT:
              strategy: backoff_and_retry
              max_attempts: 5
            default: escalate

        Built-in strategy names: ``backoff_and_retry``, ``retry_with_tool_manifest``,
        ``replan``, ``resume_from_subgoal``, ``rollback_to_checkpoint``,
        ``escalate``, ``abort``.

        Parameters
        ----------
        path:
            Path to the config file.
        strategy_registry:
            Optional dict mapping name → callable (or factory). When provided,
            merged with the built-in registry (custom names take precedence).
        """
        # Lazy imports to keep core deps minimal
        import os
        raw: dict[str, Any]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".toml":
            try:
                import tomllib  # stdlib 3.11+
            except ImportError:
                import tomli as tomllib  # type: ignore[import-untyped,import-not-found,no-redef]
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError(
                    "pyyaml is required for YAML config files. "
                    "Install it with: pip install triage-agent[yaml]"
                ) from e
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raise ValueError(
                f"Unsupported config file extension {ext!r}. Use .toml, .yaml, or .yml"
            )

        return cls._from_raw_config(raw, strategy_registry=strategy_registry)

    @classmethod
    def _from_raw_config(
        cls,
        raw: dict[str, Any],
        *,
        strategy_registry: dict[str, Any] | None = None,
    ) -> FailurePolicy:
        """Build a FailurePolicy from a parsed config dict (internal)."""
        # Lazy import strategies here to avoid circular imports at module level
        from triage.strategies.replan import replan, resume_from_subgoal
        from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
        from triage.strategies.rollback import rollback_to_checkpoint

        _builtin: dict[str, Any] = {
            "backoff_and_retry": backoff_and_retry,
            "retry_with_tool_manifest": retry_with_tool_manifest,
            "replan": replan,
            "resume_from_subgoal": resume_from_subgoal,
            "rollback_to_checkpoint": rollback_to_checkpoint,
            "escalate": lambda **_kw: cls.escalate_by_default(),
            "abort": lambda **_kw: cls.abort_by_default(),
        }
        registry = {**_builtin, **(strategy_registry or {})}

        valid_fields = {ft.name for ft in FailureType}
        kwargs: dict[str, StrategyFn | None] = {}

        for key, value in raw.items():
            if key == "default":
                # default = "escalate" or default = "abort"
                if isinstance(value, str):
                    if value not in registry:
                        raise ValueError(
                            f"Unknown strategy {value!r} for 'default'. "
                            f"Available: {sorted(registry)}"
                        )
                    kwargs["default"] = registry[value]()
                continue

            if key not in valid_fields:
                raise ValueError(
                    f"Unknown FailureType {key!r}. "
                    f"Valid types: {sorted(valid_fields)}"
                )

            if isinstance(value, str):
                # EXTERNAL_FAULT = "backoff_and_retry"
                strategy_name = value
                strategy_kwargs: dict[str, Any] = {}
            elif isinstance(value, dict):
                strategy_name = value.get("strategy", "")
                strategy_kwargs = {k: v for k, v in value.items() if k != "strategy"}
            else:
                raise ValueError(
                    f"Invalid value for {key!r}: expected string or dict, got {type(value)!r}"
                )

            if strategy_name not in registry:
                raise ValueError(
                    f"Unknown strategy {strategy_name!r} for {key!r}. "
                    f"Available: {sorted(registry)}"
                )

            factory = registry[strategy_name]
            kwargs[key] = factory(**strategy_kwargs)

        return cls(**kwargs)
