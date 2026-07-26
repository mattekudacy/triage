"""
triage.suspension
~~~~~~~~~~~~~~~~~
Human-in-the-loop pause/resume support.

When a policy returns ``RecoveryAction.SUSPEND()``, the agent serializes its
current execution state into a ``SuspendedRun`` and raises
``TriageSuspendedError`` instead of raising ``TriageEscalationError`` or
continuing the recovery loop.  The caller catches the error, extracts the
token, routes it to a human (Slack, CLI, webhook — userland), and later calls
``agent.resume(token, action=...)``.

Design principles
-----------------
- **Transport-agnostic.**  The core only persists and reloads.  Nothing in
  this module knows about Slack, HTTP, or queues.
- **Store-agnostic.**  ``SuspensionStore`` is a ``Protocol`` so any backend
  (in-memory for tests, Redis for prod) can implement it.
- **Token is just a string.**  ``SuspendedRun.token`` is a UUID; callers
  can embed it in a URL, a message, or a job queue payload.
- **Single decision per resume.**  ``agent.resume()`` accepts one
  ``RecoveryAction`` that replaces the original escalation.  If the human
  decides to abort, pass ``RecoveryAction.ABORT()``.

Usage::

    from triage.suspension import RecoveryAction, SuspensionStore

    store = InMemorySuspensionStore()
    agent = Agent(my_fn, policy, suspension_store=store)

    try:
        result = await agent.run("task")
    except TriageSuspendedError as e:
        token = e.token
        # ... notify human, wait for decision ...
        result = await agent.resume(token, action=RecoveryAction.RETRY())
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from triage.taxonomy import FailureContext, FailureType, Step
from triage.usage import Usage

logger = logging.getLogger("triage")


@dataclass
class SuspendedRun:
    """Serialized state of a paused agent run.

    Fields
    ------
    token:
        Opaque identifier.  Pass to ``agent.resume()`` to restart.
    context:
        The ``FailureContext`` that triggered the suspension — contains
        trajectory, failure type, checkpoint id, and attempt history.
    task:
        The original task string passed to ``agent.run()``.
    kwargs:
        The kwargs that would have been forwarded to the next attempt.
    attempt:
        Attempt counter at the point of suspension; resumed from here.
    attempt_history:
        List of ``(FailureType, action_kind)`` tuples from prior attempts.
    timestamp:
        Wall-clock time of suspension (``time.time()``).
    message:
        Human-readable reason for suspension (from ``RecoveryAction.SUSPEND()``).
    metadata:
        Caller-supplied key/value pairs stored alongside the run; useful for
        routing (e.g. ``{"channel": "#ops", "user": "alice"}``).
    """

    token: str
    context: FailureContext
    task: str
    kwargs: dict[str, Any]
    attempt: int
    attempt_history: list[tuple[Any, str]]
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    usage_snapshot: Usage = field(default_factory=Usage)


@runtime_checkable
class SuspensionStore(Protocol):
    """Protocol for storing and retrieving suspended runs."""

    async def save(self, run: SuspendedRun) -> None: ...
    async def load(self, token: str) -> SuspendedRun: ...
    async def delete(self, token: str) -> None: ...


class InMemorySuspensionStore:
    """Default in-memory ``SuspensionStore``.  Not durable across restarts."""

    def __init__(self) -> None:
        self._store: dict[str, SuspendedRun] = {}

    async def save(self, run: SuspendedRun) -> None:
        self._store[run.token] = run

    async def load(self, token: str) -> SuspendedRun:
        if token not in self._store:
            raise KeyError(f"No suspended run with token {token!r}")
        return self._store[token]

    async def delete(self, token: str) -> None:
        self._store.pop(token, None)


def make_token() -> str:
    """Generate a fresh suspension token."""
    return str(uuid.uuid4())


_JSON_PRIMITIVES = (str, int, float, bool, list, dict, type(None))


def serialize_run(run: SuspendedRun) -> str:
    """Serialize a ``SuspendedRun`` to a JSON string.

    ``FailureContext.raw_error`` is an ``Exception`` and is not
    JSON-serializable; it is intentionally dropped here because it is never
    needed to resume a run (only the classified ``failure_type`` and
    ``trajectory`` matter).  Custom ``SuspensionStore`` implementations
    (e.g. Redis-backed) should use this helper rather than
    ``json.dumps(dataclasses.asdict(run))`` which would raise ``TypeError``.

    **kwargs filtering:** any value that is not a JSON primitive (str, int,
    float, bool, list, dict, or None) is silently dropped.  The resumed run
    will therefore execute with fewer kwargs than the suspended one —
    non-serializable values such as client handles, datetime objects, or
    custom types will be absent.  A ``logger.warning`` is emitted for each
    dropped key so callers can diagnose the mismatch.

    **tool_output coercion:** ``Step.tool_output`` values that are not JSON
    primitives are coerced to ``str()`` before serialization.  Strategies
    that inspect ``tool_output`` directly (rather than via the LLM prompt)
    will see a string after deserialization even if the original value was a
    custom object — keep this in mind when writing strategies for agents that
    set non-primitive tool outputs.
    """
    ctx = run.context

    # Warn about kwargs that will be dropped so callers can diagnose resume mismatches.
    serializable_kwargs: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in run.kwargs.items():
        if isinstance(v, _JSON_PRIMITIVES):
            serializable_kwargs[k] = v
        else:
            dropped.append(k)
    if dropped:
        logger.warning(
            "[triage] serialize_run: dropping non-serializable kwargs %r — "
            "the resumed run will not receive these values",
            dropped,
            extra={"triage_event": "serialize_kwargs_dropped", "dropped_keys": dropped},
        )

    return json.dumps({
        "token": run.token,
        "task": run.task,
        "attempt": run.attempt,
        "attempt_history": [[ft.value, kind] for ft, kind in run.attempt_history],
        "timestamp": run.timestamp,
        "message": run.message,
        "metadata": run.metadata,
        "kwargs": serializable_kwargs,
        "usage_snapshot": {
            "input_tokens": run.usage_snapshot.input_tokens,
            "output_tokens": run.usage_snapshot.output_tokens,
            "cost_usd": run.usage_snapshot.cost_usd,
            "calls": run.usage_snapshot.calls,
        },
        "context": {
            "failure_type": ctx.failure_type.value,
            "original_task": ctx.original_task,
            "critical_step_index": ctx.critical_step_index,
            "last_checkpoint_id": ctx.last_checkpoint_id,
            "loop_steps": ctx.loop_steps,
            "violated_constraint": ctx.violated_constraint,
            "expected_schema": ctx.expected_schema,
            "metadata": ctx.metadata,
            "attempt_history": [[ft.value, kind] for ft, kind in ctx.attempt_history],
            "trajectory": [
                {
                    "index": s.index,
                    "action": s.action,
                    "tool_called": s.tool_called,
                    "tool_input": s.tool_input,
                    "tool_output": s.tool_output
                    if isinstance(s.tool_output, _JSON_PRIMITIVES)
                    else str(s.tool_output),
                    "llm_output": s.llm_output,
                    "error": s.error,
                    "timestamp": s.timestamp,
                    "state_hash": s.state_hash,
                    "metadata": s.metadata,
                    "idempotent": s.idempotent,
                    "partial": s.partial,
                }
                for s in ctx.trajectory
            ],
        },
    })


def deserialize_run(data: str) -> SuspendedRun:
    """Deserialize a ``SuspendedRun`` from a JSON string produced by ``serialize_run``."""
    d = json.loads(data)
    ctx_d = d["context"]
    trajectory = [
        Step(
            index=s["index"],
            action=s["action"],
            tool_called=s.get("tool_called"),
            tool_input=s.get("tool_input"),
            tool_output=s.get("tool_output"),
            llm_output=s.get("llm_output"),
            error=s.get("error"),
            timestamp=s.get("timestamp", 0.0),
            state_hash=s.get("state_hash"),
            metadata=s.get("metadata") or {},
            idempotent=s.get("idempotent", False),
            partial=s.get("partial", False),
        )
        for s in ctx_d["trajectory"]
    ]
    ctx = FailureContext(
        failure_type=FailureType(ctx_d["failure_type"]),
        trajectory=trajectory,
        critical_step_index=ctx_d["critical_step_index"],
        original_task=ctx_d["original_task"],
        last_checkpoint_id=ctx_d.get("last_checkpoint_id"),
        loop_steps=ctx_d.get("loop_steps"),
        violated_constraint=ctx_d.get("violated_constraint"),
        expected_schema=ctx_d.get("expected_schema"),
        metadata=ctx_d.get("metadata", {}),
        attempt_history=[
            (FailureType(ft), kind) for ft, kind in ctx_d.get("attempt_history", [])
        ],
    )
    u = d.get("usage_snapshot", {})
    return SuspendedRun(
        token=d["token"],
        context=ctx,
        task=d["task"],
        kwargs=d.get("kwargs", {}),
        attempt=d["attempt"],
        attempt_history=[
            (FailureType(ft), kind) for ft, kind in d.get("attempt_history", [])
        ],
        timestamp=d.get("timestamp", 0.0),
        message=d.get("message", ""),
        metadata=d.get("metadata", {}),
        usage_snapshot=Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cost_usd=u.get("cost_usd", 0.0),
            calls=u.get("calls", 0),
        ),
    )
