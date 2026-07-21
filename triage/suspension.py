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

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from triage.taxonomy import FailureContext


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
