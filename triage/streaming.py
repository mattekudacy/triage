"""
triage.streaming
~~~~~~~~~~~~~~~~
Types and helpers for streaming agent support.

A streaming agent is an ``async def`` generator — it ``yield``s tokens or
chunks as it produces them, rather than returning a single value.  Triage
intercepts failures in the same way as for regular agents, but instead of
silently restarting, it yields a ``StreamRetryEvent`` so the caller knows to
discard its accumulated buffer before the next attempt begins.

Usage::

    from triage.streaming import StreamRetryEvent

    async def my_streaming_agent(task: str, *, record_step, **kwargs):
        async for chunk in llm.stream(task):
            yield chunk
        record_step(Step(index=0, action="stream"))

    agent = Agent(my_streaming_agent, policy=policy)

    buffer = []
    async for item in agent.stream("summarise this doc"):
        if isinstance(item, StreamRetryEvent):
            buffer.clear()   # discard partial output before retry
        else:
            buffer.append(item)

    result = "".join(buffer)

.. warning:: **Streaming retry is only safe for buffered consumers.**

    When a retry occurs, triage re-starts the generator from the top and
    yields a ``StreamRetryEvent`` so the caller can discard accumulated
    output.  If chunks have already been forwarded to a live-rendered output
    (a terminal, a streaming HTTP response, a WebSocket), there is no way to
    un-show them — the user will see duplicated content.

    Only use ``Agent.stream()`` with a buffer that you control and can clear
    on ``StreamRetryEvent``.  If you are streaming directly to a user, either
    accept that retries may produce visible duplication or disable recovery
    by setting ``max_recovery_attempts=0``.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage.taxonomy import FailureType


@dataclass
class StreamRetryEvent:
    """Yielded by ``Agent.stream()`` at each retry boundary.

    The caller should discard any accumulated output before continuing to
    consume chunks — triage re-starts the generator from the top.

    Attributes
    ----------
    attempt:
        The attempt number that just failed (0-based).
    failure_type:
        The classified failure type for the failed attempt.
    action_kind:
        The recovery action kind that was dispatched (e.g. ``"retry"``,
        ``"replan"``).
    hint:
        Optional hint string from the recovery action (from ``_triage_hint``
        in the next call's kwargs), if any.
    """

    attempt: int
    failure_type: FailureType
    action_kind: str
    hint: str | None = None
