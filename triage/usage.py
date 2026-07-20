"""
triage.usage
~~~~~~~~~~~~
Token and cost accounting for agent runs.

Usage inside a wrapped agent::

    from triage.agent import get_usage_recorder
    from triage.usage import Usage

    async def my_agent(task: str, *, record_step, **kwargs) -> str:
        result = call_llm(prompt)
        record_usage = get_usage_recorder()
        record_usage(Usage(input_tokens=result.usage.input_tokens,
                           output_tokens=result.usage.output_tokens,
                           cost_usd=0.0001))
        return result.content

Or accept ``record_usage`` directly as a keyword argument::

    async def my_agent(task: str, *, record_step, record_usage, **kwargs) -> str:
        ...
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Token and cost for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageMeter:
    """Accumulates Usage across all LLM calls within a run.

    Thread-safe: ``record()`` is called from the agent body (event-loop thread)
    and from ``LLMClassifier`` (which may run in a worker thread via
    ``anyio.to_thread.run_sync``), so all mutations are guarded with a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = Usage(calls=0)

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._total = Usage(
                input_tokens=self._total.input_tokens + usage.input_tokens,
                output_tokens=self._total.output_tokens + usage.output_tokens,
                cost_usd=self._total.cost_usd + usage.cost_usd,
                calls=self._total.calls + usage.calls,
            )

    @property
    def total(self) -> Usage:
        with self._lock:
            return Usage(
                input_tokens=self._total.input_tokens,
                output_tokens=self._total.output_tokens,
                cost_usd=self._total.cost_usd,
                calls=self._total.calls,
            )

    @property
    def total_tokens(self) -> int:
        return self.total.total_tokens

    @property
    def cost_usd(self) -> float:
        return self.total.cost_usd

    def reset(self) -> None:
        with self._lock:
            self._total = Usage(calls=0)
