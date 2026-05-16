"""
triage.bench
~~~~~~~~~~~~
Minimal benchmark harness. Run an agent callable with triage and measure
success rate, latency, and recovery count across a set of tasks.

Usage::

    from triage import FailurePolicy
    from triage.bench import run_benchmark
    from triage.strategies.retry import backoff_and_retry

    policy = FailurePolicy(EXTERNAL_FAULT=backoff_and_retry(max_attempts=3))

    report = await run_benchmark(
        my_agent_fn,
        tasks=["task A", "task B", "task C"],
        policy=policy,
        n_runs=3,
        label="with-triage",
    )
    print(report.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from triage.agent import Agent, TriageAbortError, TriageEscalationError
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import FailureContext


@dataclass
class BenchResult:
    """Result for a single (task, run) pair."""

    task: str
    success: bool
    duration_s: float
    recoveries: int
    failure_types: list[str] = field(default_factory=list)


@dataclass
class BenchReport:
    """Aggregated results from run_benchmark()."""

    label: str
    results: list[BenchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def mean_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.duration_s for r in self.results) / len(self.results)

    @property
    def total_recoveries(self) -> int:
        return sum(r.recoveries for r in self.results)

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.label}",
            f"  runs:          {len(self.results)}",
            f"  success_rate:  {self.success_rate:.1%}",
            f"  mean_latency:  {self.mean_latency_s:.3f}s",
            f"  recoveries:    {self.total_recoveries}",
        ]
        return "\n".join(lines)


async def run_benchmark(
    agent_fn: Callable[..., Any],
    tasks: list[str],
    policy: FailurePolicy,
    *,
    n_runs: int = 1,
    label: str = "triage",
    classifier: Any = None,
    max_recovery_attempts: int = 3,
) -> BenchReport:
    """Run agent_fn on each task n_runs times, wrapped with triage.

    TriageEscalationError and TriageAbortError are caught and recorded as
    failures (success=False). All other exceptions propagate to the caller.

    Recovery count is tracked via the on_recovery lifecycle hook — no
    changes to agent.py required.

    Parameters
    ----------
    agent_fn:
        Async callable with signature ``(task: str, *, record_step, update_state, **kwargs) -> Any``.
    tasks:
        List of task strings to run.
    policy:
        FailurePolicy to use for recovery.
    n_runs:
        Number of times to run each task (for averaging).
    label:
        Human-readable label for the report.
    classifier:
        Optional classifier to pass to Agent. Defaults to RulesClassifier.
    max_recovery_attempts:
        Passed to Agent.__init__.
    """
    report = BenchReport(label=label)

    for task in tasks:
        for _ in range(n_runs):
            recoveries = 0
            failure_types: list[str] = []

            def on_recovery(ctx: FailureContext, action: RecoveryAction) -> None:
                nonlocal recoveries
                recoveries += 1
                failure_types.append(ctx.failure_type.value)

            kwargs: dict[str, Any] = dict(
                max_recovery_attempts=max_recovery_attempts,
                on_recovery=on_recovery,
            )
            if classifier is not None:
                kwargs["classifier"] = classifier

            wrapped = Agent(agent_fn, policy, **kwargs)

            t0 = time.perf_counter()
            success = True
            try:
                await wrapped.run(task)
            except (TriageEscalationError, TriageAbortError):
                success = False
            duration = time.perf_counter() - t0

            report.results.append(BenchResult(
                task=task,
                success=success,
                duration_s=duration,
                recoveries=recoveries,
                failure_types=failure_types,
            ))

    return report
