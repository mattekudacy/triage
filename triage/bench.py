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
        baseline_fn=my_raw_agent_fn,
    )
    print(report.compare())
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    baseline_label: str = "baseline"
    baseline_results: list[BenchResult] = field(default_factory=list)

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

    @property
    def failure_type_counts(self) -> dict[str, int]:
        """Count how many times each failure type triggered a recovery."""
        counts: dict[str, int] = {}
        for r in self.results:
            for ft in r.failure_types:
                counts[ft] = counts.get(ft, 0) + 1
        return counts

    @property
    def _baseline_success_rate(self) -> float:
        if not self.baseline_results:
            return 0.0
        return sum(1 for r in self.baseline_results if r.success) / len(self.baseline_results)

    @property
    def _baseline_mean_latency_s(self) -> float:
        if not self.baseline_results:
            return 0.0
        return sum(r.duration_s for r in self.baseline_results) / len(self.baseline_results)

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.label}",
            f"  runs:          {len(self.results)}",
            f"  success_rate:  {self.success_rate:.1%}",
            f"  mean_latency:  {self.mean_latency_s:.3f}s",
            f"  recoveries:    {self.total_recoveries}",
        ]
        counts = self.failure_type_counts
        if counts:
            lines.append("  failure_types:")
            for ft, n in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {ft}: {n}")
        return "\n".join(lines)

    def compare(self) -> str:
        """Side-by-side comparison of triage vs baseline.

        Returns an empty string if no baseline results are available.
        """
        if not self.baseline_results:
            return ""

        w = max(len(self.label), len(self.baseline_label), 10)
        bl = self.baseline_label.ljust(w)
        tr = self.label.ljust(w)

        bsr = f"{self._baseline_success_rate:.1%}"
        bml = f"{self._baseline_mean_latency_s:.3f}s"
        lines = [
            f"{'':20}  {bl}  {tr}",
            f"{'success_rate:':<20}  {bsr}{'':<{w - 4}}  {self.success_rate:.1%}",
            f"{'mean_latency_s:':<20}  {bml}{'':<{w - 6}}  {self.mean_latency_s:.3f}s",
            f"{'recoveries:':<20}  {'—':<{w}}  {self.total_recoveries}",
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
    baseline_fn: Callable[..., Any] | None = None,
    baseline_label: str = "baseline",
) -> BenchReport:
    """Run agent_fn on each task n_runs times, wrapped with triage.

    TriageEscalationError and TriageAbortError are caught and recorded as
    failures (success=False). All other exceptions propagate to the caller.

    Recovery count is tracked via the on_recovery lifecycle hook — no
    changes to agent.py required.

    To compare triage vs a raw baseline, pass ``baseline_fn``::

        async def raw_agent(task: str) -> str:
            ...  # same logic, no triage callbacks needed

        report = await run_benchmark(
            triage_agent,
            tasks=["task A", "task B"],
            policy=policy,
            label="with-triage",
            baseline_fn=raw_agent,
            baseline_label="baseline",
        )
        print(report.compare())   # side-by-side table

    Parameters
    ----------
    agent_fn:
        Async callable with signature
        ``(task: str, *, record_step, update_state, **kwargs) -> Any``.
    tasks:
        List of task strings to run.
    policy:
        FailurePolicy to use for recovery.
    n_runs:
        Number of times to run each task (for averaging).
    label:
        Human-readable label for the triage results in the report.
    classifier:
        Optional classifier to pass to Agent. Defaults to RulesClassifier.
    max_recovery_attempts:
        Passed to Agent.__init__.
    baseline_fn:
        Optional raw agent callable ``(task: str) -> Any`` (no triage wrap).
        Run for each (task, run) pair alongside agent_fn. Exceptions are caught
        as failures. Results stored in ``report.baseline_results`` and shown
        in ``report.compare()``. ``baseline_fn`` does not receive ``record_step``
        or ``update_state`` — it is the unmodified agent.
    baseline_label:
        Label for baseline results in ``report.compare()``.
    """
    report = BenchReport(label=label, baseline_label=baseline_label)

    for task in tasks:
        for _ in range(n_runs):
            # ── triage run ────────────────────────────────────────────────────
            # recoveries and failure_types are re-bound each iteration, so the
            # on_recovery closure below always captures the current iteration's
            # variables — not the loop variable. Do not hoist these out of the loop.
            recoveries = 0
            failure_types: list[str] = []

            def on_recovery(
                ctx: FailureContext,
                action: RecoveryAction,
                _ft: list[str] = failure_types,
            ) -> None:
                nonlocal recoveries
                recoveries += 1
                _ft.append(ctx.failure_type.value)

            agent_kwargs: dict[str, Any] = dict(
                max_recovery_attempts=max_recovery_attempts,
                on_recovery=on_recovery,
            )
            if classifier is not None:
                agent_kwargs["classifier"] = classifier

            wrapped = Agent(agent_fn, policy, **agent_kwargs)

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

            # ── baseline run (no triage) ──────────────────────────────────────
            if baseline_fn is not None:
                bt0 = time.perf_counter()
                b_success = True
                try:
                    await baseline_fn(task)
                except Exception:
                    b_success = False
                b_duration = time.perf_counter() - bt0

                report.baseline_results.append(BenchResult(
                    task=task,
                    success=b_success,
                    duration_s=b_duration,
                    recoveries=0,
                ))

    return report
