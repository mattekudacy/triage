"""Tests for triage.bench — BenchResult, BenchReport, run_benchmark."""

from triage.bench import BenchReport, BenchResult, run_benchmark
from triage.policy import FailurePolicy, RecoveryAction
from triage.taxonomy import Step

# ── BenchReport properties ────────────────────────────────────────────────────


def test_bench_report_success_rate_all_pass():
    report = BenchReport(
        label="test",
        results=[
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=0),
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=0),
        ],
    )
    assert report.success_rate == 1.0


def test_bench_report_success_rate_half():
    report = BenchReport(
        label="test",
        results=[
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=0),
            BenchResult(task="t", success=False, duration_s=1.0, recoveries=0),
        ],
    )
    assert report.success_rate == 0.5


def test_bench_report_success_rate_empty():
    report = BenchReport(label="test")
    assert report.success_rate == 0.0


def test_bench_report_mean_latency():
    report = BenchReport(
        label="test",
        results=[
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=0),
            BenchResult(task="t", success=True, duration_s=3.0, recoveries=0),
        ],
    )
    assert report.mean_latency_s == 2.0


def test_bench_report_mean_latency_empty():
    report = BenchReport(label="test")
    assert report.mean_latency_s == 0.0


def test_bench_report_total_recoveries():
    report = BenchReport(
        label="test",
        results=[
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=2),
            BenchResult(task="t", success=True, duration_s=1.0, recoveries=3),
        ],
    )
    assert report.total_recoveries == 5


def test_bench_report_summary_contains_label():
    report = BenchReport(
        label="my-experiment",
        results=[
            BenchResult(task="t", success=True, duration_s=0.1, recoveries=0),
        ],
    )
    summary = report.summary()
    assert "my-experiment" in summary
    assert "100.0%" in summary


# ── run_benchmark ─────────────────────────────────────────────────────────────


async def test_run_benchmark_all_success():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        record_step(Step(index=0, action="done"))
        return "ok"

    policy = FailurePolicy()
    report = await run_benchmark(agent, tasks=["task1", "task2"], policy=policy, label="test")
    assert report.success_rate == 1.0
    assert report.total_recoveries == 0
    assert len(report.results) == 2


async def test_run_benchmark_with_recovery():
    calls = {"n": 0}

    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        calls["n"] += 1
        record_step(Step(index=0, action="step", error="tool foo not found"))
        if calls["n"] == 1:
            raise RuntimeError("tool foo not found")
        return "ok"

    async def retry_strategy(ctx):
        return RecoveryAction.RETRY()

    policy = FailurePolicy(WRONG_TOOL_CALLED=retry_strategy)
    report = await run_benchmark(agent, tasks=["task1"], policy=policy, label="test")
    assert report.results[0].success is True
    assert report.total_recoveries == 1
    assert "wrong_tool_called" in report.results[0].failure_types


async def test_run_benchmark_escalation_counts_as_failure():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    policy = FailurePolicy(default=FailurePolicy.escalate_by_default())
    report = await run_benchmark(agent, tasks=["task1"], policy=policy, label="test")
    assert report.success_rate == 0.0
    assert report.results[0].success is False


async def test_run_benchmark_abort_counts_as_failure():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        record_step(Step(index=0, action="step", error="tool foo not found"))
        raise RuntimeError("tool foo not found")

    policy = FailurePolicy(default=FailurePolicy.abort_by_default())
    report = await run_benchmark(agent, tasks=["task1"], policy=policy, label="test")
    assert report.success_rate == 0.0
    assert report.results[0].success is False


async def test_run_benchmark_multiple_runs_per_task():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        record_step(Step(index=0, action="done"))
        return "ok"

    policy = FailurePolicy()
    report = await run_benchmark(agent, tasks=["t"], policy=policy, n_runs=3, label="test")
    assert len(report.results) == 3
    assert report.success_rate == 1.0


async def test_run_benchmark_label_in_report():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        return "ok"

    policy = FailurePolicy()
    report = await run_benchmark(agent, tasks=["t"], policy=policy, label="my-label")
    assert report.label == "my-label"


# ── baseline comparison ───────────────────────────────────────────────────────


async def test_run_benchmark_with_baseline_fn():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        return "ok"

    async def baseline(task: str) -> str:
        return "ok"

    policy = FailurePolicy()
    report = await run_benchmark(agent, tasks=["t1", "t2"], policy=policy, baseline_fn=baseline)
    assert len(report.baseline_results) == 2
    assert all(r.success for r in report.baseline_results)


async def test_bench_report_compare_output():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        return "ok"

    async def baseline(task: str) -> str:
        raise RuntimeError("baseline fails")

    policy = FailurePolicy()
    report = await run_benchmark(
        agent,
        tasks=["t"],
        policy=policy,
        baseline_fn=baseline,
        label="triage",
        baseline_label="raw",
    )
    cmp = report.compare()
    assert "triage" in cmp
    assert "raw" in cmp
    assert "100.0%" in cmp  # triage success rate
    assert "0.0%" in cmp  # baseline failure rate


async def test_bench_report_compare_no_baseline_returns_empty():
    async def agent(task: str, *, record_step, update_state, **kwargs) -> str:
        return "ok"

    policy = FailurePolicy()
    report = await run_benchmark(agent, tasks=["t"], policy=policy)
    assert report.compare() == ""
