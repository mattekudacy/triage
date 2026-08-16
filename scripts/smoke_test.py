"""
End-to-end smoke test — mirrors the README quickstart and key documented patterns.
Run with: PYTHONPATH=. python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys

import triage
from triage.agent import TriageAbortError, TriageEscalationError, TriageSuspendedError
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry, retry_with_tool_manifest
from triage.strategies.rollback import rollback_to_checkpoint
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

PASS = "\033[32m  PASS\033[0m"
FAIL = "\033[31m  FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"{PASS}  {label}")
    else:
        print(f"{FAIL}  {label}" + (f"  — {detail}" if detail else ""))
    return condition


results: list[bool] = []


def test(label: str, condition: bool, detail: str = "") -> None:
    results.append(check(label, condition, detail))


# ---------------------------------------------------------------------------
# 1. imports
# ---------------------------------------------------------------------------

print("\n── 1. imports ──────────────────────────────────────────────────")

try:
    import triage  # noqa: F811
    import triage.adapters.langchain
    import triage.adapters.langgraph
    import triage.agent
    import triage.breaker
    import triage.checkpoint
    import triage.checkpoint.redis
    import triage.checkpoint.sqlite
    import triage.classifier.hybrid
    import triage.classifier.llm
    import triage.classifier.rules
    import triage.strategies.circuit_breaker
    import triage.strategies.replan
    import triage.strategies.retry
    import triage.strategies.rollback
    import triage.suspension
    import triage.testing
    import triage.usage

    test("all documented modules importable", True)
except ImportError as exc:
    test("all documented modules importable", False, str(exc))

# ---------------------------------------------------------------------------
# 2. README quickstart — happy path
# ---------------------------------------------------------------------------

print("\n── 2. quickstart happy path ────────────────────────────────────")


async def test_happy_path() -> None:
    def fetch_data(task: str) -> str:
        return f"result for {task}"

    async def my_agent(task: str, *, record_step, update_state, _triage_hint=None, **kwargs):
        data = fetch_data(task)
        record_step(
            Step(
                index=0,
                action="called search",
                tool_called="search",
                tool_input={"q": task},
                tool_output=data,
            )
        )
        update_state({"data": data})
        return "done"

    policy = triage.FailurePolicy(
        WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
        EXTERNAL_FAULT=backoff_and_retry(max_attempts=5),
        LOOP_DETECTED=replan(hint="Try a different approach."),
        default=triage.FailurePolicy.escalate_by_default(),
    )

    agent = triage.Agent(my_agent, policy=policy)
    result = await agent.run("search for recent AI papers")
    test("quickstart happy path returns result", result == "done")


asyncio.run(test_happy_path())

# ---------------------------------------------------------------------------
# 3. EXTERNAL_FAULT — step recorded before raise (the regression)
# ---------------------------------------------------------------------------

print("\n── 3. EXTERNAL_FAULT with prior record_step (regression) ───────")


async def test_external_fault_with_step() -> None:
    calls = [0]

    async def my_agent(task: str, *, record_step, **kwargs):
        calls[0] += 1
        record_step(Step(index=0, action="fetch", tool_called="fetch"))
        if calls[0] == 1:
            raise RuntimeError("HTTP 503 Service Unavailable")
        return "recovered"

    classified_as: list[FailureType] = []

    async def capturing_retry(ctx: triage.FailureContext) -> triage.RecoveryAction:
        classified_as.append(ctx.failure_type)
        return triage.RecoveryAction.RETRY()

    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=capturing_retry,
        UNKNOWN=lambda ctx: triage.RecoveryAction.ABORT(reason="unexpected UNKNOWN"),
    )
    agent = triage.Agent(my_agent, policy=policy, max_recovery_attempts=3)
    result = await agent.run("task")

    test(
        "EXTERNAL_FAULT classified when step recorded before raise",
        classified_as and classified_as[0] == FailureType.EXTERNAL_FAULT,
        f"got {classified_as[0].value if classified_as else 'nothing'}",
    )
    test("agent recovers and returns result", result == "recovered")


asyncio.run(test_external_fault_with_step())

# ---------------------------------------------------------------------------
# 4. WRONG_TOOL_CALLED recovery
# ---------------------------------------------------------------------------

print("\n── 4. WRONG_TOOL_CALLED recovery ───────────────────────────────")


async def test_wrong_tool() -> None:
    calls = [0]

    async def my_agent(task: str, *, record_step, _triage_hint=None, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            record_step(
                Step(
                    index=0,
                    action="call tool",
                    error="tool 'search_v2' not found in the manifest",
                )
            )
            raise RuntimeError("tool 'search_v2' not found in the manifest")
        record_step(Step(index=0, action="call tool ok"))
        return "ok"

    policy = triage.FailurePolicy(
        WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
        default=triage.FailurePolicy.escalate_by_default(),
    )
    agent = triage.Agent(my_agent, policy=policy)
    result = await agent.run("task")
    test("WRONG_TOOL_CALLED recovers successfully", result == "ok")
    test("_triage_hint injected on retry", calls[0] == 2)


asyncio.run(test_wrong_tool())

# ---------------------------------------------------------------------------
# 5. LOOP_DETECTED recovery
# ---------------------------------------------------------------------------

print("\n── 5. LOOP_DETECTED recovery ───────────────────────────────────")


async def test_loop_detected() -> None:
    calls = [0]

    async def my_agent(task: str, *, record_step, _triage_hint=None, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            # record 3 identical steps to trip the loop detector
            for i in range(3):
                record_step(
                    Step(
                        index=i,
                        action="loop",
                        tool_called="search",
                        tool_input={"q": "same"},
                    )
                )
            raise RuntimeError("stuck in loop")
        return "replanned"

    classified_as: list[FailureType] = []

    async def capturing_replan(ctx: triage.FailureContext) -> triage.RecoveryAction:
        classified_as.append(ctx.failure_type)
        return triage.RecoveryAction.REPLAN(hint="try something else")

    policy = triage.FailurePolicy(
        LOOP_DETECTED=capturing_replan,
        default=triage.FailurePolicy.escalate_by_default(),
    )
    agent = triage.Agent(my_agent, policy=policy)
    result = await agent.run("task")
    test(
        "LOOP_DETECTED classified correctly",
        classified_as and classified_as[0] == FailureType.LOOP_DETECTED,
        f"got {classified_as[0].value if classified_as else 'nothing'}",
    )
    test("replan recovery returns result", result == "replanned")


asyncio.run(test_loop_detected())

# ---------------------------------------------------------------------------
# 6. Decorator form (@triage.agent)
# ---------------------------------------------------------------------------

print("\n── 6. decorator form ───────────────────────────────────────────")


async def test_decorator() -> None:
    policy = triage.FailurePolicy(default=triage.FailurePolicy.escalate_by_default())

    @triage.agent(policy=policy)
    async def decorated(task: str, *, record_step, **kwargs):
        record_step(Step(index=0, action="run"))
        return f"done:{task}"

    result = await decorated.run("x")
    test("decorator form: instance is Agent", isinstance(decorated, triage.Agent))
    test("decorator form: run returns result", result == "done:x")


asyncio.run(test_decorator())

# ---------------------------------------------------------------------------
# 7. Lifecycle hooks
# ---------------------------------------------------------------------------

print("\n── 7. lifecycle hooks ──────────────────────────────────────────")


async def test_hooks() -> None:
    steps_seen: list[Step] = []
    failures_seen: list[triage.FailureContext] = []
    recoveries_seen: list[tuple] = []

    calls = [0]

    async def my_agent(task: str, *, record_step, **kwargs):
        calls[0] += 1
        record_step(Step(index=0, action="step"))
        if calls[0] == 1:
            raise RuntimeError("HTTP 503 Service Unavailable")
        return "ok"

    async def fast_retry(ctx: triage.FailureContext) -> triage.RecoveryAction:
        return triage.RecoveryAction.RETRY()

    policy = triage.FailurePolicy(EXTERNAL_FAULT=fast_retry)
    agent = triage.Agent(
        my_agent,
        policy=policy,
        on_step=lambda s: steps_seen.append(s),
        on_failure=lambda ctx: failures_seen.append(ctx),
        on_recovery=lambda ctx, action: recoveries_seen.append((ctx, action)),
    )
    await agent.run("task")

    test("on_step fired", len(steps_seen) >= 1)
    got_type = failures_seen[0].failure_type if failures_seen else None
    test("on_failure fired with correct type", got_type == FailureType.EXTERNAL_FAULT)
    test("on_recovery fired", len(recoveries_seen) >= 1)


asyncio.run(test_hooks())

# ---------------------------------------------------------------------------
# 8. Escalation and TriageEscalationError
# ---------------------------------------------------------------------------

print("\n── 8. TriageEscalationError ────────────────────────────────────")


async def test_escalation() -> None:
    async def my_agent(task: str, *, record_step, **kwargs):
        record_step(Step(index=0, action="step", error="tool 'x' not found"))
        raise RuntimeError("tool 'x' not found")

    policy = triage.FailurePolicy(
        WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=1),
        default=triage.FailurePolicy.escalate_by_default(),
    )
    agent = triage.Agent(my_agent, policy=policy, max_recovery_attempts=5)
    try:
        await agent.run("task")
        test("TriageEscalationError raised", False, "no exception raised")
    except TriageEscalationError as exc:
        test("TriageEscalationError raised", True)
        test("context.failure_type set", exc.context.failure_type == FailureType.WRONG_TOOL_CALLED)
        test("context.trajectory non-empty", len(exc.context.trajectory) > 0)


asyncio.run(test_escalation())

# ---------------------------------------------------------------------------
# 9. TriageAbortError
# ---------------------------------------------------------------------------

print("\n── 9. TriageAbortError ─────────────────────────────────────────")


async def test_abort() -> None:
    async def my_agent(task: str, *, record_step, **kwargs):
        record_step(Step(index=0, action="step"))
        raise RuntimeError("fatal")

    async def abort_strategy(ctx: triage.FailureContext) -> triage.RecoveryAction:
        return triage.RecoveryAction.ABORT(reason="hard stop")

    policy = triage.FailurePolicy(default=abort_strategy)
    agent = triage.Agent(my_agent, policy=policy)
    try:
        await agent.run("task")
        test("TriageAbortError raised", False, "no exception raised")
    except TriageAbortError as exc:
        test("TriageAbortError raised", True)
        test("abort reason preserved", "hard stop" in str(exc))


asyncio.run(test_abort())

# ---------------------------------------------------------------------------
# 10. contextvar injection (get_recorder)
# ---------------------------------------------------------------------------

print("\n── 10. contextvar injection (get_recorder) ─────────────────────")


async def test_get_recorder() -> None:
    from triage.agent import get_recorder, get_state_updater

    async def my_agent(task: str, **kwargs):
        record_step = get_recorder()
        update_state = get_state_updater()
        record_step(Step(index=0, action="fetched"))
        update_state({"done": True})
        return "ok"

    policy = triage.FailurePolicy(default=triage.FailurePolicy.escalate_by_default())
    agent = triage.Agent(my_agent, policy=policy)
    result = await agent.run("task")
    test("get_recorder() works without kwarg", result == "ok")


asyncio.run(test_get_recorder())

# ---------------------------------------------------------------------------
# 11. auto_checkpoint + rollback
# ---------------------------------------------------------------------------

print("\n── 11. auto_checkpoint + rollback ──────────────────────────────")


async def test_checkpoint_rollback() -> None:
    calls = [0]
    store = triage.InMemoryCheckpointStore()

    async def my_agent(task: str, *, record_step, update_state, _triage_state=None, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            record_step(Step(index=0, action="step-a"))
            update_state({"phase": "a"})
            raise RuntimeError("HTTP 503 oops")
        # On recovery: _triage_state restored from checkpoint
        restored = _triage_state or {}
        record_step(Step(index=0, action="step-b"))
        return f"recovered:{restored.get('phase', 'none')}"

    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=rollback_to_checkpoint(),
        default=triage.FailurePolicy.escalate_by_default(),
    )
    agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
    result = await agent.run("task")
    test("rollback restores state from checkpoint", result == "recovered:a")


asyncio.run(test_checkpoint_rollback())

# ---------------------------------------------------------------------------
# 12. FailurePolicy.sequence()
# ---------------------------------------------------------------------------

print("\n── 12. FailurePolicy.sequence() ────────────────────────────────")


async def test_sequence() -> None:
    calls = [0]
    actions_taken: list[str] = []

    async def my_agent(task: str, *, record_step, **kwargs):
        calls[0] += 1
        record_step(Step(index=0, action="step", error="HTTP 503"))
        if calls[0] <= 2:
            raise RuntimeError("HTTP 503")
        return "ok"

    async def tracking_retry(ctx):
        actions_taken.append("retry")
        return triage.RecoveryAction.RETRY()

    async def tracking_replan(ctx):
        actions_taken.append("replan")
        return triage.RecoveryAction.REPLAN(hint="try again")

    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=triage.FailurePolicy.sequence(
            tracking_retry,
            tracking_replan,
        )
    )
    agent = triage.Agent(my_agent, policy=policy, max_recovery_attempts=5)
    result = await agent.run("task")
    test(
        "sequence steps through strategies in order",
        actions_taken[:2] == ["retry", "replan"],
        str(actions_taken),
    )
    test("sequence: agent eventually succeeds", result == "ok")


asyncio.run(test_sequence())

# ---------------------------------------------------------------------------
# 13. Concurrent runs (agent.clone())
# ---------------------------------------------------------------------------

print("\n── 13. concurrent runs via clone() ─────────────────────────────")


async def test_concurrent() -> None:
    async def my_agent(task: str, *, record_step, **kwargs):
        record_step(Step(index=0, action=f"run:{task}"))
        await asyncio.sleep(0.01)
        return f"done:{task}"

    policy = triage.FailurePolicy(default=triage.FailurePolicy.escalate_by_default())
    base = triage.Agent(my_agent, policy=policy)

    tasks = ["a", "b", "c"]
    agents = [base.clone() for _ in tasks]
    results = await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks, strict=True)])
    test("clone(): concurrent results correct", sorted(results) == ["done:a", "done:b", "done:c"])


asyncio.run(test_concurrent())

# ---------------------------------------------------------------------------
# 14. Suspend / resume
# ---------------------------------------------------------------------------

print("\n── 14. suspend / resume ────────────────────────────────────────")


async def test_suspend_resume() -> None:
    calls = [0]
    from triage.suspension import InMemorySuspensionStore

    store = InMemorySuspensionStore()

    async def my_agent(task: str, *, record_step, **kwargs):
        calls[0] += 1
        record_step(Step(index=0, action="step"))
        if calls[0] == 1:
            raise RuntimeError("needs human review")
        return "resumed"

    async def suspend_strategy(ctx: triage.FailureContext) -> triage.RecoveryAction:
        return triage.RecoveryAction.SUSPEND(message="approve?")

    policy = triage.FailurePolicy(default=suspend_strategy)
    agent = triage.Agent(my_agent, policy=policy, suspension_store=store)

    token = None
    try:
        await agent.run("task")
    except TriageSuspendedError as exc:
        token = exc.token
        test("TriageSuspendedError raised with token", bool(token))
        test("suspended message preserved", exc.run.message == "approve?")

    result = await agent.resume(token, action=triage.RecoveryAction.RETRY())
    test("resume returns result", result == "resumed")


asyncio.run(test_suspend_resume())

# ---------------------------------------------------------------------------
# 15. max_recovery_attempts cap
# ---------------------------------------------------------------------------

print("\n── 15. max_recovery_attempts cap ───────────────────────────────")


async def test_max_attempts() -> None:
    async def my_agent(task: str, *, record_step, **kwargs):
        record_step(Step(index=0, action="step"))
        raise RuntimeError("HTTP 503 always fails")

    async def always_retry(ctx: triage.FailureContext) -> triage.RecoveryAction:
        return triage.RecoveryAction.RETRY()

    policy = triage.FailurePolicy(EXTERNAL_FAULT=always_retry)
    agent = triage.Agent(my_agent, policy=policy, max_recovery_attempts=2)
    try:
        await agent.run("task")
        test("escalates after max_recovery_attempts", False)
    except TriageEscalationError:
        test("escalates after max_recovery_attempts", True)


asyncio.run(test_max_attempts())

# ---------------------------------------------------------------------------
# 16. RulesClassifier — direct usage
# ---------------------------------------------------------------------------

print("\n── 16. RulesClassifier direct usage ────────────────────────────")


def test_rules_classifier() -> None:
    from triage.classifier.rules import RulesClassifier

    clf = RulesClassifier()

    def traj(*error_msgs: str) -> Trajectory:
        t = Trajectory()
        for i, msg in enumerate(error_msgs):
            t.append(Step(index=i, action="step", error=msg))
        return t

    cases = [
        ("tool 'search_v2' not found", FailureType.WRONG_TOOL_CALLED),
        ("validation error: field 'x' required", FailureType.SCHEMA_MISMATCH),
        ("HTTP 503 Service Unavailable", FailureType.EXTERNAL_FAULT),
        ("Request timed out", FailureType.TIMEOUT),
        ("some unrecognized error", FailureType.UNKNOWN),
    ]
    for msg, expected in cases:
        got = clf.classify(traj(msg), "task")
        test(f"RulesClassifier: {expected.value}", got == expected, f"got {got.value}")


test_rules_classifier()

# ---------------------------------------------------------------------------
# 17. triage.testing helpers
# ---------------------------------------------------------------------------

print("\n── 17. triage.testing helpers ──────────────────────────────────")


async def test_testing_helpers() -> None:
    from triage.testing import RecordingAgent, assert_classifies_as, make_step

    s = make_step(0, tool_called="search", tool_input={"q": "x"})
    test("make_step constructs Step", isinstance(s, Step) and s.tool_called == "search")

    # RecordingAgent is the wrapped fn; fails once then succeeds
    recording = RecordingAgent(
        succeed_after=1,
        error=RuntimeError("HTTP 503 Service Unavailable"),
    )

    async def fast_retry(ctx: triage.FailureContext) -> triage.RecoveryAction:
        return triage.RecoveryAction.RETRY()

    policy = triage.FailurePolicy(
        EXTERNAL_FAULT=fast_retry,
        default=triage.FailurePolicy.escalate_by_default(),
    )
    ag = triage.Agent(recording, policy=policy)
    await ag.run("task")
    test("RecordingAgent.calls tracks all attempts", len(recording.calls) == 2)
    test("second call has _triage_context injected", "_triage_context" in recording.calls[1])

    # assert_classifies_as — keyword-arg API
    assert_classifies_as(
        steps=[make_step(0, error="HTTP 503")],
        task="task",
        expected=FailureType.EXTERNAL_FAULT,
    )
    test("assert_classifies_as passes on correct classification", True)


asyncio.run(test_testing_helpers())

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

print("\n" + "─" * 60)
passed = sum(results)
total = len(results)
color = "\033[32m" if passed == total else "\033[31m"
print(f"{color}{passed}/{total} checks passed\033[0m")
if passed < total:
    sys.exit(1)
