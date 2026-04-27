"""Tests for triage.taxonomy — FailureType, Step, FailureContext."""

from triage.taxonomy import FailureContext, FailureType, Step


def make_step(index: int = 0, error: str | None = None) -> Step:
    return Step(index=index, action="test step", error=error)


def test_failure_type_count():
    assert len(FailureType) == 10


def test_failure_type_values_are_strings():
    for ft in FailureType:
        assert isinstance(ft.value, str)
        assert ft.value  # non-empty


def test_failure_type_all_members_present():
    names = {ft.name for ft in FailureType}
    expected = {
        "WRONG_TOOL_CALLED",
        "CONSTRAINT_IGNORED",
        "LOOP_DETECTED",
        "HALLUCINATED_STATE",
        "PLAN_INCOMPLETE",
        "SCHEMA_MISMATCH",
        "CONTEXT_OVERFLOW",
        "GOAL_DRIFT",
        "EXTERNAL_FAULT",
        "UNKNOWN",
    }
    assert names == expected


def test_failed_step_valid_index():
    steps = [make_step(0), make_step(1), make_step(2)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=1,
        original_task="test",
    )
    assert ctx.failed_step is steps[1]


def test_failed_step_first_index():
    steps = [make_step(0), make_step(1)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=0,
        original_task="test",
    )
    assert ctx.failed_step is steps[0]


def test_failed_step_out_of_range_high():
    steps = [make_step(0)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=5,
        original_task="test",
    )
    assert ctx.failed_step is None


def test_failed_step_out_of_range_negative():
    steps = [make_step(0)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=-1,
        original_task="test",
    )
    assert ctx.failed_step is None


def test_steps_after_failure_empty_when_last():
    steps = [make_step(0), make_step(1), make_step(2)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=2,
        original_task="test",
    )
    assert ctx.steps_after_failure == []


def test_steps_after_failure_populated():
    steps = [make_step(0), make_step(1), make_step(2)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=0,
        original_task="test",
    )
    assert ctx.steps_after_failure == [steps[1], steps[2]]


def test_steps_after_failure_middle():
    steps = [make_step(i) for i in range(5)]
    ctx = FailureContext(
        failure_type=FailureType.UNKNOWN,
        trajectory=steps,
        critical_step_index=2,
        original_task="test",
    )
    assert ctx.steps_after_failure == [steps[3], steps[4]]
