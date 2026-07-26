"""Tests for triage.trajectory.Trajectory — append, replay, last_n_steps."""

from __future__ import annotations

import logging

import pytest

from triage.taxonomy import Step
from triage.trajectory import Trajectory


def make_step(index: int = 0) -> Step:
    return Step(index=index, action="step")


# ── append / steps ────────────────────────────────────────────────────────────


def test_append_and_steps():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    assert [s.index for s in t.steps] == [0, 1]


def test_steps_property_returns_copy():
    t = Trajectory()
    t.append(make_step(0))
    steps = t.steps
    steps.append(make_step(1))
    assert len(t.steps) == 1  # external mutation does not affect internal list


def test_len_and_iter():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    assert len(t) == 2
    assert [s.index for s in t] == [0, 1]


def test_getitem():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    assert t[1].index == 1


# ── non-monotonic index warning ────────────────────────────────────────────────


def test_append_monotonic_index_does_not_warn(caplog):
    t = Trajectory()
    with caplog.at_level(logging.WARNING, logger="triage"):
        t.append(make_step(0))
        t.append(make_step(1))
        t.append(make_step(2))
    assert "non-monotonic" not in caplog.text


def test_append_repeated_index_does_not_warn(caplog):
    # Equal index is normal when agents reset to index=0 on each retry attempt.
    t = Trajectory()
    with caplog.at_level(logging.WARNING, logger="triage"):
        t.append(make_step(0))
        t.append(make_step(0))
    assert "non-monotonic" not in caplog.text


def test_append_decreasing_index_warns(caplog):
    t = Trajectory()
    with caplog.at_level(logging.WARNING, logger="triage"):
        t.append(make_step(2))
        t.append(make_step(1))
    assert "non-monotonic" in caplog.text


def test_append_non_monotonic_index_still_appends():
    """Warning does not block the append — index is informational, not enforced."""
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(0))
    assert len(t) == 2


def test_append_first_step_never_warns(caplog):
    t = Trajectory()
    with caplog.at_level(logging.WARNING, logger="triage"):
        t.append(make_step(5))  # arbitrary starting index — nothing to compare against
    assert "non-monotonic" not in caplog.text


def test_append_skipped_index_does_not_warn(caplog):
    """Gaps are fine — only strictly decreasing indices are flagged."""
    t = Trajectory()
    with caplog.at_level(logging.WARNING, logger="triage"):
        t.append(make_step(0))
        t.append(make_step(5))
    assert "non-monotonic" not in caplog.text


# ── replay_from ────────────────────────────────────────────────────────────────


def test_replay_from_returns_suffix():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    t.append(make_step(2))
    replayed = t.replay_from(1)
    assert [s.index for s in replayed.steps] == [1, 2]


def test_replay_from_does_not_mutate_self():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    t.replay_from(1)
    assert len(t) == 2


def test_replay_from_out_of_range_raises():
    t = Trajectory()
    t.append(make_step(0))
    with pytest.raises(IndexError):
        t.replay_from(5)


def test_replay_from_empty_trajectory_does_not_raise():
    t = Trajectory()
    replayed = t.replay_from(0)
    assert len(replayed) == 0


# ── last_n_steps ───────────────────────────────────────────────────────────────


def test_last_n_steps_returns_last_n():
    t = Trajectory()
    for i in range(5):
        t.append(make_step(i))
    assert [s.index for s in t.last_n_steps(2)] == [3, 4]


def test_last_n_steps_zero_returns_empty():
    t = Trajectory()
    t.append(make_step(0))
    assert t.last_n_steps(0) == []


def test_last_n_steps_negative_returns_empty():
    t = Trajectory()
    t.append(make_step(0))
    assert t.last_n_steps(-1) == []


def test_last_n_steps_n_greater_than_length_returns_all():
    t = Trajectory()
    t.append(make_step(0))
    t.append(make_step(1))
    assert [s.index for s in t.last_n_steps(10)] == [0, 1]


# ── from_steps ─────────────────────────────────────────────────────────────────


def test_from_steps_constructs_with_given_steps():
    steps = [make_step(0), make_step(1)]
    t = Trajectory.from_steps(steps)
    assert [s.index for s in t.steps] == [0, 1]
