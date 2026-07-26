"""Tests for triage.feedback — Correction, record_correction, load_corrections, coverage_report."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from triage.classifier.rules import RulesClassifier
from triage.feedback import Correction, coverage_report, load_corrections, record_correction
from triage.taxonomy import FailureContext, FailureType, Step


def make_ctx(
    failure_type: FailureType = FailureType.WRONG_TOOL_CALLED,
    task: str = "test task",
) -> FailureContext:
    return FailureContext(
        failure_type=failure_type,
        trajectory=[Step(index=0, action="step", error="tool foo not found")],
        critical_step_index=0,
        original_task=task,
    )


# ── record_correction ─────────────────────────────────────────────────────────


def test_record_correction_writes_to_file(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)

    with open(store) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["expected_type"] == "external_fault"
    assert data["observed_type"] == "wrong_tool_called"


def test_record_correction_appends(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)
    record_correction(ctx, FailureType.TIMEOUT, store_path=store)

    with open(store) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    assert len(lines) == 2


def test_record_correction_fields(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.SCHEMA_MISMATCH, task="my important task")
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)

    with open(store) as f:
        data = json.loads(f.read().strip())

    assert data["task"] == "my important task"
    assert data["expected_type"] == "external_fault"
    assert data["observed_type"] == "schema_mismatch"
    assert "timestamp" in data
    assert isinstance(data["steps_summary"], list)
    assert data["steps_summary"][0]["action"] == "step"


# ── rotation ───────────────────────────────────────────────────────────────────


def test_record_correction_does_not_rotate_below_threshold(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    for _ in range(5):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store, max_lines=10)

    assert not os.path.exists(store + ".1")
    with open(store) as f:
        assert len(f.read().splitlines()) == 5


def test_record_correction_rotates_when_threshold_exceeded(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    for _ in range(4):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store, max_lines=3)

    # With max_lines=3, rotation fires as soon as the file reaches 3 lines.
    # Write 3 → backup holds 3 lines, primary is empty.
    # Write 4 → goes to fresh primary (1 line); backup still holds 3.
    assert os.path.exists(store + ".1")
    with open(store + ".1") as f:
        assert len(f.read().splitlines()) == 3
    with open(store) as f:
        assert len(f.read().splitlines()) == 1


def test_record_correction_rotation_overwrites_previous_backup(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    backup = store + ".1"
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)

    for _ in range(3):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store, max_lines=2)
    with open(backup) as f:
        first_backup_lines = len(f.read().splitlines())

    for _ in range(3):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store, max_lines=2)
    with open(backup) as f:
        second_backup_lines = len(f.read().splitlines())

    # Backup was overwritten, not appended to — same shape both times
    assert first_backup_lines == second_backup_lines


def test_record_correction_max_lines_none_disables_rotation(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    for _ in range(10):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store, max_lines=None)

    assert not os.path.exists(store + ".1")
    with open(store) as f:
        assert len(f.read().splitlines()) == 10


def test_record_correction_default_max_lines_does_not_rotate_small_files(tmp_path: Any):
    """Sanity check that the default threshold (10,000) doesn't kick in for
    ordinary usage — only an explicit low max_lines should trigger rotation
    in these tests."""
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    for _ in range(5):
        record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)

    assert not os.path.exists(store + ".1")


def test_load_corrections_returns_empty_for_missing_file(tmp_path: Any):
    corrections = load_corrections(str(tmp_path / "nonexistent.jsonl"))
    assert corrections == []


def test_load_corrections_round_trips(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)

    corrections = load_corrections(store)
    assert len(corrections) == 1
    assert corrections[0].expected_type == "external_fault"
    assert corrections[0].observed_type == "wrong_tool_called"


# ── RulesClassifier.fit() ────────────────────────────────────────────────────


def test_fit_warns_on_misclassification(tmp_path: Any, caplog: Any):
    store = str(tmp_path / "corrections.jsonl")
    # Write a correction where rules would classify WRONG_TOOL_CALLED but
    # the expected type is EXTERNAL_FAULT
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)

    clf = RulesClassifier()
    with caplog.at_level(logging.WARNING, logger="triage"):
        clf.fit(store)

    fit_events = [
        r for r in caplog.records if getattr(r, "triage_event", None) == "fit_misclassification"
    ]
    assert fit_events, "Expected a fit_misclassification warning"


def test_fit_no_warnings_on_correct_entry(tmp_path: Any, caplog: Any):
    store = str(tmp_path / "corrections.jsonl")
    # Write a correction where expected == what rules would actually classify
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.WRONG_TOOL_CALLED, store_path=store)

    clf = RulesClassifier()
    with caplog.at_level(logging.WARNING, logger="triage"):
        clf.fit(store)

    misclass = [
        r for r in caplog.records if getattr(r, "triage_event", None) == "fit_misclassification"
    ]
    assert not misclass, "Expected no fit_misclassification warnings"


def test_fit_returns_coverage_dict(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.WRONG_TOOL_CALLED, store_path=store)  # correct
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)  # wrong

    clf = RulesClassifier()
    coverage = clf.fit(store)

    assert "wrong_tool_called" in coverage
    assert coverage["wrong_tool_called"]["correct"] == 1
    assert "external_fault" in coverage
    assert coverage["external_fault"]["wrong"] == 1


def test_fit_empty_file(tmp_path: Any):
    store = str(tmp_path / "empty.jsonl")
    # File doesn't exist — should return empty dict without error
    clf = RulesClassifier()
    coverage = clf.fit(store)
    assert coverage == {}


# ── coverage_report ───────────────────────────────────────────────────────────


def test_coverage_report_counts_correct_and_wrong():
    corrections = [
        Correction(
            task="t",
            steps_summary=[],
            expected_type="external_fault",
            observed_type="external_fault",
            timestamp=0.0,
        ),
        Correction(
            task="t",
            steps_summary=[],
            expected_type="external_fault",
            observed_type="unknown",
            timestamp=0.0,
        ),
        Correction(
            task="t",
            steps_summary=[],
            expected_type="external_fault",
            observed_type="unknown",
            timestamp=0.0,
        ),
        Correction(
            task="t",
            steps_summary=[],
            expected_type="timeout",
            observed_type="timeout",
            timestamp=0.0,
        ),
    ]
    report = coverage_report(corrections)
    assert report["external_fault"] == {"correct": 1, "wrong": 2}
    assert report["timeout"] == {"correct": 1, "wrong": 0}


def test_coverage_report_empty_list():
    assert coverage_report([]) == {}


def test_coverage_report_all_correct():
    corrections = [
        Correction(
            task="t",
            steps_summary=[],
            expected_type="timeout",
            observed_type="timeout",
            timestamp=0.0,
        ),
        Correction(
            task="t",
            steps_summary=[],
            expected_type="timeout",
            observed_type="timeout",
            timestamp=0.0,
        ),
    ]
    report = coverage_report(corrections)
    assert report["timeout"] == {"correct": 2, "wrong": 0}
