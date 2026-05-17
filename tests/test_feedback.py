"""Tests for triage.feedback — Correction, record_correction, load_corrections."""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest

from triage.classifier.rules import RulesClassifier
from triage.feedback import Correction, load_corrections, record_correction
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
        lines = [l for l in f.read().splitlines() if l.strip()]
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
        lines = [l for l in f.read().splitlines() if l.strip()]
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

    events = [r for r in caplog.records if getattr(r, "triage_event", None) == "fit_misclassification"]
    assert events, "Expected a fit_misclassification warning"


def test_fit_no_warnings_on_correct_entry(tmp_path: Any, caplog: Any):
    store = str(tmp_path / "corrections.jsonl")
    # Write a correction where expected == what rules would actually classify
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.WRONG_TOOL_CALLED, store_path=store)

    clf = RulesClassifier()
    with caplog.at_level(logging.WARNING, logger="triage"):
        clf.fit(store)

    events = [r for r in caplog.records if getattr(r, "triage_event", None) == "fit_misclassification"]
    assert not events, "Expected no fit_misclassification warnings"


def test_fit_returns_coverage_dict(tmp_path: Any):
    store = str(tmp_path / "corrections.jsonl")
    ctx = make_ctx(FailureType.WRONG_TOOL_CALLED)
    record_correction(ctx, FailureType.WRONG_TOOL_CALLED, store_path=store)  # correct
    record_correction(ctx, FailureType.EXTERNAL_FAULT, store_path=store)    # wrong

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


# type annotation fix for pytest fixtures
from typing import Any
