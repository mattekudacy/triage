"""
triage.feedback
~~~~~~~~~~~~~~~
Record classifier misclassifications and review coverage.

Usage::

    # After a failed run where the classifier got it wrong, record the truth:
    agent.report_misclassification(FailureType.EXTERNAL_FAULT)

    # Later, review which failure types are being misclassified most often:
    corrections = load_corrections("corrections.jsonl")
    report = coverage_report(corrections)
    # {"external_fault": {"correct": 12, "wrong": 3}, ...}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from triage.taxonomy import FailureContext, FailureType

_DEFAULT_MAX_LINES = 10_000


@dataclass
class Correction:
    """A single labeled misclassification record."""

    task: str
    steps_summary: list[dict[str, Any]]
    expected_type: str
    observed_type: str
    timestamp: float = field(default_factory=time.time)


def record_correction(
    ctx: FailureContext,
    expected_type: FailureType,
    *,
    store_path: str = "corrections.jsonl",
    max_lines: int | None = _DEFAULT_MAX_LINES,
) -> None:
    """Append a labeled correction to a JSONL file.

    Parameters
    ----------
    ctx:
        The FailureContext from the failed run (available via agent._last_ctx
        or passed directly from the exception's .context attribute).
    expected_type:
        The correct FailureType the user believes should have been classified.
    store_path:
        Path to the corrections JSONL file. Appended to, not overwritten.
    max_lines:
        Rotation threshold. When appending pushes the file over this many
        lines, the whole file (including the line just written) is moved to
        ``<store_path>.1`` (overwriting any previous backup there) and a fresh
        empty file starts at ``store_path``. Pass ``None`` to disable rotation
        and let the file grow unbounded. Default 10,000 lines — comfortably
        larger than any reasonable local feedback loop, small enough that an
        unattended long-running agent can't silently fill the disk.
    """
    correction = Correction(
        task=ctx.original_task,
        steps_summary=[
            {
                "index": s.index,
                "action": s.action,
                "tool_called": s.tool_called,
                "error": s.error,
            }
            for s in ctx.trajectory
        ],
        expected_type=expected_type.value,
        observed_type=ctx.failure_type.value,
    )
    with open(store_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "task": correction.task,
                    "steps_summary": correction.steps_summary,
                    "expected_type": correction.expected_type,
                    "observed_type": correction.observed_type,
                    "timestamp": correction.timestamp,
                }
            )
            + "\n"
        )
    if max_lines is not None:
        _rotate_if_needed(store_path, max_lines)


def _rotate_if_needed(store_path: str, max_lines: int) -> None:
    try:
        with open(store_path, encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
    except FileNotFoundError:
        return
    if line_count < max_lines:
        return
    backup_path = store_path + ".1"
    os.replace(store_path, backup_path)


def load_corrections(store_path: str = "corrections.jsonl") -> list[Correction]:
    """Read all corrections from a JSONL file."""
    corrections: list[Correction] = []
    try:
        with open(store_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                corrections.append(
                    Correction(
                        task=data["task"],
                        steps_summary=data["steps_summary"],
                        expected_type=data["expected_type"],
                        observed_type=data["observed_type"],
                        timestamp=data.get("timestamp", 0.0),
                    )
                )
    except FileNotFoundError:
        pass
    return corrections


def coverage_report(
    corrections: list[Correction],
) -> dict[str, dict[str, int]]:
    """Summarise classifier accuracy from a list of corrections.

    Returns a dict keyed by the *expected* failure type value, each containing
    ``{"correct": N, "wrong": N}`` counts.  "correct" means the classifier
    agreed with the human label; "wrong" means it didn't.

    Example::

        corrections = load_corrections("corrections.jsonl")
        report = coverage_report(corrections)
        for ft, counts in sorted(report.items()):
            total = counts["correct"] + counts["wrong"]
            pct = 100 * counts["correct"] / total
            print(f"{ft}: {pct:.0f}% correct ({total} samples)")
    """
    result: dict[str, dict[str, int]] = {}
    for c in corrections:
        entry = result.setdefault(c.expected_type, {"correct": 0, "wrong": 0})
        if c.observed_type == c.expected_type:
            entry["correct"] += 1
        else:
            entry["wrong"] += 1
    return result
