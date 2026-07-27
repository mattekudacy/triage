"""tests/test_classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classifier accuracy tests with three labeled measurement blocks.

  Block 1 — Regression:
      32 positive examples from test_classifier_rules.py fixtures.
      Expected: 100%. This is tautological by design — the regexes were
      written against these strings. The purpose is to prevent *regression*,
      not to measure generalization.

  Block 2 — False-positive resistance:
      Near-miss strings that must NOT trigger a rule. A false positive routes
      a failure to the wrong recovery strategy, which is worse than UNKNOWN.
      Expected: 100%. Treated as a hard constraint.

  Block 3 — Held-out accuracy (corpus A):
      Real exceptions provoked from installed SDKs (json, asyncio, httpx,
      pydantic, openai, anthropic, langchain, langgraph). These strings were
      NOT used to write the regexes. The score here is the number to improve
      against across releases.

      A floor assertion pegs the held-out score at its measured baseline.
      If it drops below that floor, the test fails — it is a ratchet, not a
      target. The floor is updated (upward only) when the patterns improve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def _t(error: str | None = None, exception_type: str | None = None) -> Trajectory:
    t = Trajectory()
    t.append(Step(index=0, action="test", error=error, exception_type=exception_type))
    return t


def _classify(error: str | None, exception_type: str | None = None) -> FailureType:
    return RulesClassifier().classify(_t(error, exception_type), "task")


# ── Block 1: Regression corpus (positive examples from named tests) ────────────

REGRESSION_POSITIVES: list[tuple[str, FailureType]] = [
    # WRONG_TOOL_CALLED
    ("no tool named calculator", FailureType.WRONG_TOOL_CALLED),
    ("Tool 'bar' not found", FailureType.WRONG_TOOL_CALLED),
    ("NO TOOL NAMED foo", FailureType.WRONG_TOOL_CALLED),
    ("tool_not_found: the requested tool does not exist", FailureType.WRONG_TOOL_CALLED),
    ("function 'send_email' does not exist", FailureType.WRONG_TOOL_CALLED),
    # SCHEMA_MISMATCH
    ("validation error: field required", FailureType.SCHEMA_MISMATCH),
    ("JSONDecodeError: Expecting value at line 1", FailureType.SCHEMA_MISMATCH),
    ("json parse failed", FailureType.SCHEMA_MISMATCH),
    ("validation error: expected string at field 'name'", FailureType.SCHEMA_MISMATCH),
    ("jsondecodeerror at line 1", FailureType.SCHEMA_MISMATCH),
    ("invalid json in response body", FailureType.SCHEMA_MISMATCH),
    ("unexpected token '{' in json", FailureType.SCHEMA_MISMATCH),
    ("failed to json parse the response", FailureType.SCHEMA_MISMATCH),
    # EXTERNAL_FAULT
    ("HTTP 429 Too Many Requests", FailureType.EXTERNAL_FAULT),
    ("500 Internal Server Error", FailureType.EXTERNAL_FAULT),
    ("502 Bad Gateway", FailureType.EXTERNAL_FAULT),
    ("503 Service Unavailable", FailureType.EXTERNAL_FAULT),
    ("rate limited, status 429", FailureType.EXTERNAL_FAULT),
    ("HTTP 429: rate limited", FailureType.EXTERNAL_FAULT),
    ("status code 500", FailureType.EXTERNAL_FAULT),
    ("received 503 from upstream", FailureType.EXTERNAL_FAULT),
    ("server returned 502 bad gateway", FailureType.EXTERNAL_FAULT),
    ("upstream error 429", FailureType.EXTERNAL_FAULT),
    ("got 500 from remote", FailureType.EXTERNAL_FAULT),
    # TIMEOUT
    ("asyncio.TimeoutError: timeout", FailureType.TIMEOUT),
    ("request timed out after 30s", FailureType.TIMEOUT),
    ("deadline exceeded", FailureType.TIMEOUT),
    ("time limit reached", FailureType.TIMEOUT),
    ("operation timed out after 30s", FailureType.TIMEOUT),
    ("deadline exceeded for request", FailureType.TIMEOUT),
    ("async time limit reached", FailureType.TIMEOUT),
    ("timed out waiting for response", FailureType.TIMEOUT),
]


# ── Block 2: False-positive resistance ────────────────────────────────────────
# Near-miss strings that must return UNKNOWN (or any non-positive type).
# Keyed as (error_string, type_that_must_NOT_fire).

FALSE_POSITIVE_GUARDS: list[tuple[str, FailureType]] = [
    # Numbers that look like HTTP codes but aren't
    ("expected 500 items but got 42", FailureType.EXTERNAL_FAULT),
    ("processed 503 records successfully", FailureType.EXTERNAL_FAULT),
    ("returned 429 results", FailureType.EXTERNAL_FAULT),
    ("502 bytes written", FailureType.EXTERNAL_FAULT),
    ("step 500 completed", FailureType.EXTERNAL_FAULT),
    ("line 503: syntax error", FailureType.EXTERNAL_FAULT),
    ("error in row 429", FailureType.EXTERNAL_FAULT),
    ("expected 200 records", FailureType.EXTERNAL_FAULT),
    # 'tool' in non-error contexts
    ("tooltip not found in DOM", FailureType.WRONG_TOOL_CALLED),
    ("found 3 tools available", FailureType.WRONG_TOOL_CALLED),
    ("initialize tool chain", FailureType.WRONG_TOOL_CALLED),
    ("toolbox is empty", FailureType.WRONG_TOOL_CALLED),
    ("retool configuration loaded", FailureType.WRONG_TOOL_CALLED),
    # Unrelated errors
    ("connection refused", FailureType.WRONG_TOOL_CALLED),
    ("index out of range", FailureType.SCHEMA_MISMATCH),
]


# ── Block 3: Held-out accuracy (corpus A) ─────────────────────────────────────

CORPUS_A_PATH = Path(__file__).parent / "data" / "error_corpus_a.json"
# Ratchet floor: held-out accuracy must not drop below this.
# Update upward when patterns improve; never decrease.
HELD_OUT_FLOOR = 1.0  # 30/30 after v0.25 pattern fixes


@dataclass
class _HeldOutResult:
    total: int
    correct: int
    misses: list[tuple[str, str, str, str]]  # (expected, got, exc_type, error)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _score_corpus_a() -> _HeldOutResult:
    entries = json.loads(CORPUS_A_PATH.read_text())
    clf = RulesClassifier()
    correct = 0
    misses = []
    for entry in entries:
        t = Trajectory()
        t.append(
            Step(
                index=0,
                action="a",
                error=entry["error"],
                exception_type=entry.get("exception_type"),
            )
        )
        got = clf.classify(t, "task").value
        exp = entry["label"]
        if got == exp:
            correct += 1
        else:
            misses.append((exp, got, entry.get("exception_type", ""), entry["error"][:60]))
    return _HeldOutResult(total=len(entries), correct=correct, misses=misses)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRegression:
    """Block 1: Regression — in-corpus positive examples must classify correctly."""

    @pytest.mark.parametrize("error,expected", REGRESSION_POSITIVES)
    def test_positive(self, error: str, expected: FailureType) -> None:
        assert _classify(error) == expected


class TestFalsePositiveResistance:
    """Block 2: False-positive resistance — near-miss strings must not fire the named type."""

    @pytest.mark.parametrize("error,forbidden_type", FALSE_POSITIVE_GUARDS)
    def test_no_false_positive(self, error: str, forbidden_type: FailureType) -> None:
        assert _classify(error) != forbidden_type


class TestHeldOut:
    """Block 3: Held-out accuracy — real SDK exceptions not used to write the rules."""

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_A_PATH.exists(), f"Corpus file missing: {CORPUS_A_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_A_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_held_out_accuracy_above_floor(self) -> None:
        result = _score_corpus_a()
        if result.misses:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Held-out accuracy {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" is below floor {HELD_OUT_FLOOR:.0%}.\n"
                f"Misses:\n{detail}"
            )
        assert result.accuracy >= HELD_OUT_FLOOR, (
            f"Held-out accuracy {result.accuracy:.0%} < floor {HELD_OUT_FLOOR:.0%}"
        )
