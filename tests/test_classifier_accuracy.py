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

  Block 3 — Corpus A (second regression suite, NOT held-out):
      Real exceptions from json/asyncio/httpx/pydantic, plus SDK error strings
      transcribed from published exception formats. This corpus was used to guide
      the v0.25 pattern fixes — rules.py was edited until it passed. It is now a
      second regression suite, not a generalization measurement.

      Corpus B (from sources not seen when writing the rules) will replace this
      as the held-out block. A floor assertion here prevents regression against
      the patterns that were already tuned.
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
# Regression floor: corpus A was used to tune v0.25 patterns, so 100% is expected.
# This is a regression guard, not a generalization claim.
CORPUS_A_FLOOR = 1.0

CORPUS_B_PATH = Path(__file__).parent / "data" / "error_corpus_b.json"
# Floor ratchet: corpus B guided the v0.26 botocore/schema fixes — it is now
# training data (same status as corpus A). 90% is expected because the rules
# were tuned against it. Update upward only; never decrease.
# Remaining misses (improvement targets for next release):
#   ServerConnectionError "Server disconnected after N seconds" → timeout
#   ValueError "Tool X is not registered" → wrong_tool_called
CORPUS_B_FLOOR = 0.90  # 18/20, measured 2026-07-27

CORPUS_C_PATH = Path(__file__).parent / "data" / "error_corpus_c.json"
# Genuine held-out floor: corpus C was scored ONCE without consulting or changing
# rules.py. Sources are disjoint from A and B: azure-core, Mistral, Cohere, Groq,
# LiteLLM, Vertex AI (aiplatform SDK), LlamaIndex, and novel phrasings.
# All 13 misses returned UNKNOWN — zero misroutes. Precision is 100%.
# Do NOT tune rules.py against corpus C misses — it will then become training data.
# Generate corpus D first, then improve, then score D.
CORPUS_C_FLOOR = 0.51  # 14/27 = 51.9%, measured 2026-07-27

# The corpus C aggregate averages two groups whose value to an adopter is opposite.
# SELF_HEALING types recover from any retry — a bare `for _ in range(3)` loop fixes
# them, so classifying them correctly adds nothing over blind retry.
# ROUTING_SENSITIVE types only recover when the matched hint reaches the strategy;
# they are the reason this library exists. Held-out recall on the two groups is
# 86% and 8% respectively, so the 52% aggregate overstates delivered value.
# These floors are ratchets, tracked separately so the aggregate cannot rise on the
# back of the group that doesn't matter.
SELF_HEALING_LABELS = ("external_fault", "timeout")
ROUTING_SENSITIVE_LABELS = ("wrong_tool_called", "schema_mismatch")
CORPUS_C_SELF_HEALING_FLOOR = 0.85  # 12/14 = 85.7%, measured 2026-07-27
# Headline goal for the next release: raise this to 0.70 via corpus D.
# Until then it is the honest ceiling on what triage delivers over a retry loop.
CORPUS_C_ROUTING_SENSITIVE_FLOOR = 0.08  # 1/12 = 8.3%, measured 2026-07-27


@dataclass
class _HeldOutResult:
    total: int
    correct: int
    misses: list[tuple[str, str, str, str]]  # (expected, got, exc_type, error)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _score_corpus(path: Path) -> _HeldOutResult:
    entries = json.loads(path.read_text())
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


def _score_corpus_group(path: Path, labels: tuple[str, ...]) -> _HeldOutResult:
    """Score only the entries whose true label is in ``labels``."""
    entries = [e for e in json.loads(path.read_text()) if e["label"] in labels]
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


def _score_corpus_a() -> _HeldOutResult:
    return _score_corpus(CORPUS_A_PATH)


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


class TestCorpusA:
    """Block 3: Corpus A regression guard.

    Corpus A was used to guide v0.25 pattern fixes — it is a second regression
    suite, not a held-out generalization measurement. 100% is expected because
    the rules were tuned against it. A miss here means a regression.

    Corpus B (from unseen sources) carries the generalization claim.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_A_PATH.exists(), f"Corpus file missing: {CORPUS_A_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_A_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_a_no_regression(self) -> None:
        result = _score_corpus_a()
        if result.misses:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus A regression: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_A_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_A_FLOOR


class TestCorpusB:
    """Block 4: Corpus B — genuine held-out generalization measurement.

    Corpus B was assembled from sources not consulted when writing or fixing the
    v0.25 patterns (botocore, google-genai, aiohttp, requests/urllib3, structurally
    different phrasings). It was scored exactly once, without editing rules.py first.

    The floor is a ratchet: update upward when patterns improve, never decrease.
    Current baseline: 50% (10/20), measured 2026-07-27.

    Misses in this block identify the next improvement targets.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_B_PATH.exists(), f"Corpus file missing: {CORPUS_B_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_B_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_b_held_out_floor(self) -> None:
        result = _score_corpus(CORPUS_B_PATH)
        if result.accuracy < CORPUS_B_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus B held-out: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_B_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_B_FLOOR


class TestCorpusC:
    """Block 5: Corpus C — genuine held-out generalization measurement.

    Corpus C was built from sources not in A or B: azure-core, Mistral AI SDK,
    Cohere SDK, Groq SDK, LiteLLM, Vertex AI (aiplatform SDK), LlamaIndex, and
    novel structural phrasings. It was scored exactly once without editing rules.py.

    52% recall (14/27), 100% precision — every miss returned UNKNOWN, no misroutes.
    The floor is a ratchet. Do NOT use corpus C misses to tune rules.py — the moment
    you do, C becomes training data. Generate corpus D first, then improve, then score D.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_C_PATH.exists(), f"Corpus file missing: {CORPUS_C_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_C_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_c_held_out_floor(self) -> None:
        result = _score_corpus(CORPUS_C_PATH)
        if result.accuracy < CORPUS_C_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus C held-out: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_C_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_C_FLOOR

    def test_corpus_c_self_healing_floor(self) -> None:
        """Recall on types a bare retry loop would recover anyway.

        High here is expected and not worth much: EXTERNAL_FAULT and TIMEOUT heal
        on any retry, so correct classification buys nothing over blind retry.
        Guarded only so a regression here is still caught.
        """
        result = _score_corpus_group(CORPUS_C_PATH, SELF_HEALING_LABELS)
        assert result.accuracy >= CORPUS_C_SELF_HEALING_FLOOR, (
            f"Self-healing held-out recall {result.accuracy:.0%}"
            f" ({result.correct}/{result.total}) below floor"
            f" {CORPUS_C_SELF_HEALING_FLOOR:.0%}."
        )

    def test_corpus_c_routing_sensitive_floor(self) -> None:
        """Recall on the types that justify the library.

        WRONG_TOOL_CALLED and SCHEMA_MISMATCH only recover when the matched hint
        reaches the strategy — these are the types scripts/bench_synthetic.py shows
        triage winning on, and the only ones where classification beats blind retry.
        Held-out recall here is the honest measure of delivered value, and it is
        currently 1/12. Raising this floor is the next release's headline goal;
        it must never be lowered.
        """
        result = _score_corpus_group(CORPUS_C_PATH, ROUTING_SENSITIVE_LABELS)
        if result.accuracy < CORPUS_C_ROUTING_SENSITIVE_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Routing-sensitive held-out recall: {result.accuracy:.0%}"
                f" ({result.correct}/{result.total}) below floor"
                f" {CORPUS_C_ROUTING_SENSITIVE_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_C_ROUTING_SENSITIVE_FLOOR

    def test_every_corpus_c_label_is_grouped(self) -> None:
        """Guard the split: a new failure type must be classified into a group.

        If a future corpus entry carries a label in neither group, the two group
        floors stop covering the corpus and the headline number silently drifts.
        UNKNOWN is exempt — it is the fall-through, not a detection target.
        """
        entries = json.loads(CORPUS_C_PATH.read_text())
        grouped = set(SELF_HEALING_LABELS) | set(ROUTING_SENSITIVE_LABELS) | {"unknown"}
        ungrouped = {e["label"] for e in entries} - grouped
        assert not ungrouped, (
            f"Corpus C labels not assigned to a group: {sorted(ungrouped)}."
            " Add them to SELF_HEALING_LABELS or ROUTING_SENSITIVE_LABELS."
        )
