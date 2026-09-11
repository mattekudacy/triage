"""tests/test_classifier_ambiguity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regression guards for tests/data/error_corpus_ambiguous.json — the corpus
scripts/hybrid_ambiguity_accuracy.py uses to measure how often
HybridClassifier overturns a correct, conservative RulesClassifier UNKNOWN
into a confident wrong guess (see that script's module docstring, README.md's
"Does LLMClassifier/HybridClassifier actually close the gap?", and
docs/known-limitations.md's "LLMClassifier/HybridClassifier close the recall
gap, but not the precision gap").

These tests make ZERO API calls — they only exercise RulesClassifier, and
they exist to guard the corpus's precondition: every entry here must be a
RulesClassifier UNKNOWN, because the whole measurement depends on
HybridClassifier escalating all of them to the LLM. If a future rules.py
pattern accidentally starts matching one of these entries, this drifts
silently and hybrid_ambiguity_accuracy.py starts measuring something other
than what it claims to — these tests catch that immediately, without needing
an LLM API key to run.

This is NOT corpus E (see scripts/README.md's Corpus discipline) and these
are not RulesClassifier accuracy floors — RulesClassifier returning UNKNOWN
here is the *correct*, desired outcome for every single entry, unlike
corpus C/D where UNKNOWN counts as a miss for routing-sensitive types.
"""

from __future__ import annotations

import json
from pathlib import Path

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

CORPUS_PATH = Path(__file__).parent / "data" / "error_corpus_ambiguous.json"
VALID_GROUPS = ("unknown_labeled", "tricky_but_classifiable")


def _classify(error: str, exception_type: str | None) -> FailureType:
    t = Trajectory()
    t.append(Step(index=0, action="test", error=error, exception_type=exception_type))
    return RulesClassifier().classify(t, "task")


def test_corpus_file_exists() -> None:
    assert CORPUS_PATH.exists(), f"Corpus file missing: {CORPUS_PATH}"


def test_corpus_all_valid_labels() -> None:
    valid = {ft.value for ft in FailureType}
    entries = json.loads(CORPUS_PATH.read_text())
    for entry in entries:
        assert entry["label"] in valid, f"Invalid label {entry['label']!r}"


def test_corpus_all_valid_groups() -> None:
    entries = json.loads(CORPUS_PATH.read_text())
    for entry in entries:
        assert entry["group"] in VALID_GROUPS, (
            f"Invalid group {entry['group']!r} on entry {entry['error'][:40]!r}. "
            f"Must be one of {VALID_GROUPS}."
        )


def test_unknown_labeled_entries_are_labeled_unknown() -> None:
    """The unknown_labeled group's whole point is entries with true label
    'unknown' — catches a mislabeled entry landing in the wrong group."""
    entries = json.loads(CORPUS_PATH.read_text())
    for entry in entries:
        if entry["group"] == "unknown_labeled":
            assert entry["label"] == "unknown", (
                f"unknown_labeled entry has label {entry['label']!r}, expected 'unknown': "
                f"{entry['error'][:40]!r}"
            )


def test_tricky_but_classifiable_entries_are_not_labeled_unknown() -> None:
    """The tricky_but_classifiable group's whole point is real, answerable
    FailureTypes phrased obliquely — a label of 'unknown' here would defeat
    the purpose (recall on oblique-but-real failures)."""
    entries = json.loads(CORPUS_PATH.read_text())
    for entry in entries:
        if entry["group"] == "tricky_but_classifiable":
            assert entry["label"] != "unknown", (
                f"tricky_but_classifiable entry is labeled 'unknown': {entry['error'][:40]!r}"
            )


def test_every_entry_is_rules_classifier_unknown() -> None:
    """The precondition the whole measurement depends on: RulesClassifier
    must return UNKNOWN for every entry in this corpus (both groups), so
    HybridClassifier escalates all of them to the LLM. If this ever fails,
    hybrid_ambiguity_accuracy.py's numbers no longer mean what its docstring
    says they mean — fix the corpus (swap or drop the newly-matched entry)
    rather than silently letting the escalation rate drop below 100%.
    """
    entries = json.loads(CORPUS_PATH.read_text())
    drifted = []
    for entry in entries:
        got = _classify(entry["error"], entry.get("exception_type"))
        if got != FailureType.UNKNOWN:
            drifted.append((entry["error"][:50], got.value))
    assert not drifted, (
        "RulesClassifier no longer returns UNKNOWN for these corpus entries "
        f"(a rules.py pattern now matches them): {drifted}. "
        "hybrid_ambiguity_accuracy.py's escalation-to-LLM precondition is broken."
    )


def test_corpus_has_both_groups_nonempty() -> None:
    entries = json.loads(CORPUS_PATH.read_text())
    groups = {e["group"] for e in entries}
    assert groups == set(VALID_GROUPS), f"Expected both groups present, got {groups}"
