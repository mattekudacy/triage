"""tests/test_mast_pilot_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structural regression guards for tests/data/mast_pilot_corpus.json — the
pilot corpus scripts/mast_mode_pilot_accuracy.py scores (see that script's
module docstring and docs/concepts/multi-agent-failures.md's "Phase 3
scoping" section).

Zero API calls, zero triage/ imports — these only guard the corpus's own
shape (every entry cites a real source, every mode code is one of the three
piloted modes, every "ambiguous" entry is actually excluded from scoring),
not classifier accuracy. If a future edit to the corpus accidentally drops a
required field or introduces a typo'd mode code, scripts/mast_mode_pilot_
accuracy.py would silently score something other than what it claims to —
these tests catch that without needing an LLM API key.
"""

from __future__ import annotations

import json
from pathlib import Path

CORPUS_PATH = Path(__file__).parent / "data" / "mast_pilot_corpus.json"
PILOTED_MODES = {"2.5", "2.4", "1.4"}


def _entries() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text())


def test_corpus_file_exists() -> None:
    assert CORPUS_PATH.exists(), f"Corpus file missing: {CORPUS_PATH}"


def test_every_entry_has_required_fields() -> None:
    required = {"task", "steps", "provenance", "source_url", "quote", "notes"}
    for entry in _entries():
        missing = required - entry.keys()
        assert not missing, f"Entry {entry.get('provenance', '?')!r} missing {missing}"


def test_every_entry_has_exactly_one_mode_field() -> None:
    """Every entry is either scorable (mast_mode set, no ambiguous_with) or
    ambiguous (ambiguous_with set, no single mast_mode) — never both, never
    neither."""
    for entry in _entries():
        has_mode = "mast_mode" in entry
        has_ambiguous = "ambiguous_with" in entry
        assert has_mode != has_ambiguous, (
            f"Entry {entry['provenance']!r} must set exactly one of "
            "'mast_mode' or 'ambiguous_with', not both or neither"
        )


def test_scorable_entries_use_a_piloted_mode_code() -> None:
    for entry in _entries():
        if "mast_mode" in entry:
            assert entry["mast_mode"] in PILOTED_MODES, (
                f"Entry {entry['provenance']!r} has mast_mode "
                f"{entry['mast_mode']!r}, not one of the piloted modes "
                f"{PILOTED_MODES} — widening this corpus beyond the pilot "
                "is a deliberate decision, not a typo"
            )


def test_ambiguous_entries_point_at_a_piloted_mode() -> None:
    for entry in _entries():
        if "ambiguous_with" in entry:
            assert entry["ambiguous_with"] in PILOTED_MODES


def test_each_piloted_mode_meets_its_own_documented_source_count() -> None:
    """Guards the specific counts the module docstrings claim — not a
    blanket ">= 2 for every mode" floor. 2.5 and 1.4 each got two
    independently-sourced clean examples; 2.4's second real candidate
    (openai-agents-python#348) turned out ambiguous with 2.5 rather than a
    clean second source, and the docstrings say so rather than padding the
    count. If this test needs updating, the docstrings need updating too —
    that's the point of pinning it, not a number to raise silently."""
    entries = _entries()
    minimum_by_mode = {"2.5": 2, "2.4": 1, "1.4": 2}
    for mode, minimum in minimum_by_mode.items():
        scored = [e for e in entries if e.get("mast_mode") == mode]
        assert len(scored) == minimum, (
            f"Mode {mode} has {len(scored)} scorable entries, docstrings "
            f"claim {minimum} — update both together"
        )
        sources = {e["source_url"] for e in scored}
        assert len(sources) == len(scored), f"Mode {mode} reuses a source_url across entries"


def test_every_step_has_index_and_action() -> None:
    for entry in _entries():
        for step in entry["steps"]:
            assert "index" in step
            assert "action" in step


def test_provenance_values_are_unique() -> None:
    entries = _entries()
    provenances = [e["provenance"] for e in entries]
    assert len(provenances) == len(set(provenances)), "Duplicate provenance value found"
