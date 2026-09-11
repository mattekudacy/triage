"""
scripts/llm_classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests a claim made in README.md and docs/known-limitations.md: that
``LLMClassifier`` / ``HybridClassifier`` close the routing-sensitive gap
``RulesClassifier`` leaves open on held-out data. As of v1.1, corpus D shows
``RulesClassifier`` at 1/12 = 8% recall on WRONG_TOOL_CALLED + SCHEMA_MISMATCH
(the two types where classification actually changes the recovery outcome —
see scripts/classifier_accuracy.py blocks 7-8). The docs recommend semantic
classification as the mitigation. That recommendation has never been measured
against held-out data. This script measures it.

Scores the SAME corpus D entries with three classifiers, side by side:

  1. RulesClassifier            — the v1.1 baseline (free, zero API calls)
  2. LLMClassifier alone        — diagnostic upper bound: how well does raw
                                   semantic classification do on these strings?
  3. HybridClassifier(rules + llm) — the actually-recommended configuration:
                                   rules first (free), LLM only on rules' UNKNOWN

IMPORTANT — this does NOT touch rules.py or any corpus JSON file. It only
*reads* corpus D and classifies it. Re-run it as many times as you like: it
cannot "burn" corpus D as training data the way editing rules.py against its
misses would, because nothing here feeds back into rules.py.

``LLMClassifier.classify()`` swallows every exception (auth failures
included) and returns UNKNOWN — see its docstring. That means a broken API
key would silently produce a report that looks identical to "the LLM doesn't
help either," which is exactly the kind of misleading number this repo's
corpus discipline exists to prevent. This script runs a pre-flight sanity
check (one unambiguous classification) before scoring anything, and refuses
to print a report if the classifier can't get that one right.

Requires an LLM backend. Anthropic (default — picks up ANTHROPIC_API_KEY the
same way the Anthropic SDK always does; TRIAGE_LLM_API_KEY also works):
    ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/llm_classifier_accuracy.py

Any OpenAI-compatible endpoint (Groq, Ollama, OpenAI, ...):
    TRIAGE_LLM_BASE_URL=https://api.groq.com/openai/v1 \\
    TRIAGE_LLM_API_KEY=gsk_... TRIAGE_LLM_MODEL=llama-3.1-8b-instant \\
    PYTHONPATH=. python scripts/llm_classifier_accuracy.py

Model defaults to claude-haiku-4-5-20251001 (the model used elsewhere in this
repo's docs/examples) unless TRIAGE_LLM_MODEL or --model overrides it. Makes
one classification call per corpus entry per classifier under test (up to
~40 calls total against a 20-entry corpus) — cheap on a small model, but not
free; this is why it's a separate opt-in script from classifier_accuracy.py,
which makes zero API calls by design.

Run:
    PYTHONPATH=. .venv/bin/python scripts/llm_classifier_accuracy.py [--model MODEL]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier
from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

CORPUS_D_PATH = Path("tests/data/error_corpus_d.json")

# Same grouping as scripts/classifier_accuracy.py blocks 7-8.
SELF_HEALING = ("external_fault", "timeout")
ROUTING_SENSITIVE = ("wrong_tool_called", "schema_mismatch")

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _trajectory_for(entry: dict[str, Any]) -> Trajectory:
    t = Trajectory()
    t.append(
        Step(
            index=0,
            action="test",
            error=entry["error"],
            exception_type=entry.get("exception_type"),
        )
    )
    return t


def _check_backend_installed() -> None:
    """Check the optional dependency LLMClassifier needs is importable.

    classify() catches ImportError from a missing 'anthropic'/'openai' package
    the same as any other exception and returns UNKNOWN — see
    triage/classifier/llm.py. Left unchecked, that produces the exact same
    generic sanity-check failure as a bad API key, with no hint that the fix
    is `pip install`, not a credential. Check explicitly, upfront.
    """
    base_url = os.environ.get("TRIAGE_LLM_BASE_URL")
    pkg, extra = ("openai", "openai") if base_url else ("anthropic", "anthropic")
    try:
        __import__(pkg)
    except ImportError:
        reason = "TRIAGE_LLM_BASE_URL is set" if base_url else "no base_url set — Anthropic backend"
        raise SystemExit(
            f"Missing dependency: '{pkg}' is not installed ({reason}).\n"
            f"  pip install triage-agent[{extra}]"
        ) from None


def _sanity_check(clf: LLMClassifier) -> None:
    """One unambiguous classification before trusting anything else.

    classify() returns UNKNOWN on *any* error, including a missing/invalid
    API key or an unreachable base_url — see triage/classifier/llm.py. If
    this check itself comes back UNKNOWN, the classifier almost certainly
    isn't actually calling the model, and every number below would be
    silently wrong in the most misleading possible direction (looking like
    "the LLM doesn't help either"). Fail loudly instead of printing that.
    """
    t = Trajectory()
    t.append(Step(index=0, action="test", error="HTTP 429 Too Many Requests"))
    got = clf.classify(t, "task")
    if got != FailureType.EXTERNAL_FAULT:
        print(
            "SANITY CHECK FAILED: LLMClassifier returned "
            f"{got.value!r} for an unambiguous '429 Too Many Requests' string "
            "(expected 'external_fault').\n"
            "classify() returns UNKNOWN on ANY error — network, auth, bad "
            "model name, unreachable base_url — so this almost certainly means "
            "the classifier never successfully reached the model, not that the "
            "model is inaccurate. Refusing to score corpus D against a "
            "classifier that fails this trivially.\n\n"
            "Check: API key set and valid, --model / TRIAGE_LLM_MODEL correct, "
            "TRIAGE_LLM_BASE_URL reachable if set.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _score(
    clf: RulesClassifier | LLMClassifier | HybridClassifier,
    entries: list[dict[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """Return {label: (hits, total, unknowns)} — unknowns tracked separately
    from other misses so a UNKNOWN-heavy classifier is visibly distinguishable
    from one making confident wrong guesses (misroutes)."""
    total: collections.Counter[str] = collections.Counter()
    hits: collections.Counter[str] = collections.Counter()
    unknowns: collections.Counter[str] = collections.Counter()
    for entry in entries:
        got = clf.classify(_trajectory_for(entry), "task")
        label = entry["label"]
        total[label] += 1
        if got.value == label:
            hits[label] += 1
        elif got == FailureType.UNKNOWN:
            unknowns[label] += 1
    return {label: (hits[label], total[label], unknowns[label]) for label in sorted(total)}


def _group_totals(
    by_type: dict[str, tuple[int, int, int]], labels: tuple[str, ...]
) -> tuple[int, int]:
    hits = sum(by_type.get(label, (0, 0, 0))[0] for label in labels)
    total = sum(by_type.get(label, (0, 0, 0))[1] for label in labels)
    return hits, total


def _print_report(name: str, by_type: dict[str, tuple[int, int, int]]) -> None:
    print(f"── {name} " + "─" * max(0, 60 - len(name)))
    for label, (hits, total, unknowns) in by_type.items():
        misroutes = total - hits - unknowns
        group = ""
        if label in SELF_HEALING:
            group = "  (self-healing)"
        elif label in ROUTING_SENSITIVE:
            group = "  (routing-sensitive)"
        flag = f"  [{misroutes} misroute(s)]" if misroutes else ""
        print(f"  {label:20} {hits}/{total} = {hits / total:3.0%}{group}{flag}")
    sh_hits, sh_total = _group_totals(by_type, SELF_HEALING)
    rs_hits, rs_total = _group_totals(by_type, ROUTING_SENSITIVE)
    print()
    if sh_total:
        print(f"  self-healing        {sh_hits}/{sh_total} = {sh_hits / sh_total:3.0%}")
    if rs_total:
        print(f"  routing-sensitive    {rs_hits}/{rs_total} = {rs_hits / rs_total:3.0%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("TRIAGE_LLM_MODEL") or _DEFAULT_MODEL,
        help=f"Model name (default: TRIAGE_LLM_MODEL env var, else {_DEFAULT_MODEL!r})",
    )
    args = parser.parse_args()

    if not CORPUS_D_PATH.exists():
        raise SystemExit(f"Corpus D not found: {CORPUS_D_PATH} — run scripts/gen_error_corpus_d.py")
    entries = json.loads(CORPUS_D_PATH.read_text())

    _check_backend_installed()

    llm = LLMClassifier(model=args.model)
    print(f"Model: {args.model}")
    if os.environ.get("TRIAGE_LLM_BASE_URL"):
        print(f"Base URL: {os.environ['TRIAGE_LLM_BASE_URL']}")
    print("Running sanity check...", end=" ", flush=True)
    _sanity_check(llm)
    print("ok\n")

    print("=" * 65)
    print("Corpus D scored by three classifiers (20 entries, held-out — see")
    print("scripts/README.md's Corpus discipline). Nothing here edits rules.py")
    print("or any corpus file; safe to re-run.")
    print("=" * 65)
    print()

    rules_by_type = _score(RulesClassifier(), entries)
    _print_report("RulesClassifier (v1.1 baseline)", rules_by_type)

    llm_by_type = _score(llm, entries)
    _print_report(f"LLMClassifier alone ({args.model})", llm_by_type)

    hybrid = HybridClassifier(llm=LLMClassifier(model=args.model))
    hybrid_by_type = _score(hybrid, entries)
    _print_report("HybridClassifier (rules + LLM fallback — recommended config)", hybrid_by_type)

    print("─" * 65)
    print("Routing-sensitive recall (wrong_tool_called + schema_mismatch) —")
    print("the number that answers whether semantic classification closes")
    print("the gap RulesClassifier's v1.1 pattern-tuning pass could not:")
    print()
    for label, by_type in (
        ("RulesClassifier", rules_by_type),
        ("LLMClassifier alone", llm_by_type),
        ("HybridClassifier", hybrid_by_type),
    ):
        hits, total = _group_totals(by_type, ROUTING_SENSITIVE)
        pct = f"{hits}/{total} = {hits / total:3.0%}" if total else "n/a"
        print(f"  {label:32} {pct}")
    print()
    print("Read this together with scripts/classifier_accuracy.py blocks 7-8 —")
    print("same corpus, same grouping, RulesClassifier's row here should match")
    print("that script's corpus D routing-sensitive number exactly (1/12 = 8%")
    print("as of v1.1). If it doesn't, something has drifted; investigate before")
    print("trusting this script's LLM/Hybrid numbers.")


if __name__ == "__main__":
    main()
