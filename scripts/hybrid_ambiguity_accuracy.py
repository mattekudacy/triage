"""
scripts/hybrid_ambiguity_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measures a mechanism corpus D could only demonstrate with n=1: how often does
HybridClassifier overturn a correct, conservative RulesClassifier UNKNOWN into
a confident wrong guess?

Background — see README.md's "Does LLMClassifier/HybridClassifier actually
close the gap?" and docs/known-limitations.md's "LLMClassifier/HybridClassifier
close the recall gap, but not the precision gap". Corpus D's one true-`unknown`
entry (a genuinely out-of-taxonomy IAM string) was correctly classified UNKNOWN
by RulesClassifier — and HybridClassifier overturned it into a wrong concrete
guess in every LLM-involving run. That's a confirmed mechanism:
HybridClassifier.classify() is `if rules_result is not UNKNOWN: return it; else
ask the LLM` — unconditional, with no way to tell "rules doesn't recognize this
wording, but a real answer exists" from "this genuinely has no answer." n=1 is
not a measured rate. This script scores tests/data/error_corpus_ambiguous.json
(16 entries: 12 genuinely out-of-taxonomy, 4 real-but-obliquely-phrased) to get
one.

Two questions, two groups:

  unknown_labeled (12) — RulesClassifier correctly says UNKNOWN for all of
  these (verified at corpus-build time; see gen_error_corpus_ambiguous.py).
  HybridClassifier escalates every one to the LLM. The OVERRIDE RATE is how
  many of those 12 come back as something other than UNKNOWN — every one of
  those is a correct, safe answer turned into a wrong, confident one.

  tricky_but_classifiable (4) — real FailureTypes, phrased obliquely enough
  that RulesClassifier also can't match them (same escalation). Recall here
  answers the other half of the question: does caution about the ambiguous
  group cost recall on genuinely answerable-but-awkwardly-phrased failures,
  or are these independent?

Does NOT touch rules.py or any corpus file — read-only, same discipline as
llm_classifier_accuracy.py. NOT a frozen benchmark: this corpus is designed
to grow over time (see its module docstring), and LLM results are
non-deterministic run to run. No CI-enforced floor.

Requires an LLM backend — open by default, no key needed. Same
backend-resolution rules as llm_classifier_accuracy.py: with nothing
configured this talks to a local Ollama server; set an Anthropic credential
or TRIAGE_LLM_BASE_URL to use something else. See that script's module
docstring and docs/concepts/classifiers.md's "Open by default" note.

    ollama pull llama3.2   # once
    PYTHONPATH=. python scripts/hybrid_ambiguity_accuracy.py

    ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/hybrid_ambiguity_accuracy.py

    TRIAGE_LLM_BASE_URL=https://ollama.com/v1 \\
    TRIAGE_LLM_API_KEY=... TRIAGE_LLM_MODEL=gpt-oss:120b-cloud TRIAGE_LLM_MAX_TOKENS=500 \\
    PYTHONPATH=. python scripts/hybrid_ambiguity_accuracy.py

Run:
    PYTHONPATH=. .venv/bin/python scripts/hybrid_ambiguity_accuracy.py \\
        [--model MODEL] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

CORPUS_PATH = Path("tests/data/error_corpus_ambiguous.json")
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _resolve_backend(cli_model: str | None) -> tuple[str | None, str]:
    """See llm_classifier_accuracy.py's version for the full rationale —
    open-by-default: local Ollama unless the caller explicitly configured
    an Anthropic credential or a TRIAGE_LLM_BASE_URL."""
    base_url = os.environ.get("TRIAGE_LLM_BASE_URL")
    env_model = os.environ.get("TRIAGE_LLM_MODEL")
    has_anthropic_key = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("TRIAGE_LLM_API_KEY")
    )
    if base_url:
        return base_url, cli_model or env_model or _DEFAULT_ANTHROPIC_MODEL
    if has_anthropic_key:
        return None, cli_model or env_model or _DEFAULT_ANTHROPIC_MODEL
    return _DEFAULT_OLLAMA_BASE_URL, cli_model or env_model or _DEFAULT_OLLAMA_MODEL


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


def _check_backend_installed(base_url: str | None) -> None:
    """See llm_classifier_accuracy.py — same check, duplicated per this
    repo's convention of self-contained scripts."""
    pkg, extra = ("openai", "openai") if base_url else ("anthropic", "anthropic")
    try:
        __import__(pkg)
    except ImportError:
        reason = f"base_url={base_url!r}" if base_url else "no base_url — Anthropic backend"
        raise SystemExit(
            f"Missing dependency: '{pkg}' is not installed ({reason}).\n"
            f"  pip install triage-agent[{extra}]"
        ) from None


def _sanity_check(clf: LLMClassifier) -> None:
    """See llm_classifier_accuracy.py's version for the full rationale."""
    t = Trajectory()
    t.append(Step(index=0, action="test", error="HTTP 429 Too Many Requests"))
    got = clf.classify(t, "task")
    if got != FailureType.EXTERNAL_FAULT:
        print(
            "SANITY CHECK FAILED: LLMClassifier returned "
            f"{got.value!r} for an unambiguous '429 Too Many Requests' string "
            "(expected 'external_fault'). classify() returns UNKNOWN both on "
            "ANY error (network/auth/bad model/unreachable base_url) AND on a "
            "successful call with empty content (reasoning-model token budget "
            "too small — try --max-tokens 500). Refusing to score against a "
            "classifier that fails this trivially.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name (default: TRIAGE_LLM_MODEL env var, else "
            f"{_DEFAULT_OLLAMA_MODEL!r} on local Ollama unless an Anthropic "
            f"credential is set with no base_url, then {_DEFAULT_ANTHROPIC_MODEL!r} "
            "— see _resolve_backend())"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Output token budget — see LLMClassifier docstring. Pass a few "
        "hundred for reasoning models (default: TRIAGE_LLM_MAX_TOKENS env var, "
        "else 32).",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found: {CORPUS_PATH} — run scripts/gen_error_corpus_ambiguous.py"
        )
    entries = json.loads(CORPUS_PATH.read_text())

    base_url, model = _resolve_backend(args.model)
    _check_backend_installed(base_url)

    llm = LLMClassifier(model=model, base_url=base_url, max_tokens=args.max_tokens)
    print(f"Model: {model}")
    print(f"Max tokens: {llm._max_tokens}")
    print(f"Base URL: {base_url or '(Anthropic default client)'}")
    print("Running sanity check...", end=" ", flush=True)
    _sanity_check(llm)
    print("ok\n")

    hybrid = HybridClassifier(
        llm=LLMClassifier(model=model, base_url=base_url, max_tokens=args.max_tokens)
    )

    print("=" * 65)
    print(f"{len(entries)} entries — RulesClassifier says UNKNOWN for all of")
    print("them by construction (verified at corpus-build time), so every")
    print("entry is escalated to the LLM under HybridClassifier. Read-only;")
    print("nothing here touches rules.py or a corpus file.")
    print("=" * 65)
    print()

    unknown_group = [e for e in entries if e["group"] == "unknown_labeled"]
    tricky_group = [e for e in entries if e["group"] == "tricky_but_classifiable"]

    # ── unknown_labeled: override rate ──────────────────────────────────────
    print("── unknown_labeled — override rate ─────────────────────────────")
    print("  RulesClassifier correctly says UNKNOWN for every one of these.")
    print("  HybridClassifier escalates all of them to the LLM. Any answer")
    print("  other than 'unknown' here is a correct, safe verdict turned")
    print("  into a confident wrong one.")
    print()
    overridden = []
    kept_unknown = 0
    for entry in unknown_group:
        got = hybrid.classify(_trajectory_for(entry), "task")
        if got == FailureType.UNKNOWN:
            kept_unknown += 1
        else:
            overridden.append((entry, got.value))
    total_u = len(unknown_group)
    print(f"  Correctly kept UNKNOWN: {kept_unknown}/{total_u}")
    print(
        f"  Overridden into a wrong guess: {len(overridden)}/{total_u}"
        f"  <- OVERRIDE RATE = {len(overridden) / total_u:.0%}"
        if total_u
        else ""
    )
    if overridden:
        print("\n  Overridden entries:")
        for entry, got in overridden:
            print(f"    got={got:20} true=unknown  {entry['error'][:55]!r}")
    print()

    # ── tricky_but_classifiable: recall despite oblique phrasing ────────────
    print("── tricky_but_classifiable — recall on oblique-but-real failures ")
    print("  RulesClassifier also can't match these (deliberately, by")
    print("  phrasing). Does escalating to the LLM recover them anyway?")
    print()
    hits = 0
    for entry in tricky_group:
        got = hybrid.classify(_trajectory_for(entry), "task")
        ok = got.value == entry["label"]
        hits += ok
        flag = "OK" if ok else f"MISS (got {got.value})"
        print(f"  {flag:22} exp={entry['label']:20} {entry['error'][:45]!r}")
    total_t = len(tricky_group)
    print(f"\n  Recall: {hits}/{total_t}" + (f" = {hits / total_t:.0%}" if total_t else ""))
    print()

    print("─" * 65)
    print("Summary:")
    override_pct = f" = {len(overridden) / total_u:.0%}" if total_u else ""
    override_line = f"  Override rate (unknown_labeled, lower better): {len(overridden)}/{total_u}"
    print(override_line + override_pct)
    recall_pct = f" = {hits / total_t:.0%}" if total_t else ""
    recall_line = f"  Recall (tricky_but_classifiable, higher is better): {hits}/{total_t}"
    print(recall_line + recall_pct)
    print()
    print("Not a frozen benchmark — LLM results vary run to run, and this")
    print("corpus is designed to grow over time (see its module docstring).")
    print("Compare against corpus D's single n=1 data point, not a fixed floor.")


if __name__ == "__main__":
    main()
