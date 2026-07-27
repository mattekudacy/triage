"""
scripts/classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three-block precision / recall report for RulesClassifier.

  Block 1 — Regression
      In-corpus positive examples from test_classifier_rules.py.
      Tautological by construction; the regexes were written against these
      strings.  A score < 100% here means a regression.  Not evidence of
      generalization.

  Block 2 — False-positive resistance
      Near-miss strings that must NOT fire a rule.  A false positive is worse
      than UNKNOWN: it routes to the wrong recovery strategy.

  Block 3 — Held-out accuracy (corpus A)
      Real exceptions provoked from installed SDKs (json, asyncio, httpx,
      pydantic, openai, anthropic, langchain, langgraph).  These strings were
      NOT used to write the rules.  This is the number to improve against.

Run:
    PYTHONPATH=. .venv/bin/python scripts/classifier_accuracy.py
"""

from __future__ import annotations

import json
from pathlib import Path

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def _classify(error: str | None, exception_type: str | None = None) -> FailureType:
    t = Trajectory()
    t.append(Step(index=0, action="test", error=error, exception_type=exception_type))
    return RulesClassifier().classify(t, "task")


# ── Block 1: Regression positives ─────────────────────────────────────────────

REGRESSION: list[tuple[str, FailureType]] = [
    ("no tool named calculator", FailureType.WRONG_TOOL_CALLED),
    ("Tool 'bar' not found", FailureType.WRONG_TOOL_CALLED),
    ("tool_not_found: the requested tool does not exist", FailureType.WRONG_TOOL_CALLED),
    ("function 'send_email' does not exist", FailureType.WRONG_TOOL_CALLED),
    ("validation error: field required", FailureType.SCHEMA_MISMATCH),
    ("json parse failed", FailureType.SCHEMA_MISMATCH),
    ("invalid json in response body", FailureType.SCHEMA_MISMATCH),
    ("unexpected token '{' in json", FailureType.SCHEMA_MISMATCH),
    ("HTTP 429 Too Many Requests", FailureType.EXTERNAL_FAULT),
    ("500 Internal Server Error", FailureType.EXTERNAL_FAULT),
    ("502 Bad Gateway", FailureType.EXTERNAL_FAULT),
    ("503 Service Unavailable", FailureType.EXTERNAL_FAULT),
    ("rate limited, status 429", FailureType.EXTERNAL_FAULT),
    ("asyncio.TimeoutError: timeout", FailureType.TIMEOUT),
    ("request timed out after 30s", FailureType.TIMEOUT),
    ("deadline exceeded", FailureType.TIMEOUT),
    ("time limit reached", FailureType.TIMEOUT),
]

# ── Block 2: False-positive guards ────────────────────────────────────────────
# (error_string, type_that_must_NOT_fire)

FALSE_POSITIVES: list[tuple[str, FailureType]] = [
    ("expected 500 items but got 42", FailureType.EXTERNAL_FAULT),
    ("processed 503 records successfully", FailureType.EXTERNAL_FAULT),
    ("returned 429 results", FailureType.EXTERNAL_FAULT),
    ("step 500 completed", FailureType.EXTERNAL_FAULT),
    ("line 503: syntax error", FailureType.EXTERNAL_FAULT),
    ("expected 200 records", FailureType.EXTERNAL_FAULT),
    ("tooltip not found in DOM", FailureType.WRONG_TOOL_CALLED),
    ("found 3 tools available", FailureType.WRONG_TOOL_CALLED),
    ("toolbox is empty", FailureType.WRONG_TOOL_CALLED),
    ("retool configuration loaded", FailureType.WRONG_TOOL_CALLED),
    ("index out of range", FailureType.SCHEMA_MISMATCH),
    ("connection refused", FailureType.WRONG_TOOL_CALLED),
]

# ── Block 3: Held-out (corpus A) ───────────────────────────────────────────────

CORPUS_PATH = Path("tests/data/error_corpus_a.json")


def _run_block(label: str, note: str) -> None:
    print(f"── {label} {'─' * max(0, 60 - len(label))}")
    print(f"   {note}")
    print()


def _score_regression() -> tuple[int, int, list[str]]:
    ok = 0
    fails = []
    for error, expected in REGRESSION:
        got = _classify(error)
        if got == expected:
            ok += 1
        else:
            fails.append(f"  MISS exp={expected.value} got={got.value} {error!r}")
    return ok, len(REGRESSION), fails


def _score_fp_resistance() -> tuple[int, int, list[str]]:
    ok = 0
    fails = []
    for error, forbidden in FALSE_POSITIVES:
        got = _classify(error)
        if got != forbidden:
            ok += 1
        else:
            fails.append(f"  FP   type={forbidden.value} fired for {error!r}")
    return ok, len(FALSE_POSITIVES), fails


def _score_held_out() -> tuple[int, int, list[str]]:
    if not CORPUS_PATH.exists():
        return 0, 0, [f"  Corpus not found: {CORPUS_PATH} — run scripts/gen_error_corpus.py"]
    entries = json.loads(CORPUS_PATH.read_text())
    ok = 0
    fails = []
    for entry in entries:
        got = _classify(entry["error"], entry.get("exception_type"))
        exp = entry["label"]
        if got.value == exp:
            ok += 1
        else:
            fails.append(
                f"  MISS exp={exp:16} got={got.value:16}"
                f" [{entry.get('exception_type', '')}] {entry['error'][:50]!r}"
            )
    return ok, len(entries), fails


def main() -> None:
    reg_ok, reg_total, reg_fails = _score_regression()
    fp_ok, fp_total, fp_fails = _score_fp_resistance()
    ho_ok, ho_total, ho_fails = _score_held_out()

    print("RulesClassifier accuracy report")
    print("=" * 65)
    print()

    # Block 1
    print(f"Block 1 — Regression  ({reg_ok}/{reg_total} = {reg_ok / reg_total:.0%})")
    print("  In-corpus positives from test_classifier_rules.py.")
    print("  100% expected — this is tautological. A miss = regression.")
    if reg_fails:
        print("\n".join(reg_fails))
    print()

    # Block 2
    print(f"Block 2 — False-positive resistance  ({fp_ok}/{fp_total} = {fp_ok / fp_total:.0%})")
    print("  Near-miss strings that must NOT fire a rule.")
    print("  A FP routes to the wrong strategy — worse than UNKNOWN.")
    if fp_fails:
        print("\n".join(fp_fails))
    print()

    # Block 3
    if ho_total > 0:
        ratio = f"{ho_ok}/{ho_total} = {ho_ok / ho_total:.0%}"
        print(f"Block 3 — Held-out accuracy (corpus A)  ({ratio})")
        print("  Real exceptions from json/asyncio/httpx/pydantic/openai/")
        print("  anthropic/langchain/langgraph. NOT used to write the rules.")
        print("  This is the number to improve against across releases.")
    else:
        print("Block 3 — Held-out accuracy (corpus A)  [SKIPPED]")
    if ho_fails:
        print("\n".join(ho_fails))
    print()

    print("─" * 65)
    print("Notes:")
    print("  PLAN_INCOMPLETE and CONTEXT_OVERFLOW: RulesClassifier returns UNKNOWN")
    print("  by design — semantic types, no pattern rules.")
    print("  CONSTRAINT_IGNORED: depends on RulesClassifier(constraints=[...]).")
    print("  LOOP_DETECTED: requires multi-step trajectory; not in single-step corpus.")


if __name__ == "__main__":
    main()
