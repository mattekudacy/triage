"""
scripts/classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-type precision / recall report for RulesClassifier.

The labeled dataset is extracted directly from test_classifier_rules.py's
parametrized corpora and named positive/negative tests — the same inputs
pytest already validates.  Running this script turns that test corpus into
a scored report without any new test infrastructure.

Run:
    PYTHONPATH=. .venv/bin/python scripts/classifier_accuracy.py

Output:
    Per-type table of TP / FP / FN, precision, recall, and F1, plus
    a macro-averaged F1 at the bottom.  Results are deterministic (no LLM
    calls) and should be re-run after any change to RulesClassifier regexes.
"""

from __future__ import annotations

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory


def _t(error: str | None = None, llm_output: str | None = None) -> Trajectory:
    t = Trajectory()
    t.append(Step(index=0, action="test", error=error, llm_output=llm_output))
    return t


# ---------------------------------------------------------------------------
# Labeled dataset
# Each entry: (trajectory, expected_label)
# Labels come from the parametrized corpora and explicit named tests in
# tests/test_classifier_rules.py.  Negative examples use FailureType.UNKNOWN.
# ---------------------------------------------------------------------------

clf = RulesClassifier()

DATASET: list[tuple[Trajectory, FailureType]] = [
    # ── EXTERNAL_FAULT true positives (from test_external_fault_true_positive_corpus) ──
    (_t("HTTP 429: rate limited"), FailureType.EXTERNAL_FAULT),
    (_t("status code 500"), FailureType.EXTERNAL_FAULT),
    (_t("received 503 from upstream"), FailureType.EXTERNAL_FAULT),
    (_t("server returned 502 bad gateway"), FailureType.EXTERNAL_FAULT),
    (_t("upstream error 429"), FailureType.EXTERNAL_FAULT),
    (_t("got 500 from remote"), FailureType.EXTERNAL_FAULT),
    # individual named tests
    (_t("HTTP 429 Too Many Requests"), FailureType.EXTERNAL_FAULT),
    (_t("500 Internal Server Error"), FailureType.EXTERNAL_FAULT),
    (_t("502 Bad Gateway"), FailureType.EXTERNAL_FAULT),
    (_t("503 Service Unavailable"), FailureType.EXTERNAL_FAULT),
    (_t("rate limited, status 429"), FailureType.EXTERNAL_FAULT),
    # ── EXTERNAL_FAULT false-positive guards (expect UNKNOWN) ──
    (_t("expected 500 items but got 42"), FailureType.UNKNOWN),
    (_t("processed 503 records successfully"), FailureType.UNKNOWN),
    (_t("returned 429 results"), FailureType.UNKNOWN),
    (_t("502 bytes written"), FailureType.UNKNOWN),
    (_t("step 500 completed"), FailureType.UNKNOWN),
    (_t("line 503: syntax error"), FailureType.UNKNOWN),
    (_t("error in row 429"), FailureType.UNKNOWN),
    (_t("expected 200 records"), FailureType.UNKNOWN),
    # ── WRONG_TOOL_CALLED true positives ──
    (_t("no tool named calculator"), FailureType.WRONG_TOOL_CALLED),
    (_t("Tool 'bar' not found"), FailureType.WRONG_TOOL_CALLED),
    (_t("NO TOOL NAMED foo"), FailureType.WRONG_TOOL_CALLED),
    (_t("tool_not_found: the requested tool does not exist"), FailureType.WRONG_TOOL_CALLED),
    (_t("tool foo not found"), FailureType.WRONG_TOOL_CALLED),
    # ── WRONG_TOOL_CALLED false-positive guards ──
    (_t("tooltip not found in DOM"), FailureType.UNKNOWN),
    (_t("found 3 tools available"), FailureType.UNKNOWN),
    (_t("initialize tool chain"), FailureType.UNKNOWN),
    (_t("toolbox is empty"), FailureType.UNKNOWN),
    (_t("retool configuration loaded"), FailureType.UNKNOWN),
    (_t("connection refused"), FailureType.UNKNOWN),
    # ── SCHEMA_MISMATCH true positives ──
    (_t("validation error: field required"), FailureType.SCHEMA_MISMATCH),
    (_t("JSONDecodeError: Expecting value at line 1"), FailureType.SCHEMA_MISMATCH),
    (_t("json parse failed"), FailureType.SCHEMA_MISMATCH),
    (_t("validation error: expected string at field 'name'"), FailureType.SCHEMA_MISMATCH),
    (_t("jsondecodeerror at line 1"), FailureType.SCHEMA_MISMATCH),
    (_t("invalid json in response body"), FailureType.SCHEMA_MISMATCH),
    (_t("unexpected token '{' in json"), FailureType.SCHEMA_MISMATCH),
    (_t("failed to json parse the response"), FailureType.SCHEMA_MISMATCH),
    # ── SCHEMA_MISMATCH false-positive guard ──
    (_t("index out of range"), FailureType.UNKNOWN),
    # ── TIMEOUT true positives ──
    (_t("asyncio.TimeoutError: timeout"), FailureType.TIMEOUT),
    (_t("request timed out after 30s"), FailureType.TIMEOUT),
    (_t("deadline exceeded"), FailureType.TIMEOUT),
    (_t("time limit reached"), FailureType.TIMEOUT),
    (_t("operation timed out after 30s"), FailureType.TIMEOUT),
    (_t("deadline exceeded for request"), FailureType.TIMEOUT),
    (_t("async time limit reached"), FailureType.TIMEOUT),
    (_t("timed out waiting for response"), FailureType.TIMEOUT),
    # ── TIMEOUT false-positive guard ──
    (_t("connection refused"), FailureType.UNKNOWN),
    # ── UNKNOWN true positives (genuinely ambiguous errors) ──
    (_t("something completely unrelated"), FailureType.UNKNOWN),
    (_t("permission denied"), FailureType.UNKNOWN),
    (_t("file not found"), FailureType.UNKNOWN),
]

# PLAN_INCOMPLETE and CONTEXT_OVERFLOW are intentionally absent: they require
# semantic understanding and always return UNKNOWN from RulesClassifier per
# design (see CLAUDE.md "RulesClassifier scope").


def main() -> None:
    # Per-type tallies
    tp: dict[FailureType, int] = {ft: 0 for ft in FailureType}
    fp: dict[FailureType, int] = {ft: 0 for ft in FailureType}
    fn: dict[FailureType, int] = {ft: 0 for ft in FailureType}

    for traj, expected in DATASET:
        predicted = clf.classify(traj, "task")
        if predicted == expected:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    # Only report types that appear in the dataset
    active = {ft for _, ft in DATASET}

    print("RulesClassifier — per-type precision / recall")
    print(f"Dataset: {len(DATASET)} labeled examples")
    print()
    print(f"{'Type':<25} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>8}  {'Recall':>7}  {'F1':>6}")
    print("─" * 70)

    f1s = []
    for ft in FailureType:
        if ft not in active:
            continue
        t, f, n = tp[ft], fp[ft], fn[ft]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        print(
            f"{ft.value:<25} {t:>4} {f:>4} {n:>4}  {precision:>10.1%}  {recall:>7.1%}  {f1:>6.3f}"
        )

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print("─" * 70)
    print(f"{'macro F1':<25} {'':>4} {'':>4} {'':>4}  {'':>10}  {'':>7}  {macro_f1:>6.3f}")
    print()
    print("Notes:")
    print("  PLAN_INCOMPLETE and CONTEXT_OVERFLOW are not scored — RulesClassifier")
    print("  returns UNKNOWN for them by design (semantic types, no pattern rules).")
    print("  CONSTRAINT_IGNORED requires RulesClassifier(constraints=[...]) and is")
    print("  excluded here; precision/recall depend entirely on the constraints list.")


if __name__ == "__main__":
    main()
