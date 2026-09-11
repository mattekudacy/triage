"""
scripts/classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nine-block precision / recall report for RulesClassifier.

  Block 1 — Regression
      In-corpus positive examples from test_classifier_rules.py.
      Tautological by construction; the regexes were written against these
      strings.  A miss here means a regression, not a new gap.

  Block 2 — False-positive resistance
      Near-miss strings that must NOT fire a rule.  A false positive is worse
      than UNKNOWN: it routes to the wrong recovery strategy.

  Block 3 — Corpus A (second regression suite, NOT held-out)
      Real exceptions from json/asyncio/httpx/pydantic + SDK strings
      transcribed from published formats.  Corpus A was used to guide the v0.25
      pattern fixes, so 100% there reflects tuning, not generalization.

  Block 4 — Corpus B (training data after v0.26 fixes, 100% = 20/20)
      Assembled from sources not seen when writing the v0.25 patterns:
      botocore, google-genai, aiohttp, requests/urllib3, and structural
      phrasings that differ from corpus A.  Corpus B's misses guided the
      v0.26 fixes — it is now training data, same status as corpus A. Its
      last 2 misses (ServerConnectionError inactivity timeout, "Tool X is
      not registered") were closed in the v1.1 pattern pass.

  Block 5 — Corpus C (training data as of v1.1, 100% = 27/27)
      Sources disjoint from A and B: azure-core, Mistral, Cohere, Groq,
      LiteLLM, Vertex AI (aiplatform SDK), LlamaIndex, and novel phrasings.
      Was genuinely held-out through v1.0 (52% recall, 100% precision,
      scored once). v1.1 tuned rules.py directly against its 13 misses —
      that is what converts a corpus to training data, so 100% here now
      reflects tuning, not generalization. Corpus D (blocks 7-8) is the
      current held-out measurement.

  Block 6 — Corpus C recall by failure type (retained for history)
      Kept for continuity with the pre-v1.1 measurement. Not a held-out
      number as of v1.1 — see the caveat on block 5.

  Block 7 — Corpus D (genuine held-out, scored once after the v1.1 tuning
      pass, before any further rules.py edits)
      Sources disjoint from A, B, and C: huggingface_hub, Ollama, OpenRouter,
      Model Context Protocol (MCP), CrewAI, Semantic Kernel, and novel
      phrasings chosen to stress the boundaries of the v1.1 patterns.
      100% precision — the one misroute this scoring pass found (CrewAI's
      OutputParserError colliding with LlamaIndex's same-named exception)
      was fixed as a precision bug, not as recall tuning; see
      test_output_parser_error_exception_type_alone_does_not_fire_schema.
      Do NOT tune rules.py against D's misses — that converts D to training
      data the same way it happened to C. Generate corpus E instead.

  Block 8 — Corpus D recall by failure type
      Split the same way as block 6. This is the number that answers
      whether the v1.1 tuning pass generalized: routing-sensitive recall on
      fresh sources is 1/12 = 8%, statistically unchanged from corpus C's
      pre-tuning 1/12 = 8%. The v1.1 patterns were narrow string literals
      keyed close to corpus C's exact phrasings and did not transfer to new
      SDKs' wording. Self-healing recall held at 86%, as expected — that
      group clusters around a small, largely SDK-independent vocabulary
      (HTTP codes, "timeout", "rate limit") that routing-sensitive failures
      do not share.

Run:
    PYTHONPATH=. .venv/bin/python scripts/classifier_accuracy.py
"""

from __future__ import annotations

import collections
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

CORPUS_A_PATH = Path("tests/data/error_corpus_a.json")
CORPUS_B_PATH = Path("tests/data/error_corpus_b.json")
CORPUS_C_PATH = Path("tests/data/error_corpus_c.json")
CORPUS_D_PATH = Path("tests/data/error_corpus_d.json")


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


def _score_corpus(path: Path, missing_msg: str) -> tuple[int, int, list[str]]:
    if not path.exists():
        return 0, 0, [f"  {missing_msg}"]
    entries = json.loads(path.read_text())
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


# Types whose recovery works without knowing the failure type: a bare retry loop
# heals them, so correct classification adds nothing over blind retry.
SELF_HEALING = ("external_fault", "timeout")
# Types whose recovery only works when the strategy receives the matching hint.
# These are the types the library exists to get right.
ROUTING_SENSITIVE = ("wrong_tool_called", "schema_mismatch")


def _score_by_type(path: Path) -> dict[str, tuple[int, int]]:
    """Return {label: (hits, total)} for one corpus."""
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())
    total: collections.Counter[str] = collections.Counter()
    hits: collections.Counter[str] = collections.Counter()
    for entry in entries:
        got = _classify(entry["error"], entry.get("exception_type"))
        label = entry["label"]
        total[label] += 1
        if got.value == label:
            hits[label] += 1
    return {label: (hits[label], total[label]) for label in sorted(total)}


def _group_recall(by_type: dict[str, tuple[int, int]], labels: tuple[str, ...]) -> tuple[int, int]:
    hits = sum(by_type.get(label, (0, 0))[0] for label in labels)
    total = sum(by_type.get(label, (0, 0))[1] for label in labels)
    return hits, total


def main() -> None:
    reg_ok, reg_total, reg_fails = _score_regression()
    fp_ok, fp_total, fp_fails = _score_fp_resistance()
    a_ok, a_total, a_fails = _score_corpus(
        CORPUS_A_PATH, "Corpus A not found — run scripts/gen_error_corpus.py"
    )
    b_ok, b_total, b_fails = _score_corpus(
        CORPUS_B_PATH, "Corpus B not found — run scripts/gen_error_corpus_b.py"
    )
    c_ok, c_total, c_fails = _score_corpus(
        CORPUS_C_PATH, "Corpus C not found — run scripts/gen_error_corpus_c.py"
    )
    d_ok, d_total, d_fails = _score_corpus(
        CORPUS_D_PATH, "Corpus D not found — run scripts/gen_error_corpus_d.py"
    )

    print("RulesClassifier accuracy report")
    print("=" * 65)
    print()

    # Block 1
    print(f"Block 1 — Regression  ({reg_ok}/{reg_total} = {reg_ok / reg_total:.0%})")
    print("  In-corpus positives from test_classifier_rules.py.")
    print("  100% expected — tautological. A miss = regression.")
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
    if a_total > 0:
        print(f"Block 3 — Corpus A regression  ({a_ok}/{a_total} = {a_ok / a_total:.0%})")
        print("  CAVEAT: corpus A guided the v0.25 fixes — it is tuning data.")
        print("  100% expected here. A miss = regression.")
    else:
        print("Block 3 — Corpus A regression  [SKIPPED]")
    if a_fails:
        print("\n".join(a_fails))
    print()

    # Block 4
    if b_total > 0:
        ratio = f"{b_ok}/{b_total} = {b_ok / b_total:.0%}"
        print(f"Block 4 — Corpus B (training data after v0.26)  ({ratio})")
        print("  Sources not seen when writing v0.25 patterns: botocore,")
        print("  google-genai, aiohttp, requests/urllib3, novel phrasings.")
        print("  Corpus B's misses guided the v0.26 fixes — it is now")
        print("  training data; reflects tuning, not generalization. Its")
        print("  last 2 misses (90% -> 100%) were closed in the v1.1 pass.")
        if b_fails:
            print("  Remaining misses (targets for next release):")
            print("\n".join(b_fails))
    else:
        print("Block 4 — Corpus B  [SKIPPED]")
    print()

    # Block 5
    if c_total > 0:
        ratio = f"{c_ok}/{c_total} = {c_ok / c_total:.0%}"
        print(f"Block 5 — Corpus C (training data as of v1.1)  ({ratio})")
        print("  Sources disjoint from A and B: azure-core, Mistral, Cohere,")
        print("  Groq, LiteLLM, Vertex AI (aiplatform SDK), LlamaIndex,")
        print("  and novel phrasings. Genuinely held-out through v1.0 (52%")
        print("  recall, 100% precision). v1.1 tuned rules.py directly")
        print("  against C's 13 misses — 100% here now reflects tuning, not")
        print("  generalization. Corpus D (blocks 7-8) is current held-out.")
        if c_fails:
            print("  Remaining misses:")
            print("\n".join(c_fails))
    else:
        print("Block 5 — Corpus C  [SKIPPED]")
    print()

    # Block 6 — per-type breakdown of corpus C (history only, see block 5 caveat)
    by_type_c = _score_by_type(CORPUS_C_PATH)
    if by_type_c:
        print("Block 6 — Corpus C recall by failure type (history — not held-out)")
        for label, (hits, total) in by_type_c.items():
            group = ""
            if label in SELF_HEALING:
                group = "  (self-healing)"
            elif label in ROUTING_SENSITIVE:
                group = "  (routing-sensitive)"
            print(f"    {label:20} {hits}/{total} = {hits / total:3.0%}{group}")
        print()

    # Block 7
    if d_total > 0:
        ratio = f"{d_ok}/{d_total} = {d_ok / d_total:.0%}"
        print(f"Block 7 — Corpus D (genuine held-out)  ({ratio})")
        print("  Sources disjoint from A, B, and C: huggingface_hub, Ollama,")
        print("  OpenRouter, MCP, CrewAI, Semantic Kernel, and novel phrasings")
        print("  stress-testing the v1.1 patterns' boundaries. Scored once")
        print("  after the v1.1 tuning pass, before any further rules.py edit.")
        print("  Do NOT tune on these misses — that converts D to training")
        print("  data. Generate corpus E instead.")
        if d_fails:
            print("  Misses (held-out — do not use to guide fixes):")
            print("\n".join(d_fails))
    else:
        print("Block 7 — Corpus D  [SKIPPED]")
    print()

    # Block 8 — per-type breakdown of the held-out corpus
    by_type = _score_by_type(CORPUS_D_PATH)
    if by_type:
        print("Block 8 — Corpus D recall by failure type")
        print("  The block 7 aggregate averages two groups with opposite value.")
        print()
        for label, (hits, total) in by_type.items():
            group = ""
            if label in SELF_HEALING:
                group = "  (self-healing — any retry fixes it)"
            elif label in ROUTING_SENSITIVE:
                group = "  (routing-sensitive — needs the right hint)"
            print(f"    {label:20} {hits}/{total} = {hits / total:3.0%}{group}")
        print()

        sh_hits, sh_total = _group_recall(by_type, SELF_HEALING)
        rs_hits, rs_total = _group_recall(by_type, ROUTING_SENSITIVE)
        if sh_total:
            print(
                f"    self-healing types    {sh_hits}/{sh_total} = {sh_hits / sh_total:3.0%}"
                "  — classification buys nothing here"
            )
        if rs_total:
            print(
                f"    routing-sensitive     {rs_hits}/{rs_total} = {rs_hits / rs_total:3.0%}"
                "  — classification is the whole value"
            )
        print()
        print("  Compare to corpus C pre-v1.1 (1/12 = 8% routing-sensitive):")
        print("  the v1.1 tuning pass did not generalize past corpus C's own")
        print("  wording. See CHANGELOG and known-limitations.md.")
        print()

    print("─" * 65)
    print("Notes:")
    print("  PLAN_INCOMPLETE and CONTEXT_OVERFLOW: RulesClassifier returns UNKNOWN")
    print("  by design — semantic types, no pattern rules.")
    print("  CONSTRAINT_IGNORED: depends on RulesClassifier(constraints=[...]).")
    print("  LOOP_DETECTED: requires multi-step trajectory; not in single-step corpus.")


if __name__ == "__main__":
    main()
