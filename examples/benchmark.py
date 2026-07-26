"""
examples/benchmark.py
~~~~~~~~~~~~~~~~~~~~~
Classifier accuracy benchmark. Runs synthetic trajectories for each
FailureType through RulesClassifier (and optionally LLMClassifier /
HybridClassifier) and prints a report showing true-positive and
false-positive rates.

Run with:
    python examples/benchmark.py                 # RulesClassifier only
    ANTHROPIC_API_KEY=sk-ant-... python examples/benchmark.py --llm
    ANTHROPIC_API_KEY=sk-ant-... python examples/benchmark.py --hybrid
    ANTHROPIC_API_KEY=sk-ant-... python examples/benchmark.py --llm --hybrid

Output columns:
    EXPECTED   — the FailureType the trajectory is supposed to trigger
    GOT        — what the classifier returned
    PASS/FAIL  — match result

At the end, overall accuracy and per-type hit/miss counts are printed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType, Step
from triage.trajectory import Trajectory

# ── Test case definition ──────────────────────────────────────────────────────


@dataclass
class Case:
    label: str
    expected: FailureType
    steps: list[Step]
    task: str = "benchmark task"


def make_step(
    index: int = 0,
    tool_called: str | None = None,
    tool_input: dict | None = None,
    error: str | None = None,
    llm_output: str | None = None,
    tool_output: str | None = None,
) -> Step:
    return Step(
        index=index,
        action="benchmark step",
        tool_called=tool_called,
        tool_input=tool_input,
        error=error,
        llm_output=llm_output,
        tool_output=tool_output,
    )


# ── Test cases ────────────────────────────────────────────────────────────────

CASES: list[Case] = [
    # ── LOOP_DETECTED ──────────────────────────────────────────────────────────
    Case(
        label="loop: 3 identical tool+input steps",
        expected=FailureType.LOOP_DETECTED,
        steps=[
            make_step(0, tool_called="search", tool_input={"q": "Paris"}),
            make_step(1, tool_called="search", tool_input={"q": "Paris"}),
            make_step(2, tool_called="search", tool_input={"q": "Paris"}),
        ],
    ),
    Case(
        label="loop: 5 identical steps (loop_window=5)",
        expected=FailureType.LOOP_DETECTED,
        steps=[make_step(i, tool_called="fetch", tool_input={"url": "http://x"}) for i in range(5)],
    ),
    # ── NOT LOOP ───────────────────────────────────────────────────────────────
    Case(
        label="no-loop: only 2 identical steps",
        expected=FailureType.UNKNOWN,
        steps=[
            make_step(0, tool_called="search", tool_input={"q": "Paris"}),
            make_step(1, tool_called="search", tool_input={"q": "Paris"}),
        ],
    ),
    Case(
        label="no-loop: same tool, different inputs",
        expected=FailureType.UNKNOWN,
        steps=[
            make_step(0, tool_called="search", tool_input={"q": "Paris"}),
            make_step(1, tool_called="search", tool_input={"q": "London"}),
            make_step(2, tool_called="search", tool_input={"q": "Tokyo"}),
        ],
    ),
    # ── WRONG_TOOL_CALLED ──────────────────────────────────────────────────────
    Case(
        label="wrong-tool: 'no tool named X'",
        expected=FailureType.WRONG_TOOL_CALLED,
        steps=[make_step(0, error="no tool named nonexistent_calculator")],
    ),
    Case(
        label="wrong-tool: 'Tool X not found'",
        expected=FailureType.WRONG_TOOL_CALLED,
        steps=[make_step(0, error="Tool 'send_email' not found in manifest")],
    ),
    Case(
        label="wrong-tool: OpenAI tool_not_found code",
        expected=FailureType.WRONG_TOOL_CALLED,
        steps=[make_step(0, error="tool_not_found: the requested tool is not available")],
    ),
    Case(
        label="wrong-tool: Anthropic function does not exist",
        expected=FailureType.WRONG_TOOL_CALLED,
        steps=[make_step(0, error="function 'web_search' does not exist")],
    ),
    # ── NOT WRONG_TOOL ─────────────────────────────────────────────────────────
    Case(
        label="no-wrong-tool: generic error",
        expected=FailureType.UNKNOWN,
        steps=[make_step(0, error="connection refused")],
    ),
    # ── SCHEMA_MISMATCH ────────────────────────────────────────────────────────
    Case(
        label="schema: JSONDecodeError",
        expected=FailureType.SCHEMA_MISMATCH,
        steps=[make_step(0, error="JSONDecodeError: Expecting value at line 1 column 1")],
    ),
    Case(
        label="schema: validation error (pydantic-style)",
        expected=FailureType.SCHEMA_MISMATCH,
        steps=[make_step(0, error="validation error: field 'name' required")],
    ),
    Case(
        label="schema: json parse failed",
        expected=FailureType.SCHEMA_MISMATCH,
        steps=[make_step(0, error="json parse failed: unexpected end of input")],
    ),
    Case(
        label="schema: invalid json",
        expected=FailureType.SCHEMA_MISMATCH,
        steps=[make_step(0, error="invalid json in response body")],
    ),
    # ── NOT SCHEMA ─────────────────────────────────────────────────────────────
    Case(
        label="no-schema: unrelated error with 'error' keyword",
        expected=FailureType.UNKNOWN,
        steps=[make_step(0, error="connection error: host unreachable")],
    ),
    # ── EXTERNAL_FAULT ─────────────────────────────────────────────────────────
    Case(
        label="external: HTTP 429 rate limit",
        expected=FailureType.EXTERNAL_FAULT,
        steps=[make_step(0, error="HTTP 429 Too Many Requests")],
    ),
    Case(
        label="external: 500 Internal Server Error",
        expected=FailureType.EXTERNAL_FAULT,
        steps=[make_step(0, error="500 Internal Server Error from upstream")],
    ),
    Case(
        label="external: 502 Bad Gateway",
        expected=FailureType.EXTERNAL_FAULT,
        steps=[make_step(0, error="502 Bad Gateway")],
    ),
    Case(
        label="external: 503 Service Unavailable",
        expected=FailureType.EXTERNAL_FAULT,
        steps=[make_step(0, error="status 503 Service Unavailable")],
    ),
    # ── NOT EXTERNAL ───────────────────────────────────────────────────────────
    Case(
        label="no-external: '200 items expected' false positive guard",
        expected=FailureType.UNKNOWN,
        steps=[make_step(0, error="expected 200 items but got 42")],
    ),
    Case(
        label="no-external: port number 5000 in error",
        expected=FailureType.UNKNOWN,
        steps=[make_step(0, error="connection refused on port 5000")],
    ),
    # ── CONSTRAINT_IGNORED ────────────────────────────────────────────────────
    Case(
        label="constraint: output contains forbidden word",
        expected=FailureType.CONSTRAINT_IGNORED,
        steps=[make_step(0, llm_output="Here is the answer formatted as markdown.")],
        task="markdown",  # constraint = "markdown", present in output
    ),
    Case(
        label="constraint: case-insensitive match",
        expected=FailureType.CONSTRAINT_IGNORED,
        steps=[
            make_step(
                0, llm_output="DO NOT USE MARKDOWN is what I should avoid but here it is anyway."
            )
        ],
        task="do not use markdown",  # constraint lowercase, matched case-insensitively
    ),
    # ── NOT CONSTRAINT ─────────────────────────────────────────────────────────
    Case(
        label="no-constraint: clean output",
        expected=FailureType.UNKNOWN,
        steps=[make_step(0, llm_output="Here is a plain text answer.")],
        task="markdown",  # constraint = "markdown", NOT present in output
    ),
    # ── UNKNOWN — types RulesClassifier cannot detect ─────────────────────────
    Case(
        label="unknown: hallucinated state (semantic — needs LLM)",
        expected=FailureType.UNKNOWN,
        steps=[
            make_step(0, tool_called="fetch", tool_output='{"records": 42}'),
            make_step(1, llm_output="I successfully processed all 100 records."),
        ],
    ),
    Case(
        label="unknown: goal drift (semantic — needs LLM)",
        expected=FailureType.UNKNOWN,
        steps=[
            make_step(
                0,
                tool_called="search",
                tool_input={"q": "off-topic"},
                llm_output="I started researching something else entirely.",
            ),
        ],
    ),
    Case(
        label="unknown: empty trajectory",
        expected=FailureType.UNKNOWN,
        steps=[],
    ),
]

# ── LLM/Hybrid-only cases — semantic failures RulesClassifier returns UNKNOWN ─
# These cases exist to measure how well LLMClassifier and HybridClassifier
# handle semantic failure types that pattern-matching cannot detect.

SEMANTIC_CASES: list[Case] = [
    # ── PLAN_INCOMPLETE ────────────────────────────────────────────────────────
    Case(
        label="plan-incomplete: sub-goal never attempted",
        expected=FailureType.PLAN_INCOMPLETE,
        steps=[
            make_step(
                0,
                tool_called="search",
                tool_input={"q": "Paris weather"},
                tool_output='{"temp": "22C"}',
            ),
            make_step(1, llm_output="I found the weather. Task complete."),
        ],
        task="Find the weather in Paris AND book a hotel. Report both.",
    ),
    Case(
        label="plan-incomplete: partial execution stops early",
        expected=FailureType.PLAN_INCOMPLETE,
        steps=[
            make_step(0, tool_called="list_files", tool_output='["a.py","b.py","c.py"]'),
            make_step(
                1,
                tool_called="read_file",
                tool_input={"path": "a.py"},
                tool_output="# file a contents",
            ),
            make_step(2, llm_output="I read file a.py. Done."),
        ],
        task="Read all Python files in the directory and summarize each one.",
    ),
    Case(
        label="plan-incomplete: agent declares success with missing steps",
        expected=FailureType.PLAN_INCOMPLETE,
        steps=[
            make_step(
                0,
                tool_called="send_email",
                tool_input={"to": "alice@example.com", "body": "Hello"},
                tool_output="sent",
            ),
            make_step(1, llm_output="Email sent. All tasks are complete."),
        ],
        task="Send an email to alice@example.com and then log the sent timestamp to audit.log.",
    ),
    # ── CONTEXT_OVERFLOW ───────────────────────────────────────────────────────
    Case(
        label="context-overflow: agent forgets early constraint",
        expected=FailureType.CONTEXT_OVERFLOW,
        steps=[
            make_step(0, llm_output="I understand: output must be in JSON format only."),
            make_step(1, tool_called="search", tool_input={"q": "revenue data"}),
            make_step(2, tool_called="search", tool_input={"q": "cost data"}),
            make_step(3, tool_called="search", tool_input={"q": "profit margins"}),
            make_step(4, tool_called="search", tool_input={"q": "quarterly trends"}),
            make_step(5, tool_called="search", tool_input={"q": "year over year"}),
            make_step(
                6,
                llm_output="Here is the analysis: Revenue grew by 12%. Costs rose 8%. "
                "Profit margins improved. The company performed well overall this year.",
            ),
        ],
        task="Analyze business performance. Output must be in JSON format only.",
    ),
    Case(
        label="context-overflow: loses track of what was already processed",
        expected=FailureType.CONTEXT_OVERFLOW,
        steps=[
            make_step(
                0, tool_called="list_items", tool_output='["item1","item2","item3","item4","item5"]'
            ),
            make_step(
                1, tool_called="process_item", tool_input={"id": "item1"}, tool_output="done"
            ),
            make_step(
                2, tool_called="process_item", tool_input={"id": "item2"}, tool_output="done"
            ),
            make_step(
                3, tool_called="process_item", tool_input={"id": "item3"}, tool_output="done"
            ),
            make_step(
                4, tool_called="process_item", tool_input={"id": "item4"}, tool_output="done"
            ),
            make_step(
                5, tool_called="process_item", tool_input={"id": "item1"}, tool_output="done"
            ),
            make_step(6, llm_output="Processed items. I think I got them all."),
        ],
        task="Process each item exactly once and confirm all five are done.",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────


def run_cases(classifier: object, cases: list[Case]) -> tuple[int, int]:
    """Returns (passed, total)."""
    col_label = 46
    col_exp = 24
    col_got = 24

    header = f"{'LABEL':<{col_label}}  {'EXPECTED':<{col_exp}}  {'GOT':<{col_got}}  RESULT"
    print(header)
    print("-" * len(header))

    passed = 0
    for case in cases:
        t = Trajectory()
        for s in case.steps:
            t.append(s)

        # CONSTRAINT_IGNORED needs constraints= set — extract from task string
        if hasattr(classifier, "constraints") and not classifier.constraints:
            classifier.constraints = [case.task] if case.task != "benchmark task" else []

        got = classifier.classify(t, case.task)

        if hasattr(classifier, "constraints"):
            classifier.constraints = []

        ok = got == case.expected
        result = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        label_str = case.label[:col_label]
        exp_str = case.expected.value[:col_exp]
        got_str = got.value[:col_got]
        print(f"{label_str:<{col_label}}  {exp_str:<{col_exp}}  {got_str:<{col_got}}  {result}")

    return passed, len(cases)


def print_summary(name: str, classifier: object, cases: list[Case]) -> None:
    # Per-type breakdown using the classifier under test.
    by_type: dict[FailureType, list[bool]] = {}
    for case in cases:
        by_type.setdefault(case.expected, [])

    for case in cases:
        t = Trajectory()
        for s in case.steps:
            t.append(s)
        if hasattr(classifier, "constraints"):
            classifier.constraints = [case.task] if case.task != "benchmark task" else []  # type: ignore[attr-defined]
        got = classifier.classify(t, case.task)  # type: ignore[union-attr]
        if hasattr(classifier, "constraints"):
            classifier.constraints = []  # type: ignore[attr-defined]
        by_type[case.expected].append(got == case.expected)

    passed = sum(sum(v) for v in by_type.values())
    total = sum(len(v) for v in by_type.values())
    pct = 100 * passed / total if total else 0
    print(f"\n{name}: {passed}/{total} passed ({pct:.0f}%)")

    print(f"\n  {'TYPE':<30}  PASS  TOTAL")
    for ft, results in sorted(by_type.items(), key=lambda x: x[0].value):
        p = sum(results)
        n = len(results)
        bar = "█" * p + "░" * (n - p)
        print(f"  {ft.value:<30}  {p:>4}  {n:>5}  {bar}")


def _load_llm_classifier() -> object:
    try:
        from triage.classifier.llm import LLMClassifier
    except ImportError:
        print("LLMClassifier not available. Install with: pip install 'triage-agent[anthropic]'")
        sys.exit(1)
    return LLMClassifier(model="claude-haiku-4-5-20251001")


def main() -> None:
    parser = argparse.ArgumentParser(description="triage classifier benchmark")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also benchmark LLMClassifier (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Also benchmark HybridClassifier (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("triage classifier benchmark")
    print("=" * 80)

    # ── RulesClassifier ───────────────────────────────────────────────────────
    print("\n── RulesClassifier ──────────────────────────────────────────────────────────\n")
    clf = RulesClassifier()
    run_cases(clf, CASES)
    print_summary("RulesClassifier", clf, CASES)

    # ── LLMClassifier (optional) ──────────────────────────────────────────────
    if args.llm:
        print(
            "\n\n── LLMClassifier (claude-haiku-4-5-20251001) ─────────────────────────────────\n"
        )
        llm_clf = _load_llm_classifier()
        all_llm_cases = CASES + SEMANTIC_CASES
        run_cases(llm_clf, all_llm_cases)
        print_summary("LLMClassifier", llm_clf, all_llm_cases)

        print("\n  (semantic cases only — PLAN_INCOMPLETE / CONTEXT_OVERFLOW)")
        run_cases(llm_clf, SEMANTIC_CASES)
        print_summary("LLMClassifier [semantic only]", llm_clf, SEMANTIC_CASES)

    # ── HybridClassifier (optional) ───────────────────────────────────────────
    if args.hybrid:
        print(
            "\n\n── HybridClassifier (rules + claude-haiku-4-5-20251001) ───────────────────────\n"
        )
        try:
            from triage.classifier.hybrid import HybridClassifier
        except ImportError:
            print("HybridClassifier not available.")
            sys.exit(1)

        llm_clf_for_hybrid = _load_llm_classifier()
        hybrid_clf = HybridClassifier(llm=llm_clf_for_hybrid)
        all_hybrid_cases = CASES + SEMANTIC_CASES
        run_cases(hybrid_clf, all_hybrid_cases)
        print_summary("HybridClassifier", hybrid_clf, all_hybrid_cases)

        print("\n  (semantic cases only — PLAN_INCOMPLETE / CONTEXT_OVERFLOW)")
        hybrid_clf2 = HybridClassifier(llm=llm_clf_for_hybrid)
        run_cases(hybrid_clf2, SEMANTIC_CASES)
        print_summary("HybridClassifier [semantic only]", hybrid_clf2, SEMANTIC_CASES)

    print()


if __name__ == "__main__":
    main()
