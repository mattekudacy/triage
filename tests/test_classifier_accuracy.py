"""tests/test_classifier_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classifier accuracy tests. See scripts/classifier_accuracy.py's module
docstring for the full nine-block breakdown this file's TestCorpus* classes
correspond to; summary:

  Block 1 — Regression:
      In-corpus positive examples from test_classifier_rules.py fixtures.
      Expected: 100%. Tautological by design — the regexes were written
      against these strings. Prevents *regression*, not generalization.

  Block 2 — False-positive resistance:
      Near-miss strings that must NOT trigger a rule. A false positive routes
      a failure to the wrong recovery strategy, which is worse than UNKNOWN.
      Expected: 100%. Treated as a hard constraint.

  Blocks 3-4 — Corpus A, Corpus B (training data, regression guards):
      Both were held-out once, guided a pattern-fix pass, and are now
      regression suites at 100% — same status corpus C reaches below.

  Block 5 — Corpus C (training data as of v1.1, regression guard):
      Genuinely held-out through v1.0 (52% recall). The v1.1 pattern pass
      tuned rules.py directly against its 13 misses, which is what converts
      a corpus to training data — CORPUS_C_FLOOR is now a regression guard,
      not a generalization claim. See TestCorpusD below for the current one.

  Blocks 7-8 — Corpus D (genuine held-out as of v1.1):
      Fresh sources disjoint from A/B/C, scored once after the v1.1 tuning
      pass. Routing-sensitive recall (1/12 = 8%) is statistically unchanged
      from corpus C's pre-tuning number — the v1.1 patterns did not
      generalize past corpus C's own wording. See CHANGELOG.

  Blocks 9-10 — Corpus E (Step 2 of the Corpus E scoping plan):
      Fresh sources disjoint from A-D, testing whether Step.metadata
      structured-error-code matching (added after corpus D) generalizes
      better than message-text tuning did. Routing-sensitive recall rose to
      4/9 = 44% — but almost entirely via MCP's json_rpc_code, not HTTP
      status; see TestCorpusE's docstring and docs/known-limitations.md.
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
# training data (same status as corpus A). Its last 2 misses (both explicit
# "improvement targets" left over from v0.26 — the aiohttp inactivity-timeout
# phrasing and the "Tool X is not registered" wording) were closed in the
# v1.1 pattern pass. Update upward only; never decrease.
CORPUS_B_FLOOR = 1.0  # 20/20, measured 2026-09-11 (v1.1 tuning pass)

CORPUS_C_PATH = Path(__file__).parent / "data" / "error_corpus_c.json"
# STATUS CHANGE (v1.1): corpus C was genuinely held-out through v1.0 (52%
# recall, 100% precision, scored once). The v1.1 pattern pass tuned rules.py
# directly against its 13 misses — that is what converts a corpus to
# training data, the same way it happened to corpus A (v0.25) and corpus B
# (v0.26) before it. CORPUS_C_FLOOR is now a regression guard, like
# CORPUS_A_FLOOR, not a generalization claim. Corpus D (below) carries that
# claim now.
CORPUS_C_FLOOR = 1.0  # 27/27, measured 2026-09-11 (v1.1 tuning pass)

# Groups shared by the corpus C and corpus D per-type tests below.
# SELF_HEALING types recover from any retry — a bare `for _ in range(3)` loop fixes
# them, so classifying them correctly adds nothing over blind retry.
# ROUTING_SENSITIVE types only recover when the matched hint reaches the strategy;
# they are the reason this library exists.
SELF_HEALING_LABELS = ("external_fault", "timeout")
ROUTING_SENSITIVE_LABELS = ("wrong_tool_called", "schema_mismatch")
# Now regression guards (corpus C is training data as of v1.1 — see above).
CORPUS_C_SELF_HEALING_FLOOR = 1.0  # 14/14, measured 2026-09-11
CORPUS_C_ROUTING_SENSITIVE_FLOOR = 1.0  # 12/12, measured 2026-09-11

CORPUS_D_PATH = Path(__file__).parent / "data" / "error_corpus_d.json"
# Genuine held-out floor (v1.1): corpus D was scored ONCE, immediately after
# the v1.1 pattern pass against corpus C, without any further rules.py edit.
# Sources are disjoint from A, B, and C: huggingface_hub, Ollama, OpenRouter,
# Model Context Protocol (MCP), CrewAI, Semantic Kernel, and novel phrasings
# chosen to stress the v1.1 patterns' boundaries.
# One misroute this scoring pass found (CrewAI's OutputParserError colliding
# with LlamaIndex's same-named exception) was fixed as a precision bug before
# freezing this floor — see test_output_parser_error_exception_type_alone_
# does_not_fire_schema in test_classifier_rules.py. 100% precision as a
# result: every remaining miss returns UNKNOWN, zero misroutes.
# Do NOT tune rules.py against corpus D misses — it will then become
# training data the same way it happened to corpus C. Generate corpus E.
CORPUS_D_FLOOR = 0.40  # 8/20 = 40%, measured 2026-09-11

# Headline finding of the v1.1 cycle: this floor is statistically unchanged
# from corpus C's PRE-tuning routing-sensitive floor (1/12 = 8.3%, same
# number). The v1.1 patterns were narrow string literals keyed close to
# corpus C's exact wording and did not transfer to fresh SDKs. Raising this
# floor is the next cycle's headline goal; it must never be lowered.
CORPUS_D_SELF_HEALING_FLOOR = 0.85  # 6/7 = 85.7%, measured 2026-09-11
CORPUS_D_ROUTING_SENSITIVE_FLOOR = 0.08  # 1/12 = 8.3%, measured 2026-09-11

CORPUS_E_PATH = Path(__file__).parent / "data" / "error_corpus_e.json"
# Step 2 of docs/known-limitations.md's "Corpus E scoping" plan: does the
# Step.metadata structured-error-code matching added after corpus D (see
# triage/classifier/rules.py's _JSON_RPC_*/_HTTP_* tables) generalize better
# than the v1.1 message-text patterns did? Sources disjoint from A-D: MCP
# (this time capturing json_rpc_code, not just message text), Together AI,
# Fireworks AI, Replicate, Cerebras, Perplexity, DeepSeek, NVIDIA NIM, xAI.
# Scored once. One adversarial case (a real MCP server reusing -32600 for a
# session/auth condition, not a malformed request — see
# scripts/gen_error_corpus_e.py) found a genuine precision bug before this
# floor was frozen: -32600 was dropped from _JSON_RPC_SCHEMA_CODES, same
# "generic code reused for an unrelated failure" pattern as corpus D's
# OutputParserError/CrewAI fix. 100% precision as a result — every miss
# below returns UNKNOWN, zero misroutes.
# Do NOT tune rules.py against corpus E's remaining misses — that converts E
# to training data the same way it happened to C and would to D. Generate a
# corpus F to test a further change.
CORPUS_E_FLOOR = 0.6875  # 11/16 = 68.75%, measured 2026-09-11
CORPUS_E_SELF_HEALING_FLOOR = 1.0  # 6/6 = 100%, measured 2026-09-11
CORPUS_E_ROUTING_SENSITIVE_FLOOR = 0.44  # 4/9 = 44.4%, measured 2026-09-11


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
                metadata=entry.get("metadata") or {},
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
                metadata=entry.get("metadata") or {},
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
    """Block 5: Corpus C — training data as of v1.1 (regression guard).

    Corpus C was built from sources not in A or B: azure-core, Mistral AI SDK,
    Cohere SDK, Groq SDK, LiteLLM, Vertex AI (aiplatform SDK), LlamaIndex, and
    novel structural phrasings. It was genuinely held-out through v1.0 (52%
    recall, 100% precision, scored once).

    STATUS CHANGE (v1.1): rules.py was tuned directly against corpus C's 13
    misses — the same act that converted corpus A (v0.25) and corpus B
    (v0.26) to training data converted this one. These three tests are now
    regression guards, same status as TestCorpusA/TestCorpusB, not a
    generalization claim. See TestCorpusD for the current held-out measurement.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_C_PATH.exists(), f"Corpus file missing: {CORPUS_C_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_C_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_c_no_regression(self) -> None:
        result = _score_corpus(CORPUS_C_PATH)
        if result.accuracy < CORPUS_C_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus C regression: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_C_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_C_FLOOR

    def test_corpus_c_self_healing_no_regression(self) -> None:
        result = _score_corpus_group(CORPUS_C_PATH, SELF_HEALING_LABELS)
        assert result.accuracy >= CORPUS_C_SELF_HEALING_FLOOR, (
            f"Self-healing recall {result.accuracy:.0%}"
            f" ({result.correct}/{result.total}) below floor"
            f" {CORPUS_C_SELF_HEALING_FLOOR:.0%}."
        )

    def test_corpus_c_routing_sensitive_no_regression(self) -> None:
        result = _score_corpus_group(CORPUS_C_PATH, ROUTING_SENSITIVE_LABELS)
        if result.accuracy < CORPUS_C_ROUTING_SENSITIVE_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Routing-sensitive recall: {result.accuracy:.0%}"
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


class TestCorpusD:
    """Blocks 7-8: Corpus D — genuine held-out generalization measurement (v1.1).

    Corpus D was built from sources not in A, B, or C: huggingface_hub, Ollama,
    OpenRouter, Model Context Protocol (MCP), CrewAI, Semantic Kernel, and
    novel phrasings chosen to stress the boundaries of the v1.1 patterns added
    for corpus C. Scored exactly once, immediately after the v1.1 tuning pass
    against corpus C, before any further rules.py edit.

    One misroute this scoring pass found (CrewAI's OutputParserError colliding
    with LlamaIndex's same-named-but-unrelated exception) was fixed as a
    precision bug — not as recall tuning — before this floor was frozen; see
    test_output_parser_error_exception_type_alone_does_not_fire_schema in
    test_classifier_rules.py. 100% precision as a result.

    40% recall (8/20) overall. The routing-sensitive floor (1/12 = 8.3%) is
    the headline finding of the v1.1 cycle: it is statistically unchanged
    from corpus C's PRE-tuning number (also 1/12 = 8.3%). The v1.1 patterns
    were narrow string literals keyed close to corpus C's exact wording and
    did not transfer to fresh SDKs' phrasing — see CHANGELOG for the full
    writeup. Do NOT use corpus D misses to tune rules.py — the moment you do,
    D becomes training data like C before it. Generate corpus E first.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_D_PATH.exists(), f"Corpus file missing: {CORPUS_D_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_D_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_d_held_out_floor(self) -> None:
        result = _score_corpus(CORPUS_D_PATH)
        if result.accuracy < CORPUS_D_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus D held-out: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_D_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_D_FLOOR

    def test_corpus_d_no_misroutes(self) -> None:
        """Precision guard, independent of the recall floor above: every miss
        on corpus D must land in UNKNOWN, never in a different concrete type.
        A misroute is worse than UNKNOWN — it applies the wrong strategy."""
        result = _score_corpus(CORPUS_D_PATH)
        misroutes = [m for m in result.misses if m[1] != "unknown"]
        assert not misroutes, (
            f"Corpus D misroutes (must be zero): "
            f"{[(exp, got, err) for exp, got, _, err in misroutes]}"
        )

    def test_corpus_d_self_healing_floor(self) -> None:
        """Recall on types a bare retry loop would recover anyway.

        High here is expected and not worth much: EXTERNAL_FAULT and TIMEOUT heal
        on any retry, so correct classification buys nothing over blind retry.
        Guarded only so a regression here is still caught.
        """
        result = _score_corpus_group(CORPUS_D_PATH, SELF_HEALING_LABELS)
        assert result.accuracy >= CORPUS_D_SELF_HEALING_FLOOR, (
            f"Self-healing held-out recall {result.accuracy:.0%}"
            f" ({result.correct}/{result.total}) below floor"
            f" {CORPUS_D_SELF_HEALING_FLOOR:.0%}."
        )

    def test_corpus_d_routing_sensitive_floor(self) -> None:
        """Recall on the types that justify the library.

        WRONG_TOOL_CALLED and SCHEMA_MISMATCH only recover when the matched hint
        reaches the strategy — these are the types scripts/bench_synthetic.py shows
        triage winning on, and the only ones where classification beats blind retry.
        Held-out recall here is the honest measure of delivered value, and it is
        currently 1/12 — unchanged from corpus C's pre-tuning number. Raising this
        floor is the next cycle's headline goal; it must never be lowered.
        """
        result = _score_corpus_group(CORPUS_D_PATH, ROUTING_SENSITIVE_LABELS)
        if result.accuracy < CORPUS_D_ROUTING_SENSITIVE_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Routing-sensitive held-out recall: {result.accuracy:.0%}"
                f" ({result.correct}/{result.total}) below floor"
                f" {CORPUS_D_ROUTING_SENSITIVE_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_D_ROUTING_SENSITIVE_FLOOR

    def test_every_corpus_d_label_is_grouped(self) -> None:
        """Guard the split: a new failure type must be classified into a group.

        If a future corpus entry carries a label in neither group, the two group
        floors stop covering the corpus and the headline number silently drifts.
        UNKNOWN is exempt — it is the fall-through, not a detection target.
        """
        entries = json.loads(CORPUS_D_PATH.read_text())
        grouped = set(SELF_HEALING_LABELS) | set(ROUTING_SENSITIVE_LABELS) | {"unknown"}
        ungrouped = {e["label"] for e in entries} - grouped
        assert not ungrouped, (
            f"Corpus D labels not assigned to a group: {sorted(ungrouped)}."
            " Add them to SELF_HEALING_LABELS or ROUTING_SENSITIVE_LABELS."
        )


class TestCorpusE:
    """Blocks 9-10: Corpus E — Step 2 of the Corpus E scoping plan.

    Tests whether RulesClassifier's structured-error-code matching
    (Step.metadata["http_status"]/["json_rpc_code"], added after corpus D)
    generalizes better on fresh sources than the v1.1 message-text patterns
    did. Sources disjoint from A-D: MCP (capturing json_rpc_code this time,
    not just message text), Together AI, Fireworks AI, Replicate, Cerebras,
    Perplexity, DeepSeek, NVIDIA NIM, xAI. Scored once.

    Result: routing-sensitive recall is 4/9 = 44.4%, up from corpus D's
    1/12 = 8.3% — but nearly all of the gain is the two MCP json_rpc_code
    entries (-32601/-32700). Every fresh HTTP-only vendor's "wrong tool"/
    "bad schema" failure in this corpus used 404/400/422 — codes
    deliberately excluded from the HTTP tables for the same ambiguity
    reasons the JSON-RPC exclusions use. The structural signal generalizes
    where a spec-mandated code exists (JSON-RPC); it does nothing for
    HTTP-only vendors, because the codes they actually return for these
    failure types are the ones excluded on purpose. See CHANGELOG and
    docs/known-limitations.md's "Corpus E scoping" for the full breakdown.

    One adversarial case (a real MCP server reusing -32600 for a
    session/auth condition, not a malformed request per
    langgenius/dify#22675 — see scripts/gen_error_corpus_e.py) found a
    genuine precision bug before this floor was frozen: -32600 was dropped
    from _JSON_RPC_SCHEMA_CODES, same "generic code reused for an unrelated
    failure" pattern as corpus D's OutputParserError/CrewAI fix. 100%
    precision as a result — every miss below returns UNKNOWN.

    Do NOT use corpus E's remaining misses to tune rules.py — that converts
    E to training data the same way it happened to C and would to D.
    """

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_E_PATH.exists(), f"Corpus file missing: {CORPUS_E_PATH}"

    def test_corpus_all_valid_labels(self) -> None:
        valid = {ft.value for ft in FailureType}
        entries = json.loads(CORPUS_E_PATH.read_text())
        for entry in entries:
            assert entry["label"] in valid, f"Invalid label {entry['label']!r}"

    def test_corpus_e_held_out_floor(self) -> None:
        result = _score_corpus(CORPUS_E_PATH)
        if result.accuracy < CORPUS_E_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Corpus E held-out: {result.accuracy:.0%} ({result.correct}/{result.total})"
                f" below floor {CORPUS_E_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_E_FLOOR

    def test_corpus_e_no_misroutes(self) -> None:
        """Precision guard, independent of the recall floor above: every miss
        on corpus E must land in UNKNOWN, never in a different concrete type.
        This is also the regression guard for the -32600 adversarial case —
        if it ever fires SCHEMA_MISMATCH again, this test catches it."""
        result = _score_corpus(CORPUS_E_PATH)
        misroutes = [m for m in result.misses if m[1] != "unknown"]
        assert not misroutes, (
            f"Corpus E misroutes (must be zero): "
            f"{[(exp, got, err) for exp, got, _, err in misroutes]}"
        )

    def test_corpus_e_self_healing_floor(self) -> None:
        result = _score_corpus_group(CORPUS_E_PATH, SELF_HEALING_LABELS)
        assert result.accuracy >= CORPUS_E_SELF_HEALING_FLOOR, (
            f"Self-healing held-out recall {result.accuracy:.0%}"
            f" ({result.correct}/{result.total}) below floor"
            f" {CORPUS_E_SELF_HEALING_FLOOR:.0%}."
        )

    def test_corpus_e_routing_sensitive_floor(self) -> None:
        """The number that answers Step 2's actual question: did the
        structured-code mechanism move routing-sensitive recall on fresh
        sources? 4/9 = 44.4%, up from corpus D's 1/12 = 8.3% — see this
        class's docstring for the honest breakdown of where the gain came
        from (JSON-RPC only, not HTTP status)."""
        result = _score_corpus_group(CORPUS_E_PATH, ROUTING_SENSITIVE_LABELS)
        if result.accuracy < CORPUS_E_ROUTING_SENSITIVE_FLOOR:
            detail = "\n".join(
                f"  exp={exp:16} got={got:16} [{exc}] {err!r}"
                for exp, got, exc, err in result.misses
            )
            pytest.fail(
                f"Routing-sensitive held-out recall: {result.accuracy:.0%}"
                f" ({result.correct}/{result.total}) below floor"
                f" {CORPUS_E_ROUTING_SENSITIVE_FLOOR:.0%}.\nMisses:\n{detail}"
            )
        assert result.accuracy >= CORPUS_E_ROUTING_SENSITIVE_FLOOR

    def test_every_corpus_e_label_is_grouped(self) -> None:
        """Guard the split — see test_every_corpus_d_label_is_grouped."""
        entries = json.loads(CORPUS_E_PATH.read_text())
        grouped = set(SELF_HEALING_LABELS) | set(ROUTING_SENSITIVE_LABELS) | {"unknown"}
        ungrouped = {e["label"] for e in entries} - grouped
        assert not ungrouped, (
            f"Corpus E labels not assigned to a group: {sorted(ungrouped)}."
            " Add them to SELF_HEALING_LABELS or ROUTING_SENSITIVE_LABELS."
        )

    def test_corpus_e_unknown_labeled_entry_stays_unknown(self) -> None:
        """The adversarial MCP -32600 case must score as a hit (correctly
        left UNKNOWN), not just avoid a misroute in aggregate — pins the
        specific entry this corpus was built to catch."""
        entries = json.loads(CORPUS_E_PATH.read_text())
        unknown_entries = [e for e in entries if e["label"] == "unknown"]
        assert unknown_entries, "Expected at least one unknown-labeled entry in corpus E"
        for entry in unknown_entries:
            t = Trajectory()
            t.append(
                Step(
                    index=0,
                    action="a",
                    error=entry["error"],
                    exception_type=entry.get("exception_type"),
                    metadata=entry.get("metadata") or {},
                )
            )
            got = RulesClassifier().classify(t, "task")
            assert got == FailureType.UNKNOWN, (
                f"Expected UNKNOWN for adversarial entry {entry['error'][:50]!r}, got {got.value}"
            )
