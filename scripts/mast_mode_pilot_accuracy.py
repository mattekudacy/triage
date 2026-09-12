"""
scripts/mast_mode_pilot_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measures whether an LLM can even distinguish the three piloted MAST modes
(2.5 Ignored Other Agent's Input, 2.4 Information Withholding, 1.4 Loss of
Conversation History) against tests/data/mast_pilot_corpus.json — the first
real measurement docs/concepts/multi-agent-failures.md's "Phase 3 scoping"
section calls for, before any of these modes are considered for a stable
FailureType member or a change to LLMClassifier's production prompt.

DELIBERATELY separate from triage/classifier/llm.py: this script's prompt
and label set live only here. LLMClassifier.classify() is untouched and
still only ever returns one of the 9 stable FailureType members —
_SYSTEM_PROMPT and _parse_response() are not modified by this experiment.
That separation is the whole point of the phase 3 scoping doc's proposed
ordering: measure first, widen the stable contract (if ever) only after.

What this DOES measure: per-mode recall on 5 scorable entries (2 each for
2.5 and 1.4, 1 for 2.4 — see the corpus's docstring for why one 2.4 entry
was excluded as ambiguous ground truth) plus a qualitative look at what the
model says about the one ambiguous entry.

What this does NOT measure, and the corpus does not yet support: precision /
false-positive rate. There are no negative examples yet (real trajectories
that resemble a MAST mode but aren't one) — see the corpus generator's
docstring. A high recall number here says the model can find real MAST-mode
signal when told what to look for; it says nothing about how often it would
also fire on a trajectory that isn't one. Do not quote this script's recall
number as if it settled that question — a corpus_ambiguous.json-style
negative set is necessary follow-up before any accuracy claim is complete.

Requires an LLM backend — same env vars / --model / --max-tokens as
llm_classifier_accuracy.py / hybrid_ambiguity_accuracy.py:

    ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/mast_mode_pilot_accuracy.py

    TRIAGE_LLM_BASE_URL=https://ollama.com/v1 \\
    TRIAGE_LLM_API_KEY=... TRIAGE_LLM_MODEL=gpt-oss:120b-cloud TRIAGE_LLM_MAX_TOKENS=500 \\
    PYTHONPATH=. python scripts/mast_mode_pilot_accuracy.py

Run:
    PYTHONPATH=. .venv/bin/python scripts/mast_mode_pilot_accuracy.py \\
        [--model MODEL] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

CORPUS_PATH = Path("tests/data/mast_pilot_corpus.json")
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_MAX_TOKENS = 32

# Verbatim definitions from docs/concepts/multi-agent-failures.md's taxonomy
# table, restricted to the three piloted modes plus a "none" escape hatch —
# NOT the full twelve. Widening this to all twelve is exactly the "measure a
# pilot before committing to a full corpus" step this script exists to gate.
_MODE_DEFINITIONS = {
    "2.5": "Ignored Other Agent's Input — Not properly considering input or "
    "recommendations from other agents.",
    "2.4": "Information Withholding — An agent possesses critical information "
    "but fails to share it promptly or effectively with other agents that "
    "rely on it.",
    "1.4": "Loss of Conversation History — Unexpected context truncation, "
    "disregarding recent interaction history and reverting to an antecedent "
    "conversational state.",
}

_SYSTEM_PROMPT = (
    "You are analyzing a multi-agent AI system's execution trajectory for a "
    "specific kind of failure. Given a trajectory of steps (each may show "
    "which agent produced it) and a task description, decide whether the "
    "trajectory shows one of these failure modes:\n\n"
    + "\n".join(f"{code}: {desc}" for code, desc in _MODE_DEFINITIONS.items())
    + '\n\nRespond with only the mode code (e.g. "2.5"), or "none" if the '
    "trajectory doesn't show any of them. Nothing else."
)


def _build_prompt(entry: dict[str, Any]) -> str:
    lines = [f"Task: {entry['task']}", "", "Steps:"]
    for step in entry["steps"]:
        lines.append(f"[{step['index']}] {step['action']}")
        if step.get("agent_id"):
            lines.append(f"  agent: {step['agent_id']}")
        if step.get("tool_called"):
            lines.append(f"  tool: {step['tool_called']}")
        if step.get("tool_input"):
            lines.append(f"  tool_input: {step['tool_input']}")
        if step.get("tool_output"):
            lines.append(f"  tool_output: {step['tool_output']}")
        if step.get("error"):
            lines.append(f"  error: {step['error']}")
        if step.get("llm_output"):
            lines.append(f"  llm_output: {step['llm_output']}")
    lines.append("")
    lines.append("Which failure mode, if any, does this trajectory show?")
    return "\n".join(lines)


def _check_backend_installed() -> None:
    """See llm_classifier_accuracy.py — same check, duplicated per this
    repo's convention of self-contained scripts."""
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


class _MastModePilotClassifier:
    """Minimal, standalone classifier for this experiment only.

    Deliberately NOT LLMClassifier — that class's _parse_response() only
    recognizes the 9 stable FailureType values, and its _SYSTEM_PROMPT is
    part of triage's stable-ish production path. Reusing it here would blur
    exactly the line docs/concepts/multi-agent-failures.md's phase 3 scoping
    draws between "production classify() contract" and "measurement
    experiment" — so this duplicates the small amount of client-construction
    logic needed instead, same as _check_backend_installed() above.
    """

    def __init__(self, model: str, max_tokens: int) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = os.environ.get("TRIAGE_LLM_BASE_URL")
        self._api_key = os.environ.get("TRIAGE_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if self._base_url is not None:
            import openai

            return openai.OpenAI(api_key=self._api_key or "no-key", base_url=self._base_url)
        import anthropic

        return anthropic.Anthropic(api_key=self._api_key)

    def classify(self, prompt: str) -> str:
        if self._base_url is not None:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = str(response.choices[0].message.content or "")
        else:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = str(message.content[0].text)
        raw = raw.strip().lower()
        for code in _MODE_DEFINITIONS:
            if code in raw:
                return code
        return "none"


def _sanity_check(clf: _MastModePilotClassifier) -> None:
    """A trajectory that unambiguously shows 2.5 (a worker literally
    contradicts an explicit supervisor instruction one step earlier) —
    refuses to score anything if the classifier fails this trivially, same
    pattern as llm_classifier_accuracy.py / hybrid_ambiguity_accuracy.py."""
    sanity_entry = {
        "task": "Supervisor tells the worker exactly what to do; worker does the opposite.",
        "steps": [
            {
                "index": 0,
                "agent_id": "supervisor",
                "action": "instruct",
                "llm_output": "Retrieve only vegetarian dishes.",
            },
            {
                "index": 1,
                "agent_id": "worker",
                "action": "retrieve",
                "llm_output": "Retrieved dishes: steak, bacon, fried chicken.",
            },
        ],
    }
    got = clf.classify(_build_prompt(sanity_entry))
    if got != "2.5":
        print(
            f"SANITY CHECK FAILED: got {got!r} for an unambiguous "
            "instruction-vs-action contradiction (expected '2.5'). Refusing "
            "to score against a classifier that fails this trivially — see "
            "hybrid_ambiguity_accuracy.py's sanity check for why.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("TRIAGE_LLM_MODEL") or _DEFAULT_MODEL,
        help=f"Model name (default: TRIAGE_LLM_MODEL env var, else {_DEFAULT_MODEL!r})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("TRIAGE_LLM_MAX_TOKENS", _DEFAULT_MAX_TOKENS)),
        help="Output token budget — pass a few hundred for reasoning models "
        f"(default: TRIAGE_LLM_MAX_TOKENS env var, else {_DEFAULT_MAX_TOKENS}).",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(f"Corpus not found: {CORPUS_PATH} — run scripts/gen_mast_pilot_corpus.py")
    entries = json.loads(CORPUS_PATH.read_text())

    _check_backend_installed()

    clf = _MastModePilotClassifier(model=args.model, max_tokens=args.max_tokens)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    if os.environ.get("TRIAGE_LLM_BASE_URL"):
        print(f"Base URL: {os.environ['TRIAGE_LLM_BASE_URL']}")
    print("Running sanity check...", end=" ", flush=True)
    _sanity_check(clf)
    print("ok\n")

    scored = [e for e in entries if "ambiguous_with" not in e]
    ambiguous = [e for e in entries if "ambiguous_with" in e]

    print("=" * 65)
    print(f"{len(scored)} scorable entries (single ground-truth label),")
    print(f"{len(ambiguous)} ambiguous entries (reported separately, not")
    print("scored — see the corpus generator's docstring).")
    print("NOTE: this is a RECALL-only measurement. No negative examples")
    print("exist yet, so this says nothing about false-positive rate.")
    print("=" * 65)
    print()

    hits_by_mode: dict[str, int] = {code: 0 for code in _MODE_DEFINITIONS}
    total_by_mode: dict[str, int] = {code: 0 for code in _MODE_DEFINITIONS}
    misses = []
    for entry in scored:
        true_mode = entry["mast_mode"]
        got = clf.classify(_build_prompt(entry))
        total_by_mode[true_mode] += 1
        if got == true_mode:
            hits_by_mode[true_mode] += 1
        else:
            misses.append((entry, got))

    print("── Per-mode recall ──────────────────────────────────────────────")
    for code, name in [(c, _MODE_DEFINITIONS[c].split(" — ")[0]) for c in _MODE_DEFINITIONS]:
        total = total_by_mode[code]
        if total == 0:
            print(f"  {code} {name}: no scorable entries in this pilot")
            continue
        hits = hits_by_mode[code]
        print(f"  {code} {name}: {hits}/{total}")
    print()

    if misses:
        print("Misses:")
        for entry, got in misses:
            print(f"  true={entry['mast_mode']:5} got={got:5} {entry['provenance']}")
        print()

    if ambiguous:
        print("── Ambiguous ground-truth entries (not scored) ────────────────────")
        for entry in ambiguous:
            got = clf.classify(_build_prompt(entry))
            print(
                f"  {entry['provenance']}: model said {got!r} "
                f"(ambiguous between its true label and {entry['ambiguous_with']!r})"
            )
        print()


if __name__ == "__main__":
    main()
