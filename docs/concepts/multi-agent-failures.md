# Multi-Agent Failures: MAST Scoping

**Status: phases 1 and 2 done (see "Recommended phasing" below — phase 2's outcome is a negative result, not shipped code); phase 3 scoping has started (one low-risk prerequisite shipped — see "Phase 3 scoping" below), but the corpus, prompt template, and scoring script it depends on are not built yet.** This document exists to answer one question concretely before writing code: what would it actually take for `triage` to detect multi-agent failure modes, and how much of that can be done honestly with the same rigor the rest of this project holds itself to — measured, not assumed, and never a new `FailureType` without real disambiguation logic behind it.

## Why this exists

Single-agent failure modes — wrong tool, bad schema, a timeout — are increasingly commodity: every agent framework eventually grows its own retry loop for them. Multi-agent failures — a handoff losing context, one agent ignoring another's output, a verifier claiming success on a broken result — are what people actually hit once they move past a single-agent demo, and no comparable library targets them. Aligning with a published, peer-reviewed taxonomy instead of inventing categories from scratch gives this work citable ground instead of guesswork.

## The taxonomy

[**MAST (Multi-Agent System Failure Taxonomy)**](https://github.com/multi-agent-systems-failure-taxonomy/MAST) — Cemri, Pan, Yang, Agrawal, Chopra, Tiwari, Keutzer, Parameswaran, Klein, Ramchandran, et al., *"Why Do Multi-Agent LLM Systems Fail?"*, arXiv:2503.13657 (2025). Built from 1,600+ human- and LLM-annotated execution traces across 7 multi-agent frameworks, with human inter-annotator agreement κ = 0.88. 14 failure modes in 3 categories — definitions quoted below are verbatim from the paper's own repository (`taxonomy_definitions_examples/definitions.txt`), not paraphrased.

*(The commonly-cited ~42%/37%/21% split across the three categories comes from secondary summaries, not something verified here against the primary PDF — arxiv.org isn't reachable from this environment. Don't quote those percentages without verifying them directly first.)*

### 1. Specification Issues (pre-execution / system design)

| Code | Name | Definition (verbatim) |
|---|---|---|
| 1.1 | Disobey Task Specification | Failure to adhere to specified constraints, guidelines, or requirements associated with a task. |
| 1.2 | Disobey Role Specification | Failure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another. |
| 1.3 | Step Repetition | Unnecessarily repeating a phase, task, or stage already completed. |
| 1.4 | Loss of Conversation History | Unexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state. |
| 1.5 | Unaware of Termination Conditions | Failing to adhere to criteria designed to trigger the termination of an interaction, conversation, phase, or task. |

### 2. Inter-Agent Misalignment

| Code | Name | Definition (verbatim) |
|---|---|---|
| 2.1 | Conversation Reset | Unexpected or unwarranted restarting of a dialogue, potentially losing context and progress. |
| 2.2 | Fail to Ask for Clarification | Inability to request additional information when faced with unclear or incomplete data. |
| 2.3 | Task Derailment | Deviation from the intended objective or focus of a task. |
| 2.4 | Information Withholding | An agent possesses critical information but fails to share it promptly or effectively with other agents that rely on it. |
| 2.5 | Ignored Other Agent's Input | Not properly considering input or recommendations from other agents. |
| 2.6 | Action-Reasoning Mismatch | A discrepancy between an agent's stated reasoning/conclusion and the actual action or output the system produces. |

### 3. Task Verification and Termination

| Code | Name | Definition (verbatim) |
|---|---|---|
| 3.1 | Premature Termination | Ending an interaction before all necessary information has been exchanged or objectives have been met. |
| 3.2 | Weak Verification | Verification exists but fails to comprehensively cover all essential aspects — incomplete, superficial, or insufficiently rigorous. |
| 3.3 | No or Incorrect Verification | Verification is absent, or exists but the verifier fails to do what it was actually prompted to do (e.g. told to confirm code compiles, but the code doesn't). |

## The structural prerequisite: `Step` has no agent identity

Before any classification question, there's a data-model question that blocks all of it. `Step`/`Trajectory` today represent one flat, single-actor sequence — there is no field recording *which agent* produced a given step. None of MAST's category 2 (inter-agent) failure modes are even representable in triage's current model, independent of how good a classifier gets — "ignored other agent's input" requires knowing there *was* another agent and what it said.

**✅ Done (phase 1):** `Step.agent_id: str | None = None`, the same low-risk pattern `Step.metadata` already established — an optional field with zero effect on any existing caller. It connects directly to work already shipped: the OTel GenAI semantic conventions already define `gen_ai.agent.id`/`gen_ai.agent.name` on `invoke_agent` spans (verified directly against the spec), so `triage.observability.otel_ingest.trajectory_from_spans()` now captures agent identity the same way it already captured `tool_called`/`http_status` — real frameworks that emit per-agent spans populate this for free, the same story as the HTTP-status connection to `RulesClassifier`'s structured-code matching.

## Mapping the 14 modes

Honesty matters more here than coverage. Three groups, by how they'd actually get detected:

### Already covered by an existing `FailureType` — no new member needed

| MAST mode | Existing mapping |
|---|---|
| 1.3 Step Repetition | `LOOP_DETECTED` — ✅ done (phase 1, see below): `_is_loop_window()` in `rules.py` was already agent-identity-agnostic, so adding `Step.agent_id` made it catch a step repeated *across* agents with zero matching-logic changes — not the "small code change" this doc originally estimated, but no change at all. |
| 3.1 Premature Termination | `PLAN_INCOMPLETE` — CLAUDE.md's own taxonomy table already defines this as "agent declared success but not all required sub-goals were completed," which *is* MAST's 3.1 definition, just framed for one agent. Needs a docs update connecting the two, not a new type. |

Finding two direct hits here — including one where triage already has *working, tested code* for the single-agent case — is the strongest evidence this taxonomy is worth aligning with rather than inventing categories from scratch.

### Investigated for phase 2, found not safe to build as `RulesClassifier` rules

Both looked structurally promising when this document was first written. Actually designing them against real evidence — not just reasoning about `Step` fields — found a reason each one fails, and the two reasons are different, which is itself worth recording rather than collapsing into one "didn't pan out" line.

**3.3 No or Incorrect Verification.** The proposed proxy — compare a step's claimed outcome (`llm_output`) against its `tool_output`/`error` for a contradiction — splits into two cases under scrutiny, and neither needs a new rule:

- If the contradiction eventually produces a real, later exception (MAST's own worked example: ChatDev's `textBasedSpaceInvaders` trace, "reportedly verified" followed by a `FileNotFoundError` traceback — see `taxonomy_definitions_examples/definitions.txt` in the [MAST repo](https://github.com/multi-agent-systems-failure-taxonomy/MAST)), `RulesClassifier` already finds it. Every existing rule scans the *entire* trajectory (`for step in steps: ...`), not just the last step — the same "already covered, no new code" shape as phase 1's `LOOP_DETECTED` finding. The false verification claim adds no new detection surface; the real error was going to be found regardless of whether anything claimed success first.
- If no error ever surfaces — MAST's other worked examples: ChatDev's TicTacToe game ran fine but announced the wrong winner; a Sudoku implementation shipped missing the standard pre-filled numbers — there is no text pattern or structured code to match on at all. Knowing "the announced winner is wrong" requires knowing the rules of tic-tac-toe. This case is irreducibly semantic, same ceiling as `PLAN_INCOMPLETE`/`CONTEXT_OVERFLOW`.

No version of this survives as a new `RulesClassifier` stage. It moves to the semantic-only bucket.

**2.1 Conversation Reset.** The proposed proxy — reuse triage's checkpoint/`state_hash` machinery to spot an unexpected drop back to an earlier state — has two problems, one found by design review and one by evidence:
- triage's own `RecoveryAction.ROLLBACK` *intentionally* restores an earlier checkpoint's state. `Step` has no field marking "this step's state resulted from a deliberate rollback" versus "this step's state reset on its own" — so a rule built on `state_hash` repetition would misfire on triage's own normal, correct rollback behavior. That's a real precision bug waiting to happen, not a design nitpick.
- More fundamentally: real conversation-reset failures don't carry the kind of signal this proxy needs anyway. Two real, cited instances — [microsoft/autogen#1942](https://github.com/microsoft/autogen/issues/1942) (agents silently stopped receiving prior session messages, no error, no log line — the model just responded as if the conversation were new) and [langchain-ai/langgraph#6064](https://github.com/langchain-ai/langgraph/issues/6064) (control silently routes back to the wrong agent when a sub-agent expects a follow-up; "the system simply routes incorrectly," no exception thrown) — are both purely behavioral. Neither produces an error string, a log line, or a state signature of any kind to pattern-match on. This mirrors the corpus D/E finding for HTTP status codes at a different layer: the vocabulary this failure mode actually uses in practice isn't the vocabulary a zero-API-call rule can read.

If this is revisited later, the real prerequisite isn't a better regex — it's a `Step` field distinguishing an intentional triage-driven rollback from an unintentional reset, so *at minimum* the false-positive risk above is ruled out before anything is built on top of it. Proposing that field without a validated need for it (no evidence yet that `state_hash` alone would ever catch a real reset, given the two issues above show none exists to catch) isn't done here — same discipline as not building `agent_id`-driven detection speculatively before `agent_id` itself had a real, cited use.

Both move to the semantic-only bucket below, not because "meaning is generically hard" but for two specific, evidenced reasons: 3.3 is already covered where it's structurally reachable and semantic everywhere else; 2.1's real-world failures carry no structural signal at all to find.

### Semantic-only — `LLMClassifier` extension, not `RulesClassifier`

1.1, 1.2, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.2, 3.3 — twelve of the fourteen (ten judged semantic-only from the start, plus 2.1 and 3.3 above once actually designed against evidence). Each requires understanding *meaning* (was this role-appropriate, was this genuinely unclear, did this deviate from intent, did this verification claim actually hold) that no message-text pattern or structured code can reach — the identical limitation already documented for `PLAN_INCOMPLETE`/`CONTEXT_OVERFLOW`. These are candidates for extending `LLMClassifier`'s prompt to recognize MAST-flavored descriptions in a multi-agent trajectory, not new pattern rules.

## What this means for `FailureType` — and the mistake not to repeat

CLAUDE.md documents exactly this failure mode already happening once: `HALLUCINATED_STATE`/`GOAL_DRIFT` were added in an earlier version and removed in v0.7 because `LLMClassifier` had no real disambiguation logic for them — they looked good as taxonomy entries and didn't work as a classifier. Adding MAST's twelve semantic-only modes as new `FailureType` members today, before any classifier (rules or LLM) can reliably tell them apart, would repeat that mistake at 12x the scale.

**Recommendation: no new `FailureType` members yet.** The two direct hits (1.3, 3.1) need no new type at all. The twelve semantic-only modes should wait for a working, *measured* `LLMClassifier` prompt extension before any of them become a stable enum member — consistent with how this project now treats every accuracy claim: prove it against real data first. There is no longer a "structural candidate" bucket in between: phase 2's investigation closed it, for reasons specific to each mode rather than a generic shrug.

## Phase 3 scoping: extending `LLMClassifier`

**Status: scoping — one low-risk prerequisite shipped, everything else below is design, not built.**

### The real blocker isn't prompt wording — it's where the answer goes

`LLMClassifier.classify()` promises to return one of the 9 stable `FailureType` members: `_parse_response()` matches the raw text against `_FAILURE_TYPE_VALUES` and falls back to `UNKNOWN` for anything else, and `FailurePolicy` routes purely on that enum value. The "What this means for `FailureType`" section above already recommends **not** adding any of the twelve semantic-only MAST modes as new `FailureType` members until a classifier has measured, real disambiguation logic behind it — the same standard whose absence got `HALLUCINATED_STATE`/`GOAL_DRIFT` removed in v0.7. That creates a real ordering problem, not just a wording one: you can't measure whether an LLM prompt can tell twelve new categories apart using a return type that only recognizes nine, but you also shouldn't widen that return type before you've measured it. Resolving that ordering — not writing more prompt text — is what phase 3 actually has to do first.

**The path this doc proposes: split measurement from the stable contract.** Build a MAST-mode-labeling experiment that lives entirely outside `LLMClassifier.classify()`'s contract — a separate prompt template, scored against a separate labeled corpus the same way `classifier_accuracy.py` scores `RulesClassifier` (recall and false-positive rate, split by group, never one blended aggregate) — before touching the enum, `_SYSTEM_PROMPT`, or `_parse_response()` at all. Only once that measurement produces a number worth trusting does "does mode X earn a stable `FailureType` member, and what's its default `RecoveryAction`" get asked, per mode, with evidence behind the answer instead of a taxonomy citation.

### ✅ Shipped prerequisite: `Step.agent_id` now reaches the prompt

Before any of the above is worth attempting, the LLM classifying a failure has to be able to *see* agent identity at all. `_build_prompt()` never emitted `Step.agent_id` — every step looked identical regardless of which agent produced it, so even a perfect prompt couldn't judge 1.2 (Disobey Role Specification), 2.4 (Information Withholding), 2.5 (Ignored Other Agent's Input), or 2.6 (Action-Reasoning Mismatch) — all four are specifically about one agent's behavior *relative to another agent*. Fixed: each step's prompt line now includes `agent: <id>` when `Step.agent_id` is set. Zero behavior change for any existing single-agent caller — `agent_id` stays `None`, the line is omitted, the prompt is byte-for-byte what it was before. This does **not** add MAST-mode detection by itself: `_SYSTEM_PROMPT` and `_parse_response()` are untouched, so `classify()` still only ever returns one of the 9 stable types. It just removes a structural blindness that would have made everything below pointless. See `docs/concepts/classifiers.md`'s "Multi-agent trajectories" section and `tests/test_classifier_llm.py::test_prompt_includes_agent_id_when_set`.

### What a MAST-mode measurement needs, concretely

1. **A prompt distinct from `_SYSTEM_PROMPT`** — e.g. an experimental `_MAST_MODE_PROMPT` describing the twelve semantic-only modes (the verbatim definitions from the table above) plus a "none of these" option, asking for one label. This lives in a measurement script, not in `triage/classifier/llm.py`'s production path, until it's proven to work.
2. **A labeled corpus of real multi-agent traces** — same sourcing discipline as `tests/data/error_corpus_*.json`: real, cited examples (GitHub issues, the MAST paper's own worked examples, real framework transcripts), not synthetic examples written to make a mode look detectable. Two lessons this project already paid for apply directly:
   - **Multiple, independently-sourced examples per mode.** A single example per mode (MAST's own one worked example each) risks the exact "tuned to one example, doesn't generalize" failure already measured for `rules.py`'s v1.1 pattern pass (corpus C → corpus D, 8% routing-sensitive recall, unchanged).
   - **Real negative examples, not just positives.** Trajectories that look superficially similar to a MAST mode but aren't one — mirroring `error_corpus_ambiguous.json`'s `tricky_but_classifiable`/`unknown_labeled` split — otherwise a headline number like "80% recall" hides an unmeasured false-positive rate, the exact gap `hybrid_ambiguity_accuracy.py` exists to close for `HybridClassifier` on the single-agent side.
3. **A held-out/training split decided before the first score, not after.** Corpus C's mistake — scoring against the only corpus you have, then tuning `rules.py` against its misses, which silently converts it into training data — is easier to repeat here, not harder: multi-agent examples are scarcer, so the temptation to reuse the one corpus you found is stronger. Plan for at least two corpora, or one corpus explicitly frozen immediately after its first score, before quoting any number outside this doc.
4. **A scoring script** (`scripts/mast_mode_accuracy.py`, by analogy to `classifier_accuracy.py`) reporting per-mode recall plus an aggregate false-positive rate on the negative set — never a single blended number, the same rule CLAUDE.md already states for corpora A-E.

### Sourcing difficulty, honestly

This corpus is harder to build than any of A-E. Those all draw from a shared vocabulary vendors publish and users file bugs about in predictable, reproducible ways — an HTTP status code, a JSON-RPC code, an SDK exception message. MAST modes are behavioral judgments about multi-agent *transcripts* — "did agent B ignore agent A's input," "did the verifier actually check what it was told to check" — which are rarely filed as a GitHub issue with a clean one-line repro the way "tool not found" is. The two 2.1 citations phase 2 found (`microsoft/autogen#1942`, `langchain-ai/langgraph#6064`) took real, deliberate search effort for *one* mode; twelve modes at that same standard of rigor is a multi-session sourcing effort, not something to rush through to close a phase-3 checkbox. Where a real-world citation can't be found for a mode at the standard held everywhere else in this project, the honest move is to mark that mode unmeasured — not to backfill the gap with a synthetic example dressed up as evidence.

### Recommended next step

Pilot 2-3 modes with the best real chance of citable examples before committing to sourcing all twelve — candidates: 2.5 (Ignored Other Agent's Input) and 2.4 (Information Withholding) both surface in real AutoGen/CrewAI multi-agent bug reports; 1.4 (Loss of Conversation History) overlaps with the same "context truncation" reports frameworks already file for single-agent long-horizon issues. Measure the prompt against just that pilot set first. This mirrors how corpus E tested one structural hypothesis (JSON-RPC codes generalizing where HTTP codes didn't) before generalizing further, rather than building all twelve at once and discovering afterward that most of them don't have real evidence at the rigor this project holds itself to elsewhere.

## Recommended phasing

1. ✅ **Done.** `Step.agent_id: str | None = None` — pure additive field, zero behavior change for existing callers. `otel_ingest.trajectory_from_spans()` populates it from `gen_ai.agent.id`/`gen_ai.agent.name` when present (`id` preferred when both are set). Cross-agent loop detection ships too — but the honest correction to this doc's own earlier estimate: it needed **zero changes** to `_is_loop_window()`'s matching logic, not an extension of it. That function was already agent-agnostic (it only ever compared `tool_called`/`tool_input`, never looked at agent identity, because the field didn't exist) — adding `agent_id` to `Step` made the existing single-agent test suite's matching logic correct for the multi-agent case for free. Pinned by `test_loop_detected_across_different_agent_ids` in `tests/test_classifier_rules.py`, so a future change can't accidentally narrow it back to same-agent-only without a test failing. `RulesClassifier`'s docstring and the `classify()` comment above the loop check now document this as deliberate, not unnoticed.
2. ✅ **Investigated — negative result, no code shipped.** 3.3 (verification mismatch) and 2.1 (conversation reset) were designed against real evidence rather than prototyped speculatively, and neither survived as a safe `RulesClassifier` rule: 3.3 reduces to "already covered by existing trajectory-scanning rules" where a real error eventually appears, and is irreducibly semantic where it doesn't (MAST's own TicTacToe/Sudoku examples). 2.1's real-world instances ([microsoft/autogen#1942](https://github.com/microsoft/autogen/issues/1942), [langchain-ai/langgraph#6064](https://github.com/langchain-ai/langgraph/issues/6064)) are purely behavioral with no error signature to match, and the checkpoint-reuse proxy originally proposed would have false-positived on triage's own intentional `ROLLBACK`. See "Investigated for phase 2" above for the full reasoning per mode. This is a complete, valid phase 2 outcome, not a stalled one — building a weak or false-positive-prone rule just to have shipped *something* would have violated the same 100%-precision-by-construction discipline `rules.py`'s message-text patterns already hold themselves to.
3. 🔄 **Scoping started.** See "Phase 3 scoping" above. One low-risk prerequisite shipped (`Step.agent_id` now reaches `LLMClassifier`'s prompt when set). The real design question resolved by that scoping: measure a MAST-mode-labeling prompt against a real, cited, held-out-disciplined multi-agent corpus *before* touching `_SYSTEM_PROMPT`/`_parse_response()` or the `FailureType` enum — not "extend the prompt and see." No corpus, scoring script, or prompt-template code exists yet; sourcing real examples for all twelve modes at the same rigor as corpora A-E is scoped as a multi-session effort, with a smaller pilot (2-3 modes) recommended first.

## Non-goals

This is not a plan to add 14 new `FailureType` members. It is not a plan to build a multi-agent orchestration framework — triage still wraps whatever callable you give it, single- or multi-agent. Phases 1 and 2 are done (see above); phase 3 is scoped but not built.
