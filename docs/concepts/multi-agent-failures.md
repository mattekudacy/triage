# Multi-Agent Failures: MAST Scoping

**Status: phases 1 and 2 done (see "Recommended phasing" below — phase 2's outcome is a negative result, not shipped code); phase 3 is still a design proposal, not implemented.** This document exists to answer one question concretely before writing code: what would it actually take for `triage` to detect multi-agent failure modes, and how much of that can be done honestly with the same rigor the rest of this project holds itself to — measured, not assumed, and never a new `FailureType` without real disambiguation logic behind it.

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

## Recommended phasing

1. ✅ **Done.** `Step.agent_id: str | None = None` — pure additive field, zero behavior change for existing callers. `otel_ingest.trajectory_from_spans()` populates it from `gen_ai.agent.id`/`gen_ai.agent.name` when present (`id` preferred when both are set). Cross-agent loop detection ships too — but the honest correction to this doc's own earlier estimate: it needed **zero changes** to `_is_loop_window()`'s matching logic, not an extension of it. That function was already agent-agnostic (it only ever compared `tool_called`/`tool_input`, never looked at agent identity, because the field didn't exist) — adding `agent_id` to `Step` made the existing single-agent test suite's matching logic correct for the multi-agent case for free. Pinned by `test_loop_detected_across_different_agent_ids` in `tests/test_classifier_rules.py`, so a future change can't accidentally narrow it back to same-agent-only without a test failing. `RulesClassifier`'s docstring and the `classify()` comment above the loop check now document this as deliberate, not unnoticed.
2. ✅ **Investigated — negative result, no code shipped.** 3.3 (verification mismatch) and 2.1 (conversation reset) were designed against real evidence rather than prototyped speculatively, and neither survived as a safe `RulesClassifier` rule: 3.3 reduces to "already covered by existing trajectory-scanning rules" where a real error eventually appears, and is irreducibly semantic where it doesn't (MAST's own TicTacToe/Sudoku examples). 2.1's real-world instances ([microsoft/autogen#1942](https://github.com/microsoft/autogen/issues/1942), [langchain-ai/langgraph#6064](https://github.com/langchain-ai/langgraph/issues/6064)) are purely behavioral with no error signature to match, and the checkpoint-reuse proxy originally proposed would have false-positived on triage's own intentional `ROLLBACK`. See "Investigated for phase 2" above for the full reasoning per mode. This is a complete, valid phase 2 outcome, not a stalled one — building a weak or false-positive-prone rule just to have shipped *something* would have violated the same 100%-precision-by-construction discipline `rules.py`'s message-text patterns already hold themselves to.
3. **Not started.** Extend `LLMClassifier`'s prompt to recognize the twelve semantic-only modes (ten original, plus 2.1 and 3.3 from phase 2's finding), evaluate it against real multi-agent traces, and only then consider whether any of them earn a stable `FailureType` member — each one needs its own answer to "how does a classifier actually tell this apart from the others," not just a taxonomy citation.

## Non-goals

This is not a plan to add 14 new `FailureType` members. It is not a plan to build a multi-agent orchestration framework — triage still wraps whatever callable you give it, single- or multi-agent. Phases 1 and 2 are done (see above); phase 3 is not.
