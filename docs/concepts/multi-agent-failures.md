# Multi-Agent Failures: MAST Scoping

**Status: a design proposal, not yet implemented.** Nothing in this document has shipped. It exists to answer one question concretely before any code is written: what would it actually take for `triage` to detect multi-agent failure modes, and how much of that can be done honestly with the same rigor the rest of this project holds itself to — measured, not assumed, and never a new `FailureType` without real disambiguation logic behind it.

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

The minimal, additive fix: `Step.agent_id: str | None = None`, the same low-risk pattern `Step.metadata` already established — an optional field with zero effect on any existing caller. It connects directly to work already shipped: the OTel GenAI semantic conventions already define `gen_ai.agent.id`/`gen_ai.agent.name` on `invoke_agent` spans (verified directly against the spec), so `triage.observability.otel_ingest.trajectory_from_spans()` could capture agent identity the same way it already captures `tool_called`/`http_status` — real frameworks that emit per-agent spans would populate this for free, the same story as the HTTP-status connection to `RulesClassifier`'s structured-code matching.

## Mapping the 14 modes

Honesty matters more here than coverage. Three groups, by how they'd actually get detected:

### Already covered by an existing `FailureType` — no new member needed

| MAST mode | Existing mapping |
|---|---|
| 1.3 Step Repetition | `LOOP_DETECTED` — triage already detects this for one agent (`_is_loop_window()` in `rules.py`). Once `agent_id` exists, the same mechanism extends to catch a step repeated *across* agents — a small, concrete code change, not a new concept. |
| 3.1 Premature Termination | `PLAN_INCOMPLETE` — CLAUDE.md's own taxonomy table already defines this as "agent declared success but not all required sub-goals were completed," which *is* MAST's 3.1 definition, just framed for one agent. Needs a docs update connecting the two, not a new type. |

Finding two direct hits here — including one where triage already has *working, tested code* for the single-agent case — is the strongest evidence this taxonomy is worth aligning with rather than inventing categories from scratch.

### Structurally promising — worth a real prototype + measurement, not yet built

| MAST mode | Structural proxy | Why it's plausible |
|---|---|---|
| 3.3 No or Incorrect Verification | Compare a step's stated outcome (`llm_output`, e.g. "tests pass") against its `tool_output`/`error` content (e.g. a non-zero exit code or "FAILED" in output) | This is a message-text-pattern problem, the exact shape `RulesClassifier` already handles — no semantic understanding needed, just a mismatch between a claim and a result already sitting in two `Step` fields |
| 2.1 Conversation Reset | An unexpected drop back to an earlier checkpoint's state, detectable via triage's *existing* checkpoint/`_current_state` machinery | Reuses infrastructure that already exists for an unrelated reason (rollback), rather than building new detection from nothing |

Both need a real corpus before becoming a `RulesClassifier` rule — same discipline as `rules.py`'s existing patterns, not a guess shipped straight to the taxonomy.

### Semantic-only — `LLMClassifier` extension, not `RulesClassifier`

1.1, 1.2, 1.4, 1.5, 2.2, 2.3, 2.4, 2.5, 2.6, 3.2 — ten of the fourteen. Each requires understanding *meaning* (was this role-appropriate, was this genuinely unclear, did this deviate from intent) that no message-text pattern or structured code can reach — the identical limitation already documented for `PLAN_INCOMPLETE`/`CONTEXT_OVERFLOW`. These are candidates for extending `LLMClassifier`'s prompt to recognize MAST-flavored descriptions in a multi-agent trajectory, not new pattern rules.

## What this means for `FailureType` — and the mistake not to repeat

CLAUDE.md documents exactly this failure mode already happening once: `HALLUCINATED_STATE`/`GOAL_DRIFT` were added in an earlier version and removed in v0.7 because `LLMClassifier` had no real disambiguation logic for them — they looked good as taxonomy entries and didn't work as a classifier. Adding MAST's ten semantic-only modes as new `FailureType` members today, before any classifier (rules or LLM) can reliably tell them apart, would repeat that mistake at 10x the scale.

**Recommendation: no new `FailureType` members in phase 1.** The two direct hits (1.3, 3.1) need no new type at all. The two structural candidates (3.3, 2.1) are worth prototyping as new `RulesClassifier` rules under strong existing types once measured — not necessarily new enum members either, since "wrong verification" plausibly still routes to existing types depending on what the underlying task actually was. The ten semantic-only modes should wait for a working, *measured* `LLMClassifier` prompt extension before any of them become a stable enum member — consistent with how this project now treats every accuracy claim: prove it against real data first.

## Recommended phasing

1. **`Step.agent_id: str | None = None`** — pure additive field, zero behavior change for existing callers. Extend `otel_ingest.trajectory_from_spans()` to populate it from `gen_ai.agent.id`/`gen_ai.agent.name` when present. Extend `LOOP_DETECTED`'s matching to also catch identical steps across different `agent_id`s. This alone ships something real: cross-agent loop detection, using code that already exists and is already tested.
2. **Prototype 3.3 (verification mismatch) and 2.1 (conversation reset)** as candidate `RulesClassifier` rules, validated against a constructed corpus of real multi-agent framework traces (AutoGen, CrewAI, LangGraph multi-agent) — same sourcing discipline as `tests/data/error_corpus_*.json`, not synthetic examples built to make the rule look good.
3. **Only after (1) and (2) ship and are measured:** extend `LLMClassifier`'s prompt to recognize the ten semantic-only modes, evaluate it against real multi-agent traces, and only then consider whether any of them earn a stable `FailureType` member — each one needs its own answer to "how does a classifier actually tell this apart from the others," not just a taxonomy citation.

## Non-goals

This is not a plan to add 14 new `FailureType` members. It is not a plan to build a multi-agent orchestration framework — triage still wraps whatever callable you give it, single- or multi-agent. It is not started — `git blame` on this file should show it arriving with no corresponding change to `taxonomy.py`.
