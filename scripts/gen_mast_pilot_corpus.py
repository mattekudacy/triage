"""
scripts/gen_mast_pilot_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/mast_pilot_corpus.json — the pilot corpus recommended by
docs/concepts/multi-agent-failures.md's "Phase 3 scoping" section: source
real, cited examples for a *small* subset of the twelve semantic-only MAST
modes (rather than all twelve at once) before committing to a full corpus
build or any change to LLMClassifier's stable contract.

Three modes piloted, per the scoping doc's own recommendation:

  2.5 Ignored Other Agent's Input   — best real-world visibility of the three
  2.4 Information Withholding       — same
  1.4 Loss of Conversation History  — overlaps with single-agent "context
                                       truncation" reports frameworks already
                                       file, so real examples were expected
                                       to be easier to find than most modes

Every entry is sourced from a real, cited GitHub issue — transcribed from the
issue's own description, not invented to make a mode look detectable, same
discipline as tests/data/error_corpus_{a,b,c,d,e}.json. The goal was two
independently-sourced entries per mode (different frameworks each time),
because a single example per mode is exactly the "tuned to one example" risk
this project already measured once for rules.py's v1.1 pattern pass (corpus
C -> D). 2.5 and 1.4 got two clean entries each. 2.4 got only one: its
second real candidate turned out ambiguous with 2.5 rather than a clean
second source (below) — reported as-is, not padded to hit the target, since
forcing a second 2.4 example that didn't meet the same bar would defeat the
point of the target in the first place.

Each entry is a *trajectory* (multiple Steps, some carrying agent_id), not a
single error string like the FailureType corpora — MAST modes are judgments
about interaction between agents, so a one-line error string can't represent
one. Steps are reconstructed to match the issue's own narrated reproduction
as closely as possible; where the issue doesn't narrate an exact tool call or
output, the step's action/llm_output paraphrases the issue's own words
(flagged in "notes") rather than inventing plausible-sounding detail.

IMPORTANT — an honest finding surfaced while sourcing, not smoothed over:
openai/openai-agents-python#348 ("context management between the agents in a
multi agent set up") is genuinely ambiguous between 2.4 (agent 1 has the
modification but the wiring never carries it forward — information
withholding) and 2.5 (agent 3's output ignores what agent 2 produced —
ignored input) purely from the reporter's own description; the issue itself
suspects "context sharing" without pinning which side is at fault. Rather
than force one label, this entry carries `"ambiguous_with"` and no single
`"mast_mode"` — see docs/concepts/multi-agent-failures.md's "Phase 3 scoping"
section for what this implies about the difficulty of the labeling task
itself, independent of whether a classifier could ever resolve it.

This corpus has NO negative examples yet (trajectories that look like a MAST
mode but aren't one) — unlike error_corpus_ambiguous.json's
tricky_but_classifiable/unknown_labeled split, sourcing real negative
multi-agent examples at the same citation standard is follow-up work, not
done here. Any recall number this corpus produces is one-sided: it can show
whether an experimental prompt finds real positives, not whether it also
avoids false-positiving on trajectories that merely resemble a mode. Do not
quote a recall number from this corpus as if it were a precision measurement
too.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_mast_pilot_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUTPUT_PATH = Path("tests/data/mast_pilot_corpus.json")

ENTRIES: list[dict[str, Any]] = [
    # ── 2.5 Ignored Other Agent's Input ─────────────────────────────────────
    {
        "mast_mode": "2.5",
        "mast_name": "Ignored Other Agent's Input",
        "task": "Supervisor reinterprets a user's food request and delegates to a "
        "specialist worker agent to retrieve matching dishes.",
        "steps": [
            {
                "index": 0,
                "agent_id": "supervisor",
                "action": "reinterpret_user_request",
                "llm_output": "Retrieve Japanese dishes (user said 'Italian' but "
                "prior turns establish they mean Japanese cuisine).",
            },
            {
                "index": 1,
                "agent_id": "dishes_browser",
                "action": "retrieve_dishes",
                "tool_called": "browse_dishes",
                "tool_input": {"cuisine": "Italian"},
                "llm_output": "Here are some Italian dishes: ...",
            },
        ],
        "provenance": "transcribed:FlowiseAI-flowise-issue-3512",
        "source_url": "https://github.com/FlowiseAI/Flowise/issues/3512",
        "quote": "In a multi-agent setup using agentflow, the workers rely solely on "
        "chat history and do not access Supervisor instructions. ... The "
        "Supervisor's instruction correctly specifies 'Retrieve Japanese "
        "dishes ...,' but the dishes_browser worker outputs Italian dishes "
        "instead.",
        "notes": "Worker agent processes only the raw user message from chat "
        "history, disregarding the supervisor's explicit reinterpretation — a "
        "clean instance of not properly considering another agent's input.",
    },
    {
        "mast_mode": "2.5",
        "mast_name": "Ignored Other Agent's Input",
        "task": "SelectorGroupChat picks the next speaker from a model response that "
        "reasons about several agents before naming the actual next speaker.",
        "steps": [
            {
                "index": 0,
                "agent_id": "agent_c",
                "action": "propose_next_speaker",
                "llm_output": "<thinking> AgentA or AgentB could handle this "
                "next, but the correct choice given the current state is "
                "AgentC. </thinking> Next: AgentC.",
            },
            {
                "index": 1,
                "agent_id": "selector_group_chat_manager",
                "action": "select_next_speaker",
                "llm_output": "Selected AgentA (mentioned most frequently in the prior message).",
            },
        ],
        "provenance": "transcribed:microsoft-autogen-issue-6891",
        "source_url": "https://github.com/microsoft/autogen/issues/6891",
        "quote": "AgentA and AgentB get mentioned more (inside <thinking>), while "
        "the actual intended next speaker (AgentC) is mentioned once — in the "
        "final suggestion ... _mentioned_agents() tallies all mentions "
        "blindly, [so] it may incorrectly pick AgentA or AgentB.",
        "notes": "Borderline case, included anyway: the root cause is a selection "
        "*mechanism* bug (raw mention-counting) rather than one agent "
        "deliberately disregarding another's stated conclusion, but the "
        "observable trajectory is indistinguishable from 2.5 — the system "
        "acts on a different agent's suggestion than the one actually given. "
        "Worth keeping precisely because it tests whether an experimental "
        "prompt over-fires on 'wrong speaker chosen' generally, or correctly "
        "keys on the ignored final suggestion.",
    },
    # ── 2.4 Information Withholding ─────────────────────────────────────────
    {
        "mast_mode": "2.4",
        "mast_name": "Information Withholding",
        "task": "Agent A performs a file search over an uploaded document, then hands "
        "off the conversation to Agent B to continue file-based analysis.",
        "steps": [
            {
                "index": 0,
                "agent_id": "agent_a",
                "action": "file_search",
                "tool_called": "file_search",
                "tool_input": {"query": "summarize uploaded document"},
                "tool_output": "relevant excerpts from uploaded.pdf",
            },
            {
                "index": 1,
                "agent_id": "agent_a",
                "action": "handoff",
                "llm_output": "Handing off to Agent B to continue.",
            },
            {
                "index": 2,
                "agent_id": "agent_b",
                "action": "file_search",
                "tool_called": "file_search",
                "tool_input": {"query": "summarize uploaded document"},
                "error": "no file resources available",
            },
        ],
        "provenance": "transcribed:danny-avila-librechat-issue-10569",
        "source_url": "https://github.com/danny-avila/LibreChat/issues/10569",
        "quote": "file resources attached to the source agent's tool_resources are "
        "not transferred to the destination agent ... Upon handoff, Agent B "
        "cannot access the previously uploaded file for file search.",
        "notes": "Root cause is a framework wiring bug rather than a deliberate "
        "choice, but MAST's own definition doesn't require intent — 'possesses "
        "critical information but fails to share it' describes the observable "
        "trajectory regardless of why the sharing failed.",
    },
    {
        # No "mast_mode" here, deliberately — this entry's ground truth is
        # genuinely ambiguous between 2.4 and 2.5 (see module docstring and
        # "ambiguous_with" below). A single primary label would misrepresent
        # what was actually found while sourcing.
        "task": "Three agents modify a piece of generated code in sequence; each is "
        "supposed to build on the previous agent's edits.",
        "steps": [
            {
                "index": 0,
                "agent_id": "agent_1",
                "action": "generate_code",
                "llm_output": "def process(): ...  # base implementation",
            },
            {
                "index": 1,
                "agent_id": "agent_2",
                "action": "add_functionality",
                "llm_output": "def process(): ...  # added validation logic",
            },
            {
                "index": 2,
                "agent_id": "agent_3",
                "action": "add_functionality",
                "llm_output": "def process(): ...  # final output, missing "
                "agent_2's validation logic",
            },
        ],
        "provenance": "transcribed:openai-openai-agents-python-issue-348",
        "source_url": "https://github.com/openai/openai-agents-python/issues/348",
        "quote": "the final output is not having the functionality which must have "
        "been provided by the agent 2 and 3, so i doubt there are some issue "
        "in the context sharing between agents",
        "ambiguous_with": "2.5",
        "notes": "AMBIGUOUS GROUND TRUTH, kept unlabeled deliberately — see module "
        "docstring. Could equally be framed as agent_2's contribution being "
        "withheld from agent_3 (2.4) or agent_3 ignoring agent_2's output "
        "(2.5); the reporter's own account doesn't resolve which side is at "
        "fault, and neither does the trace. Excluded from per-mode recall "
        "scoring in scripts/mast_mode_pilot_accuracy.py — reported separately.",
    },
    # ── 1.4 Loss of Conversation History ────────────────────────────────────
    {
        "mast_mode": "1.4",
        "mast_name": "Loss of Conversation History",
        "task": "Chat agent with persistent memory answers questions about facts the "
        "user stated earlier in the conversation.",
        "steps": [
            {
                "index": 0,
                "action": "user_states_fact",
                "llm_output": "Hi! I'm Bob.",
            },
            {
                "index": 1,
                "action": "user_unrelated_query",
                "llm_output": "1+1=?",
            },
            {
                "index": 2,
                "action": "user_asks_recall",
                "llm_output": "Do you remember my name?",
                "error": "agent could not recall the user's name",
            },
        ],
        "provenance": "transcribed:langchain-ai-langgraph-issue-2395",
        "source_url": "https://github.com/langchain-ai/langgraph/issues/2395",
        "quote": "the MemorySave can only remember 1 message. If I ask 'hi! I'm "
        "bob.' then 'do you remember my name?', the agent can respond "
        "correctly. However, between these two queries add a useless query "
        "like '1+1=?', the agent would forget my name.",
        "notes": "Single-agent in this specific report, but the mechanism (an "
        "intervening turn silently evicts earlier context) is the same "
        "mechanism multi-agent handoffs hit at scale — see the second entry.",
    },
    {
        "mast_mode": "1.4",
        "mast_name": "Loss of Conversation History",
        "task": "Agent A completes a task and hands off to Agent B to continue, "
        "passing along the accumulated conversation context.",
        "steps": [
            {
                "index": 0,
                "agent_id": "agent_a",
                "action": "complete_task",
                "llm_output": "Task complete. Context window nearly full.",
            },
            {
                "index": 1,
                "agent_id": "agent_a",
                "action": "handoff",
                "llm_output": "Handing off to Agent B with accumulated context.",
            },
            {
                "index": 2,
                "agent_id": "agent_b",
                "action": "continue_task",
                "error": "insufficient context window remaining to complete task",
            },
        ],
        "provenance": "transcribed:github-copilot-cli-issue-1180",
        "source_url": "https://github.com/github/copilot-cli/issues/1180",
        "quote": "Agent A has handoff:true for the Agent B. After the Agent A "
        "completes the task, the context window is nearly full. It leads to "
        "the agent B won't have enough context window to properly do the "
        "task.",
        "notes": "The feature request's proposed fix — compact history and hand "
        "Agent B a refreshed window — implies the failure mode here is losing "
        "usable history to bloat, not a clean reset (contrast with 2.1 "
        "Conversation Reset's autogen#1942/langgraph#6064 citations already "
        "in this doc, which are abrupt discontinuities, not gradual "
        "crowding-out).",
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(ENTRIES, indent=2) + "\n")
    scored = [e for e in ENTRIES if "ambiguous_with" not in e]
    ambiguous = [e for e in ENTRIES if "ambiguous_with" in e]
    print(f"Wrote {len(ENTRIES)} entries to {OUTPUT_PATH}")
    print(f"  {len(scored)} scorable (single ground-truth label)")
    print(f"  {len(ambiguous)} ambiguous ground truth, reported separately")


if __name__ == "__main__":
    main()
