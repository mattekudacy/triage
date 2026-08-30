#!/usr/bin/env python3
"""
risk_gate.py
~~~~~~~~~~~~
Claude Code PreToolUse hook: scores the pending tool call with triage's
RulesRiskScorer and emits an allow/ask/deny decision.

Deterministic, zero-API-call, sub-millisecond regex scoring — the same
scorer used inside `triage.Agent(risk_scorer=..., risk_threshold=...)`
(see examples/risk_guard.py in the triage repo), applied here directly to
a Claude Code tool call instead of an agent trajectory step.

Reads a PreToolUse JSON payload from stdin:
    {"tool_name": "Bash", "tool_input": {"command": "rm -rf /data"}, ...}

Writes one JSON line to stdout:
    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                             "permissionDecision": "allow"|"ask"|"deny",
                             "permissionDecisionReason": "..."}}

Fails open: any internal error (triage not installed, malformed payload,
scorer failure) is caught in one place, logged to stderr, and resolved as
"allow" so a bug in this script can never block or hang every tool call in
a session. See claude-plugin/README.md for the tradeoff this makes.
"""

from __future__ import annotations

import json
import os
import sys

DENY_THRESHOLD_DEFAULT = 0.9
ASK_THRESHOLD_DEFAULT = 0.5

# Fields worth folding into the text the scorer inspects, for tool calls
# other than Bash. Deliberately narrow — identifier-like fields only, never
# full file/content bodies, which would false-positive on words like
# "update" appearing in ordinary prose or code.
INTERESTING_KEYS = (
    "path",
    "file_path",
    "url",
    "to",
    "query",
    "recipient",
    "table",
    "endpoint",
    "command",
)


def load_config() -> dict[str, object]:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            data: dict[str, object] = json.load(f)
            return data
    return {}


def build_action_text(tool_name: str, tool_input: dict[str, object]) -> str:
    if tool_name == "Bash":
        return str(tool_input.get("command", ""))

    # MCP tool names are namespaced as "mcp__<server>__<tool>". "_" is a
    # \w character, so "__" is not a regex word boundary — a pattern like
    # \bsend_email\b never matches inside "mcp__gmail__send_email"
    # unstripped. Score the bare tool name instead; the full namespaced
    # name is still preserved separately as step.tool_called.
    name_for_scoring = tool_name.split("__")[-1] if tool_name.startswith("mcp__") else tool_name

    parts = [name_for_scoring] + [str(tool_input[k]) for k in INTERESTING_KEYS if k in tool_input]
    return " ".join(parts)


def emit(decision: str, reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}

        cfg = load_config()
        deny_threshold = float(cfg.get("deny_threshold", DENY_THRESHOLD_DEFAULT))  # type: ignore[arg-type]
        ask_threshold = float(cfg.get("ask_threshold", ASK_THRESHOLD_DEFAULT))  # type: ignore[arg-type]
        high_risk_patterns = cfg.get("high_risk_patterns") or None
        medium_risk_patterns = cfg.get("medium_risk_patterns") or None
        assert high_risk_patterns is None or isinstance(high_risk_patterns, list)
        assert medium_risk_patterns is None or isinstance(medium_risk_patterns, list)

        # Deferred import: a missing/broken `triage` install lands in the
        # same fail-open branch as any other internal error below.
        from triage import RulesRiskScorer, Step
        from triage.trajectory import Trajectory

        scorer = RulesRiskScorer(
            high_risk_patterns=high_risk_patterns,
            medium_risk_patterns=medium_risk_patterns,
        )
        step = Step(
            index=0,
            action=build_action_text(tool_name, tool_input),
            tool_called=tool_name,
            tool_input=tool_input,
        )
        result = scorer(step, Trajectory())

        score_str = f"score={result.score:.2f}"
        if result.score >= deny_threshold:
            emit("deny", f"triage-risk-gate: {result.reason or 'high risk'} ({score_str})")
        elif result.score >= ask_threshold:
            emit("ask", f"triage-risk-gate: {result.reason or 'medium risk'} ({score_str})")
        else:
            emit("allow", f"triage-risk-gate: low risk ({score_str})")

    except Exception as exc:  # noqa: BLE001 - fail-open: never brick the session over a gate bug
        print(
            f"[triage-risk-gate] WARNING: gate failed open ({exc!r}); "
            f"tool call NOT scored, allowed by default. "
            f"Check that `triage-agent` is installed for the python3 this hook runs under.",
            file=sys.stderr,
        )
        # Emit an explicit "allow" decision rather than relying on exit 0's
        # implicit default, so the fail-open outcome is visible in Claude's
        # own transcript, not just on stderr.
        emit("allow", f"triage-risk-gate: gate failed open ({exc}); tool call not scored")


if __name__ == "__main__":
    main()
