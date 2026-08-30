# Triage Risk Gate

A Claude Code plugin that intercepts destructive tool calls — `rm -rf`, `drop table`,
`send_email`, `charge_card`, and similar — **before** they execute, using
[`triage`](https://github.com/mattekudacy/triage)'s `RulesRiskScorer`: a deterministic,
regex-based scorer with zero API calls and sub-millisecond latency.

The point of using a rules scorer instead of an LLM here is deliberate: you don't want the
model judging whether its own `rm -rf` is safe. This gate runs outside the model, on plain
pattern matching, so its behavior is auditable and doesn't depend on what the model decides
to tell you.

This plugin is a thin wrapper — all the actual risk logic lives in
[`triage/scorer/rules.py`](../triage/scorer/rules.py) in the main package. For the in-process
equivalent (wiring the same scorer into a `triage.Agent` you're building yourself), see
[`examples/risk_guard.py`](../examples/risk_guard.py).

## What it does

On every `Bash` or MCP (`mcp__*`) tool call, the `PreToolUse` hook:

1. Builds a `triage.Step` from the tool name and input.
2. Scores it with `RulesRiskScorer`.
3. Emits one of three decisions:
   - **`deny`** (score ≥ 0.9) — blocks the call, tells Claude why.
   - **`ask`** (0.5 ≤ score < 0.9) — prompts you for confirmation.
   - **`allow`** (score < 0.5) — lets it through.

## Install

This plugin is not published to a marketplace yet. Point Claude Code's plugin config at this
directory (a local path, or a `git clone` of this repo), then enable `triage-risk-gate` — it
ships with `"defaultEnabled": false`, so it won't activate silently.

The hook script needs `triage-agent` importable by whatever `python3` runs it:

```bash
pip install triage-agent
```

No optional extras (`[anthropic]`, `[langgraph]`, etc.) are required — the scorer only
touches stdlib, `anyio`, and `pydantic`.

## Two tradeoffs worth knowing about

**It fails open.** If `import triage` fails, or anything else inside the hook throws, the
call is allowed through with a stderr warning and an explicit `allow` decision — the gate
never blocks or hangs a tool call because of its own bug or a broken environment. The
alternative (fail closed — `ask` on any internal error) is more defensive against a broken
`triage` install silently disabling the gate, but means a misconfigured `python3` turns into
a confirmation prompt on every matched call until fixed. This plugin chooses fail-open because
it's a supplementary safety net on top of Claude Code's own permission system, not the only
line of defense.

**It doesn't match `Write`/`Edit` by default.** The scorer's medium-risk pattern includes
bare words like `write`, `create`, `update` — matching Claude Code's built-in file-editing
tools by name would fire an `ask` prompt on nearly every ordinary file edit in a coding
session. The default matcher (`hooks/hooks.json`) is `Bash|mcp__.*`: shell commands and
MCP/custom tools whose name is itself the action (`send_email`, `charge_card`). If you want
broader coverage, edit the `matcher` field yourself — the tradeoff is more noise per session.

## Configuration

Copy `config.example.json` to `config.json` in this directory to customize thresholds or add
your own patterns:

```json
{
  "high_risk_patterns": ["wire_transfer"],
  "medium_risk_patterns": ["notify_customer"],
  "deny_threshold": 0.9,
  "ask_threshold": 0.5
}
```

- `high_risk_patterns` / `medium_risk_patterns`: extra literal strings ORed into the built-in
  patterns (escaped, not raw regex — same contract as `RulesRiskScorer.__init__`).
- `deny_threshold` / `ask_threshold`: override the score cutoffs above.

No `config.json` is committed to this repo — its absence means pure defaults. Tool-name
matcher scope (which tools the hook even sees) is fixed in `hooks/hooks.json`, not this file —
Claude Code resolves the matcher at hook-load time, not per call.

## Testing

```bash
python3 claude-plugin/tests/test_risk_gate.py
```

Standalone harness — not part of the `triage` pytest suite, since this plugin lives outside
`triage/` (see `.claude/rules/core.md` Rule 1: no tool-specific code in core). Pipes sample
`PreToolUse` payloads into `scripts/risk_gate.py` and checks the resulting decision, the
matcher's `Write` exclusion, and the fail-open path.
