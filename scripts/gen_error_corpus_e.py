"""
scripts/gen_error_corpus_e.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_e.json — the corpus that answers Step 2 of
docs/known-limitations.md's "Corpus E scoping" plan: does RulesClassifier's
new structured-error-code matching (Step.metadata["http_status"] /
["json_rpc_code"], shipped in the PR that added _JSON_RPC_*/_HTTP_* tables to
triage/classifier/rules.py) actually improve routing-sensitive recall on
fresh, unseen sources — or does it hit the same wall the v1.1 message-text
tuning pass did against corpus D?

Sources are disjoint from A, B, C, and D:
  MCP (JSON-RPC codes, not just message text this time — see below)
  Together AI, Fireworks AI, Replicate, Cerebras, Perplexity, DeepSeek,
  NVIDIA NIM, xAI (Grok) — all fresh vendors, none appear in A-D.

Every entry that carries a code was sourced from a real, cited exception,
GitHub issue, or vendor's own docs — the same discipline as A-D. Unlike A-D,
each entry may also carry "metadata": {"http_status": int} or
{"json_rpc_code": int}, matching the Step.metadata convention documented in
triage/taxonomy.py and triage/classifier/rules.py.

IMPORTANT — this corpus is NOT built to make the new feature look good. Most
of the HTTP-status entries below use codes deliberately NOT in
_HTTP_EXTERNAL_STATUS_CODES/_HTTP_TIMEOUT_STATUS_CODES (404, 400, 422) —
because that is what real vendors actually return for "wrong tool"/"bad
schema" failures. Only 429/500/502/503/408/504 are mapped, by design (see
rules.py's module docstring — 404/400/422 are excluded as too ambiguous to
resolve safely). If most of this corpus's routing-sensitive entries carry an
unmapped code, that is the honest finding, not a corpus-design mistake — see
the analysis in docs/known-limitations.md after this corpus is scored once.

One entry (the MCP -32600 case) is a deliberate adversarial case, not a
straightforward positive: a real MCP server (per langgenius/dify#22675)
returned -32600 ("Invalid Request" per the JSON-RPC 2.0 spec) for a
session-termination condition, not a malformed request — the issue's own
analysis couldn't rule out an auth/session-lifecycle cause. Its true label
here is "unknown", not "schema_mismatch", specifically to test whether
rules.py's -32600 -> SCHEMA_MISMATCH mapping is as unambiguous in practice as
the JSON-RPC spec suggests on paper. If this scores as a miss (a real server
reusing -32600 loosely, the same pattern that caused corpus D's
OutputParserError/CrewAI collision), that is signal to narrow or drop -32600
from the mapping — see CHANGELOG.md for what happened after this was scored.

A caveat about the two MCP -32601 entries below (claude-vscode and WorkIQ
servers): both surfaced through the exact same wrapper phrasing, "MCP error
-32601: Method not found" / "...MCP error -32601: Method not found" — likely
a shared client-side error-wrapping convention (both were reported via
Claude Code's MCP client), not independent per-server message formats. Their
diversity is in server implementation and root cause, not message wording —
the code field is the generalizable signal under test here, not the text.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus_e.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _entry(
    label: str,
    exc_type: str,
    error: str,
    source: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "exception_type": exc_type,
        "error": error,
        "label": label,
        "provenance": f"transcribed:{source}",
        "metadata": metadata or {},
    }


def build_corpus() -> list[dict]:
    cases: list[dict] = []

    # ── wrong_tool_called ────────────────────────────────────────────────────

    # MCP -32601, spec-unambiguous. Source: anthropics/claude-code#23248
    # ("claude-vscode" server via Cursor). Quoted verbatim from the issue.
    cases.append(
        _entry(
            "wrong_tool_called",
            "McpError",
            "MCP error -32601: Method not found",
            "anthropics-claude-code-issue-23248",
            {"json_rpc_code": -32601},
        )
    )
    # MCP -32601, second real, independent server (WorkIQ). Source:
    # microsoft/work-iq#38. Quoted verbatim — see module docstring's caveat
    # about both -32601 entries sharing the same client-side wrapper phrasing.
    cases.append(
        _entry(
            "wrong_tool_called",
            "McpError",
            "Failed to fetch tools: MCP error -32601: Method not found",
            "microsoft-work-iq-issue-38",
            {"json_rpc_code": -32601},
        )
    )
    # Together AI — OpenAI-compatible 404 model_not_found. HTTP 404 is
    # deliberately NOT in _HTTP_EXTERNAL_STATUS_CODES (too ambiguous — see
    # rules.py) so this is a genuine miss test, not a gimme.
    # Source: drdroid.io Together AI integration-diagnosis page + the
    # standard OpenAI-compatible error envelope Together AI mirrors.
    cases.append(
        _entry(
            "wrong_tool_called",
            "NotFoundError",
            "Error code: 404 - {'error': {'message': \"The model "
            "`meta-llama/Llama-3-bad-id` does not exist or you do not have "
            'access to it.", "type": "invalid_request_error", "code": '
            '"model_not_found"}}',
            "together-ai-drdroid-integration-diagnosis",
            {"http_status": 404},
        )
    )
    # Fireworks AI — 404, exact message quoted from Fireworks' own docs (via
    # search index; direct fetch of docs.fireworks.ai is blocked from this
    # environment, message quoted as returned by the search tool).
    cases.append(
        _entry(
            "wrong_tool_called",
            "NotFoundError",
            "Model not found, inaccessible, and/or not deployed",
            "fireworks-ai-docs-inference-error-codes",
            {"http_status": 404},
        )
    )
    # Replicate — 422, official docs (replicate.com/docs/reference/error-codes,
    # quoted via search index; direct fetch blocked from this environment).
    # 422 is not in any mapped table (only 429/500/502/503/408/504 are) —
    # another genuine miss test, and evidence 422 is common enough for this
    # failure family that it may be worth a future table entry of its own.
    cases.append(
        _entry(
            "wrong_tool_called",
            "ReplicateError",
            "Invalid version or not permitted: The specified version does "
            "not exist (or perhaps you don't have permission to use it?)",
            "replicate-docs-error-codes",
            {"http_status": 422},
        )
    )
    # Perplexity — 400 bad request for an invalid/nonexistent model.
    # Source: community.perplexity.ai bug reports + docs.perplexity.ai's
    # documented status-code table (quoted via search index).
    cases.append(
        _entry(
            "wrong_tool_called",
            "BadRequestError",
            "400 Bad Request: Invalid model specified",
            "perplexity-community-forum-and-docs",
            {"http_status": 400},
        )
    )

    # ── schema_mismatch ──────────────────────────────────────────────────────

    # MCP -32700 Parse error — JSON-RPC 2.0 spec-defined
    # (jsonrpc.org/specification), referenced via mcpevals.io's MCP error-code
    # guide. No single canonical raised-exception string is published for
    # this one (parse errors happen before a request is even attributed to a
    # tool call), so the message is the spec's own description, same
    # convention corpus D used for its "novel phrasings" entries.
    cases.append(
        _entry(
            "schema_mismatch",
            "McpError",
            "Parse error: invalid JSON was received by the server",
            "jsonrpc-2.0-spec-and-mcpevals-io-error-codes",
            {"json_rpc_code": -32700},
        )
    )
    # Replicate — 422, quoted verbatim from replicate-javascript#249.
    # 422 is not mapped — genuine miss test, same as the wrong_tool_called
    # Replicate entry above.
    cases.append(
        _entry(
            "schema_mismatch",
            "ReplicateError",
            "Input validation failed: - Additional property signal is not allowed",
            "replicate-javascript-issue-249",
            {"http_status": 422},
        )
    )
    # Cerebras — 400, per the Cerebras Inference changelog (quoted via search
    # index; direct fetch of inference-docs.cerebras.ai is blocked from this
    # environment): validation errors return 400 (changed from 422),
    # BadRequestError, for missing required fields / malformed bodies /
    # unsupported parameters. 400 is not mapped — genuine miss test.
    cases.append(
        _entry(
            "schema_mismatch",
            "BadRequestError",
            "Missing required field: messages",
            "cerebras-inference-docs-changelog",
            {"http_status": 400},
        )
    )

    # ── unknown — deliberate adversarial case, see module docstring ─────────

    # MCP -32600, quoted verbatim from langgenius/dify#22675. rules.py maps
    # -32600 -> SCHEMA_MISMATCH (JSON-RPC "Invalid Request"), but this real
    # server used it for what the issue's own analysis could not rule out as
    # a session/auth-lifecycle condition rather than a malformed request —
    # true label is UNKNOWN, not SCHEMA_MISMATCH. Whether RulesClassifier's
    # new stage misroutes this is exactly what this entry is for.
    cases.append(
        _entry(
            "unknown",
            "McpError",
            "Failed to connect to MCP server: code=32600 message='Session terminated by server'",
            "langgenius-dify-issue-22675",
            {"json_rpc_code": -32600},
        )
    )

    # ── external_fault (self-healing) ───────────────────────────────────────

    # DeepSeek — 429 rate limit, per DeepSeek's own documented error codes
    # (quoted via search index of platform.deepseek.com-derived docs).
    cases.append(
        _entry(
            "external_fault",
            "RateLimitError",
            "Rate limit reached for requests",
            "deepseek-api-error-codes-docs",
            {"http_status": 429},
        )
    )
    # DeepSeek — 500 internal system fault, same source family.
    cases.append(
        _entry(
            "external_fault",
            "InternalServerError",
            "Server busy, please try again later",
            "deepseek-api-error-codes-docs",
            {"http_status": 500},
        )
    )
    # Perplexity — 503, per docs.perplexity.ai's error-handling guide (quoted
    # via search index) and community reports of the same condition.
    cases.append(
        _entry(
            "external_fault",
            "ServiceUnavailableError",
            "503 Service Unavailable: the model is temporarily overloaded",
            "perplexity-docs-error-handling",
            {"http_status": 503},
        )
    )
    # NVIDIA NIM — 503 unexpected error, per NIM's REST API reference.
    cases.append(
        _entry(
            "external_fault",
            "ServiceUnavailableError",
            "503: unexpected error",
            "nvidia-nim-rest-api-docs",
            {"http_status": 503},
        )
    )

    # ── timeout (self-healing) ───────────────────────────────────────────────

    # DeepSeek — 504 gateway timeout through a proxy, quoted from
    # langgenius/dify#15889 ("DeepSeek Plugin - 504 Gateway Timeout Error").
    cases.append(
        _entry(
            "timeout",
            "GatewayTimeoutError",
            "504 Gateway Timeout",
            "langgenius-dify-issue-15889",
            {"http_status": 504},
        )
    )
    # xAI (Grok) — 504 gateway timeout, per a real reported case in the
    # Zapier community forum ("Grok 4 '504 Gateway Time-out' error").
    cases.append(
        _entry(
            "timeout",
            "GatewayTimeoutError",
            "504 Gateway Time-out",
            "xai-grok-zapier-community-report",
            {"http_status": 504},
        )
    )

    return cases


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_e.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    labels: dict[str, int] = {}
    for c in corpus:
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    for label, n in sorted(labels.items()):
        print(f"  {label}: {n}")
