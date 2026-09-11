"""
scripts/gen_error_corpus_d.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_d.json — corpus D, the new held-out set.

Rules used when building this corpus: NONE — entries were selected and labeled
before running classifier_accuracy.py against them, and rules.py was not
edited afterward based on the result. rules.py WAS tuned against corpus C's
misses immediately before this corpus was built (that is the whole point of
D: C is now training data, D is the fresh, disjoint measurement of whether
that tuning generalized rather than just memorized C's 13 strings).

Sources are deliberately disjoint from A, B, and C:
  A covered: stdlib json/asyncio, httpx, pydantic, openai SDK, anthropic SDK,
             langchain, langgraph
  B covered: botocore, google-genai/gRPC, aiohttp, requests/urllib3, urllib
  C covered: azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI
             (aiplatform), LlamaIndex, novel phrasings

New sources for D:
  huggingface_hub (Inference API client)
  Ollama (local server, JSON error envelope)
  OpenRouter (multi-provider proxy, wraps upstream errors)
  Model Context Protocol (MCP) — JSON-RPC tool-call errors
  CrewAI
  Semantic Kernel (Microsoft, Python)
  Novel structural phrasings not in A, B, or C — chosen to stress the
  *boundaries* of the v1.1 patterns added for corpus C (e.g. "Model repo 'x'
  does not exist" has an extra word between "model" and the quoted name that
  the v1.1 pattern doesn't anticipate; "Unknown tool: x" has a colon the
  existing "unknown tool X" pattern doesn't expect).

Provenance:
  _entry(label, exc_type, error, source) — transcribed from published
  docs/issues/source, or constructed from the documented message shape when
  no single canonical string was published (noted per-entry below).

This corpus is scored exactly once (see tests/test_classifier_accuracy.py's
CORPUS_D_FLOOR and CORPUS_D_ROUTING_SENSITIVE_FLOOR) and then frozen, same
discipline as corpus C before it.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus_d.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _entry(label: str, exc_type: str, error: str, source: str) -> dict:
    return {
        "exception_type": exc_type,
        "error": error,
        "label": label,
        "provenance": f"transcribed:{source}",
    }


def build_corpus() -> list[dict]:
    cases: list[dict] = []

    # ── huggingface_hub (Inference API client) ──────────────────────────────
    # Source: https://github.com/huggingface/huggingface_hub/issues/2060
    #         https://discuss.huggingface.co/t/hf-inference-api-503-504-server-error/148267
    cases.append(
        _entry(
            "external_fault",
            "HfHubHTTPError",
            "503 Server Error: Service Temporarily Unavailable for url: "
            "https://api-inference.huggingface.co/models/gpt2",
            "huggingface_hub-github-issue-2060",
        )
    )
    # Source: huggingface_hub InferenceTimeoutError — raised when a model is
    # still loading on the serverless endpoint; message documented across
    # HF forum threads on the 503/504 loading behavior.
    cases.append(
        _entry(
            "timeout",
            "InferenceTimeoutError",
            "Model microsoft/DialoGPT-medium is currently loading. "
            "Please retry with a higher timeout.",
            "huggingface_hub-docs",
        )
    )
    # Constructed from huggingface_hub.utils.RepositoryNotFoundError's
    # documented purpose (raised when a repo id doesn't resolve on the Hub);
    # exact wording varies by hub version, this is the documented shape.
    cases.append(
        _entry(
            "wrong_tool_called",
            "RepositoryNotFoundError",
            "Model repo 'invalid/model-name' does not exist",
            "huggingface_hub-docs",
        )
    )

    # ── Ollama (local server) ────────────────────────────────────────────────
    # Source: https://github.com/ollama/ollama/issues/2203 — the 404 JSON
    # envelope; the Python client re-raises the "error" field as the message.
    cases.append(
        _entry(
            "wrong_tool_called",
            "ResponseError",
            "model 'codellama:7b-instruct-q6_K' not found, try pulling it first",
            "ollama-github-issue-2203",
        )
    )
    # Ollama's Go server surfaces raw JSON-decode failures from malformed
    # tool-call arguments through this Go-standard-library phrasing (no
    # "json"/"parse"/"decode" keyword — the classifier's _SCHEMA_RE was
    # written against Python's json.JSONDecodeError phrasings, not Go's).
    cases.append(
        _entry(
            "schema_mismatch",
            "ResponseError",
            "invalid character '}' looking for beginning of value",
            "ollama-go-json-error-shape",
        )
    )

    # ── OpenRouter (multi-provider proxy) ────────────────────────────────────
    # Source: https://openrouter.ai/docs — error.code/error.message envelope;
    # 429 responses carry X-RateLimit-* headers and a rate-limit message.
    cases.append(
        _entry(
            "external_fault",
            "RateLimitError",
            "Rate limit exceeded: free-tier models are limited to 20 requests per minute",
            "openrouter-docs",
        )
    )
    # 402 Payment Required — insufficient account balance. Not transient and
    # not a classifiable failure type in the current taxonomy; same status as
    # corpus C's IAM permission_denied entry (label: unknown, by design).
    cases.append(
        _entry(
            "unknown",
            "PaymentRequiredError",
            "Insufficient credits. Please top up your account to continue.",
            "openrouter-docs",
        )
    )
    # OpenRouter proxies the upstream 400 body largely unchanged when a
    # provider rejects a malformed request shape.
    cases.append(
        _entry(
            "schema_mismatch",
            "BadRequestError",
            "Invalid request: 'messages' must be a non-empty array",
            "openrouter-docs",
        )
    )

    # ── Model Context Protocol (MCP) — JSON-RPC tool errors ──────────────────
    # Source: https://github.com/modelcontextprotocol/typescript-sdk/issues/1510
    # Protocol error code -32602; message format is "Unknown tool: <name>".
    cases.append(
        _entry(
            "wrong_tool_called",
            "McpError",
            "Unknown tool: invalid_tool_name",
            "mcp-typescript-sdk-issue-1510",
        )
    )
    # Generic JSON-RPC -32601 "Method not found" — MCP servers return this
    # verbatim for an unregistered tool/method; it carries no "tool" keyword
    # at all, unlike the -32602 variant above.
    cases.append(
        _entry(
            "wrong_tool_called",
            "McpError",
            "Method not found",
            "mcpevals-io-error-codes",
        )
    )

    # ── CrewAI ────────────────────────────────────────────────────────────────
    # Source: https://community.crewai.com/t/tools-dont-exist/313 — CrewAI's
    # ReAct-style output parser reports an unrecognized Action this way.
    cases.append(
        _entry(
            "wrong_tool_called",
            "OutputParserError",
            "Action 'search_the_web' don't exist, these are the only available "
            "Actions: web_search, calculator",
            "crewai-community-forum",
        )
    )
    # crewai-tools are pydantic BaseModel schemas; a missing required
    # argument raises plain pydantic ValidationError (already in
    # _SCHEMA_EXCEPTION_TYPES — exercises the exception-type fallback path).
    cases.append(
        _entry(
            "schema_mismatch",
            "ValidationError",
            "Arguments validation failed: tool_input\n  field required (type=value_error.missing)",
            "crewai-tools-pydantic-shape",
        )
    )

    # ── Semantic Kernel (Microsoft, Python) ──────────────────────────────────
    # Source: semantic_kernel.functions.kernel_function_extension — documented
    # exact message: "Function '{function_name}' not found in any plugin."
    cases.append(
        _entry(
            "wrong_tool_called",
            "KernelFunctionNotFoundError",
            "Function 'get_weather' not found in any plugin.",
            "semantic-kernel-source",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "TimeoutError",
            "Operation timed out while waiting for the completion.",
            "semantic-kernel-azure-openai-wrapper",
        )
    )

    # ── Novel structural phrasings ────────────────────────────────────────────
    # Phrases chosen to stress the boundaries of the v1.1 patterns added for
    # corpus C, not to repeat them.
    cases.append(
        _entry(
            "external_fault",
            "OverloadedError",
            # Anthropic's 529 "overloaded" code isn't in _EXTERNAL_CODE_RE
            # (only 429/500/502/503) — this exercises the exception-type
            # fallback instead, which already lists OverloadedError.
            "Upstream provider returned a non-2xx status: 529",
            "novel-529-overloaded",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "OutputValidationError",
            # Deliberately avoids "validation error" / "json schema" / "parse"
            # — none of the SCHEMA_RE alternatives or _SCHEMA_EXCEPTION_TYPES
            # cover this exact phrasing or exception type name.
            "Failed to coerce output to schema: missing required key 'answer'",
            "novel-schema-coerce",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "ValueError",
            # No "tool"/"function"/"model"/"endpoint" keyword at all.
            "Agent attempted to call an undefined action 'convert_currency'",
            "novel-undefined-action",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "OperationCanceledError",
            "The operation was canceled due to timeout of 30000ms exceeded",
            "novel-canceled-timeout",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "RuntimeError",
            # Deliberately generic exception type (not TooManyRequestsError)
            # and message text that avoids "rate limit" — tests the message
            # pattern alone, not the v1.1 exception-type fallback.
            "Too many requests, please slow down",
            "novel-generic-rate-limit",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "KeyError",
            "No handler registered for action 'summarize_document'",
            "novel-unregistered-handler",
        )
    )

    return cases


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_d.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    labels = (
        "wrong_tool_called",
        "schema_mismatch",
        "external_fault",
        "timeout",
        "unknown",
    )
    for label_val in labels:
        n = sum(1 for c in corpus if c["label"] == label_val)
        if n:
            print(f"  {label_val}: {n}")
