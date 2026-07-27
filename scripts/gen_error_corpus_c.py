"""
scripts/gen_error_corpus_c.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_c.json — corpus C, the new held-out set.

Rules used when building this corpus: NONE. rules.py was not consulted.
Sources are deliberately disjoint from both corpus A and corpus B:
  A covered: stdlib json/asyncio, httpx, pydantic, openai SDK, anthropic SDK,
             langchain, langgraph
  B covered: botocore, google-genai/gRPC, aiohttp, requests/urllib3, stdlib urllib

New sources for C:
  azure-core / azure-sdk-for-python exceptions
  Mistral AI SDK (mistralai)
  Cohere SDK
  Groq SDK
  LiteLLM exception wrappers
  google-cloud-aiplatform (Vertex AI — different format from google-genai SDK)
  LlamaIndex agent errors
  Novel structural phrasings not in A or B

Provenance:
  _capture(label, fn)              — exception actually raised and captured
  _entry(label, exc_type, error, source) — transcribed from published source

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus_c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _capture(label: str, fn) -> dict | None:
    try:
        fn()
        print(f"  SKIP (no exception): {label}", file=sys.stderr)
        return None
    except BaseException as e:
        return {
            "exception_type": type(e).__name__,
            "error": str(e),
            "label": label,
            "provenance": "captured",
        }


def _entry(label: str, exc_type: str, error: str, source: str) -> dict:
    return {
        "exception_type": exc_type,
        "error": error,
        "label": label,
        "provenance": f"transcribed:{source}",
    }


def build_corpus() -> list[dict]:
    cases: list[dict] = []

    # ── azure-core (azure-sdk-for-python) ─────────────────────────────────
    # Source: https://learn.microsoft.com/en-us/python/api/azure-core/azure.core.exceptions
    cases.append(
        _entry(
            "external_fault",
            "HttpResponseError",
            "Operation returned an invalid status 'Too Many Requests'. "
            "Error: Code: TooManyRequests Message: Rate limit is exceeded.",
            "azure-core-docs",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "HttpResponseError",
            "Operation returned an invalid status 'Service Unavailable'. "
            "Error: Code: ServiceUnavailable Message: The service is temporarily unavailable.",
            "azure-core-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "ResourceNotFoundError",
            "The Resource 'Microsoft.CognitiveServices/accounts/my-account/deployments"
            "/gpt-4o' under resource group 'my-rg' was not found.",
            "azure-core-docs",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "HttpResponseError",
            "Operation returned an invalid status 'Bad Request'. "
            "Error: Code: InvalidRequestBody Message: Request body is not valid JSON.",
            "azure-core-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "ServiceRequestError",
            "An error occurred while sending the request. "
            "Connection to oai.azure.com timed out. (connect timeout=30)",
            "azure-core-docs",
        )
    )

    # ── Mistral AI SDK (mistralai) ─────────────────────────────────────────
    # Source: https://github.com/mistralai/client-python/blob/main/src/mistralai/models/sdkerror.py
    # and published error documentation
    cases.append(
        _entry(
            "external_fault",
            "SDKError",
            "Status 429: Too many requests. Please retry after 60 seconds.",
            "mistralai-sdk",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "SDKError",
            "Status 500: Internal server error. Please retry your request.",
            "mistralai-sdk",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "SDKError",
            "Status 422: Unprocessable Entity: messages: value is not a valid list",
            "mistralai-sdk",
        )
    )

    # ── Cohere SDK ─────────────────────────────────────────────────────────
    # Source: https://docs.cohere.com/reference/errors
    cases.append(
        _entry(
            "external_fault",
            "TooManyRequestsError",
            "You are using a Trial key, which is limited to 5 API calls / minute.",
            "cohere-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "BadRequestError",
            "invalid request: tool with name 'web_scrape' was not found in the "
            "provided tool definitions",
            "cohere-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "GatewayTimeoutError",
            "The upstream service did not respond within the 60-second timeout window.",
            "cohere-docs",
        )
    )

    # ── Groq SDK ──────────────────────────────────────────────────────────
    # Source: https://github.com/groq/groq-python — mirrors openai-python error hierarchy
    cases.append(
        _entry(
            "external_fault",
            "RateLimitError",
            "Rate limit reached for model llama3-8b-8192 in organization "
            "org-abc123 on tokens per minute (TPM): Limit 30000, Used 29842, "
            "Requested 500.",
            "groq-sdk",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "BadRequestError",
            "Error code: 400 - {'error': {'message': 'json_validate_failed: "
            "JSON schema validation failed', 'type': 'invalid_request_error'}}",
            "groq-sdk",
        )
    )

    # ── LiteLLM exception wrappers ─────────────────────────────────────────
    # Source: https://docs.litellm.ai/docs/exception_mapping
    cases.append(
        _entry(
            "external_fault",
            "RateLimitError",
            "litellm.RateLimitError: AnthropicException - "
            "{'type': 'error', 'error': {'type': 'rate_limit_error', "
            "'message': 'Number of request tokens has exceeded your daily rate limit'}}",
            "litellm-docs",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "BadRequestError",
            "litellm.BadRequestError: OpenAIException - 'messages[0].content' "
            "is invalid. Expected a string but got an object.",
            "litellm-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "Timeout",
            "litellm.Timeout: Request timed out after 600.0 seconds. "
            "Provider: anthropic, Model: claude-3-opus-20240229",
            "litellm-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "NotFoundError",
            "litellm.NotFoundError: OpenAIException - "
            "The model 'gpt-4-vision' does not exist or you do not have access to it.",
            "litellm-docs",
        )
    )

    # ── google-cloud-aiplatform (Vertex AI) ───────────────────────────────
    # Source: https://cloud.google.com/python/docs/reference/aiplatform/latest
    # Different from google-genai SDK — uses google.api_core.exceptions directly.
    cases.append(
        _entry(
            "external_fault",
            "ResourceExhausted",
            "429 Quota exceeded for quota metric 'generate_content_request_count' "
            "and limit 'GenerateContentRequestsPerMinutePerProjectPerRegion' of "
            "service 'aiplatform.googleapis.com'.",
            "vertex-aiplatform-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "DeadlineExceeded",
            "504 Deadline of 60.0s exceeded while calling "
            "aiplatform.googleapis.com:443",
            "vertex-aiplatform-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "NotFound",
            "404 Endpoint projects/123/locations/us-central1/endpoints/456 "
            "is not found.",
            "vertex-aiplatform-docs",
        )
    )

    # ── LlamaIndex (llama-index-core) ─────────────────────────────────────
    # Source: https://github.com/run-llama/llama_index/tree/main/llama-index-core
    cases.append(
        _entry(
            "schema_mismatch",
            "OutputParserError",
            "Got invalid output: Expected output to be formatted as a JSON "
            "instance that conforms to the JSON schema below.\n"
            "Here is the output schema:\n{...}\nHere is the output: None",
            "llamaindex-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "ToolException",
            "Error: Could not find tool with name `get_current_weather`. "
            "Please use a valid tool.",
            "llamaindex-docs",
        )
    )

    # ── Novel structural phrasings ─────────────────────────────────────────
    # Phrases that express a category without the usual keywords; these test
    # whether RulesClassifier generalises beyond its explicit pattern list.
    cases.append(
        _entry(
            "timeout",
            "asyncio.TimeoutError",
            "",  # empty string — only exception type is informative
            "novel-empty-message",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "requests.exceptions.HTTPError",
            "503 Server Error: Service Temporarily Unavailable for url: "
            "https://api.anthropic.com/v1/messages",
            "novel-requests-phrasing",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "KeyError",
            "'summarize_pdf' is not a registered agent tool",
            "novel-agent-registry",
        )
    )
    cases.append(
        _entry(
            "unknown",
            "PermissionError",
            "Permission denied: caller does not have required IAM role "
            "'roles/aiplatform.user'",
            "novel-iam",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "ValueError",
            "Model output does not match expected Pydantic schema: "
            "1 validation error for ExtractedData\nentities\n  "
            "field required (type=value_error.missing)",
            "novel-pydantic-agent",
        )
    )

    return [c for c in cases if c is not None]


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_c.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    captured = sum(1 for c in corpus if c.get("provenance") == "captured")
    transcribed = sum(
        1 for c in corpus if str(c.get("provenance", "")).startswith("transcribed")
    )
    print(f"  captured: {captured}, transcribed: {transcribed}")
    labels = (
        "schema_mismatch",
        "external_fault",
        "timeout",
        "wrong_tool_called",
        "unknown",
    )
    for label_val in labels:
        n = sum(1 for c in corpus if c["label"] == label_val)
        if n:
            print(f"  {label_val}: {n}")
