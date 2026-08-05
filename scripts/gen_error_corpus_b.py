"""
scripts/gen_error_corpus_b.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_b.json — corpus B.

Corpus B was held out when it was built: rules.py was NOT consulted while
assembling it. Sources are deliberately disjoint from corpus A —
boto3/botocore, google-genai/grpc, aiohttp, requests/urllib3, stdlib urllib,
plus phrasings that differ structurally from the strings used to write or fix
the v0.25 patterns.

**Corpus B is now training data.** Its misses guided the v0.26 botocore and
schema-exception fixes, so its 90% score no longer measures generalization.
Corpus C is the current held-out benchmark — see scripts/README.md.

All entries are labeled with the expected FailureType value.

Provenance per entry:
  _capture(label, fn)   — exception actually raised and captured
  _entry(label, exc_type, error, note)  — transcribed from published error
                                          documentation; note records the source

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus_b.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
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

    # ── stdlib urllib (provoked) ──────────────────────────────────────────
    cases.append(
        _capture("unknown", lambda: urllib.request.urlopen("http://127.0.0.1:9/", timeout=0.5))
    )

    # ── botocore / boto3 error strings (transcribed from AWS docs) ────────
    # Source: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
    # and botocore exception hierarchy
    cases.append(
        _entry(
            "external_fault",
            "ThrottlingException",
            "An error occurred (ThrottlingException) when calling the InvokeModel operation: "
            "Rate exceeded",
            "aws-docs",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "ServiceUnavailableException",
            "An error occurred (ServiceUnavailableException) when calling the InvokeModel "
            "operation: Service is unavailable. Please try again.",
            "aws-docs",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "ValidationException",
            "An error occurred (ValidationException) when calling the InvokeModel operation: "
            "The model returned a response that did not conform to the expected schema.",
            "aws-docs",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "ResourceNotFoundException",
            "An error occurred (ResourceNotFoundException) when calling the InvokeAgent "
            "operation: No resource with ID 'my-agent' was found.",
            "aws-docs",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "ClientError",
            "An error occurred (TooManyRequestsException) when calling the InvokeEndpoint "
            "operation: Too Many Requests",
            "aws-docs",
        )
    )

    # ── google-genai / gRPC status codes (transcribed from google-genai docs) ─
    # Source: https://ai.google.dev/gemini-api/docs/troubleshooting
    cases.append(
        _entry(
            "external_fault",
            "ResourceExhausted",
            "429 RESOURCE_EXHAUSTED: You have exceeded your current quota, please refer to "
            "https://ai.google.dev/gemini-api/docs/rate-limits",
            "google-genai-docs",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "ServiceUnavailable",
            "503 UNAVAILABLE: The service is currently unavailable. This is most likely a "
            "transient condition.",
            "google-genai-docs",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "InvalidArgument",
            "400 INVALID_ARGUMENT: * GenerateContentRequest.contents: contents is not "
            "specified\n* GenerateContentRequest.model: model is not specified",
            "google-genai-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "DeadlineExceeded",
            "504 DEADLINE_EXCEEDED: Deadline Exceeded",
            "google-genai-docs",
        )
    )

    # ── aiohttp (transcribed from aiohttp docs + source) ──────────────────
    # Source: https://docs.aiohttp.org/en/stable/client_reference.html#exceptions
    cases.append(
        _entry(
            "unknown",
            "ClientConnectorError",
            "Cannot connect to host api.openai.com:443 ssl:default [Connection refused]",
            "aiohttp-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "ServerTimeoutError",
            "Connection timeout",
            "aiohttp-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "ServerConnectionError",
            "Server disconnected after 30.0 seconds of inactivity",
            "aiohttp-docs",
        )
    )

    # ── requests / urllib3 (transcribed from requests docs) ───────────────
    # Source: https://requests.readthedocs.io/en/latest/api/#exceptions
    cases.append(
        _entry(
            "unknown",
            "ConnectionError",
            "HTTPSConnectionPool(host='api.openai.com', port=443): Max retries exceeded "
            "with url: /v1/chat/completions (Caused by NewConnectionError("
            "'<urllib3.connection.HTTPSConnection object>: Failed to establish a new "
            "connection: [Errno 8] nodename nor servname provided, or not known'))",
            "requests-docs",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "ReadTimeout",
            "HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out. "
            "(read timeout=30)",
            "requests-docs",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "HTTPError",
            "429 Client Error: Too Many Requests for url: https://api.openai.com/v1/chat",
            "requests-docs",
        )
    )

    # ── structurally different phrasings for already-covered types ────────
    # These test whether the patterns generalize beyond the exact wording in corpus A.
    cases.append(
        _entry(
            "schema_mismatch",
            "JSONDecodeError",
            "Unterminated string starting at: line 3 column 5 (char 42)",
            "python-json",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "JSONDecodeError",
            "Object key followed by : not found",
            "python-json-alt",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "RateLimitError",
            "openai.RateLimitError: 429 Too many requests. Please back off.",
            "openai-alt-phrasing",
        )
    )
    cases.append(
        _entry(
            "wrong_tool_called",
            "ValueError",
            "Tool lookup_weather is not registered. Available tools: search, calculator",
            "generic-agent",
        )
    )

    return [c for c in cases if c is not None]


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_b.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    captured = sum(1 for c in corpus if c.get("provenance") == "captured")
    transcribed = sum(1 for c in corpus if str(c.get("provenance", "")).startswith("transcribed"))
    print(f"  captured: {captured}, transcribed: {transcribed}")
    labels = ("schema_mismatch", "external_fault", "timeout", "wrong_tool_called", "unknown")
    for label_val in labels:
        n = sum(1 for c in corpus if c["label"] == label_val)
        print(f"  {label_val}: {n}")
