"""
scripts/gen_error_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_a.json by provoking real exceptions
from installed SDKs and capturing (exception_type, error_string, label).

The JSON file is checked in so scoring is reproducible without re-running
this script. Re-run it when adding new cases or installing new SDKs.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus.py
"""

from __future__ import annotations

import json
import socket
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
        }


def _entry(label: str, exc_type: str, error: str) -> dict:
    return {"exception_type": exc_type, "error": error, "label": label}


def build_corpus() -> list[dict]:
    cases: list[dict] = []

    # ── json.JSONDecodeError ───────────────────────────────────────────────
    import json as _json

    cases.append(_capture("schema_mismatch", lambda: _json.loads("{bad}")))
    cases.append(_capture("schema_mismatch", lambda: _json.loads('{"a":1,}')))
    cases.append(_capture("schema_mismatch", lambda: _json.loads("")))
    cases.append(_capture("schema_mismatch", lambda: _json.loads('{"a": [1, 2, 3}')))
    cases.append(_capture("schema_mismatch", lambda: _json.loads("null\nextra")))

    # ── pydantic ValidationError ───────────────────────────────────────────
    try:
        from pydantic import BaseModel

        class _M(BaseModel):
            name: str
            age: int
            score: float

        cases.append(_capture("schema_mismatch", lambda: _M(age=3)))
        cases.append(_capture("schema_mismatch", lambda: _M(name="x", age="abc")))
        cases.append(_capture("schema_mismatch", lambda: _M(name="x", age=1, score="nope")))
    except ImportError:
        print("  SKIP pydantic: not installed", file=sys.stderr)

    # ── stdlib timeout ─────────────────────────────────────────────────────
    import asyncio

    cases.append(
        _capture("timeout", lambda: asyncio.run(asyncio.wait_for(asyncio.sleep(1), 0.001)))
    )
    cases.append(_entry("timeout", "TimeoutError", "timed out"))
    cases.append(_entry("timeout", "TimeoutError", str(socket.timeout("timed out"))))  # noqa: UP041

    # ── httpx errors ──────────────────────────────────────────────────────
    try:
        import httpx

        _req = httpx.Request("GET", "http://x/y")
        cases.append(_capture("unknown", lambda: httpx.get("http://127.0.0.1:9/x", timeout=0.5)))
        cases.append(_entry("timeout", "ReadTimeout", str(httpx.ReadTimeout("timed out"))))
        cases.append(_entry("timeout", "ConnectTimeout", str(httpx.ConnectTimeout("timed out"))))
        for code in (429, 500, 502, 503):

            def _raise(c=code):
                r = httpx.Response(c, request=_req)
                r.raise_for_status()

            cases.append(_capture("external_fault", _raise))
    except ImportError:
        print("  SKIP httpx: not installed", file=sys.stderr)

    # ── openai SDK errors ─────────────────────────────────────────────────
    try:
        import httpx
        import openai

        _oreq = httpx.Request("POST", "https://api.openai.com/v1/messages")
        cases.append(
            _entry(
                "external_fault",
                "RateLimitError",
                str(
                    openai.RateLimitError(
                        "Rate limit reached for gpt-4o",
                        response=httpx.Response(429, request=_oreq),
                        body=None,
                    )
                ),
            )
        )
        cases.append(
            _entry(
                "timeout",
                "APITimeoutError",
                str(openai.APITimeoutError(request=_oreq)),
            )
        )
        cases.append(
            _entry(
                "unknown",
                "APIConnectionError",
                str(openai.APIConnectionError(request=_oreq)),
            )
        )
        cases.append(
            _entry(
                "external_fault",
                "InternalServerError",
                str(
                    openai.InternalServerError(
                        "The server had an error processing your request",
                        response=httpx.Response(500, request=_oreq),
                        body=None,
                    )
                ),
            )
        )
        cases.append(
            _entry(
                "wrong_tool_called",
                "BadRequestError",
                str(
                    openai.BadRequestError(
                        "Invalid value for 'tool_choice': no function named 'lookup_user' in tools",
                        response=httpx.Response(400, request=_oreq),
                        body=None,
                    )
                ),
            )
        )
    except ImportError:
        print("  SKIP openai: not installed", file=sys.stderr)

    # ── anthropic SDK errors ──────────────────────────────────────────────
    try:
        import anthropic
        import httpx

        _areq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        cases.append(
            _entry(
                "external_fault",
                "RateLimitError",
                str(
                    anthropic.RateLimitError(
                        "rate_limit_error",
                        response=httpx.Response(429, request=_areq),
                        body=None,
                    )
                ),
            )
        )
        cases.append(
            _entry(
                "timeout",
                "APITimeoutError",
                str(anthropic.APITimeoutError(request=_areq)),
            )
        )
        cases.append(
            _entry(
                "external_fault",
                "InternalServerError",
                str(
                    anthropic.InternalServerError(
                        "Internal server error",
                        response=httpx.Response(500, request=_areq),
                        body=None,
                    )
                ),
            )
        )
        cases.append(
            _entry(
                "wrong_tool_called",
                "BadRequestError",
                str(
                    anthropic.BadRequestError(
                        "Tool 'send_sms' does not exist in tools list",
                        response=httpx.Response(400, request=_areq),
                        body=None,
                    )
                ),
            )
        )
    except ImportError:
        print("  SKIP anthropic: not installed", file=sys.stderr)

    # ── langchain / langgraph errors ──────────────────────────────────────
    try:
        from langchain_core.exceptions import OutputParserException
        from langchain_core.tools.base import ToolException

        cases.append(
            _entry(
                "schema_mismatch",
                "OutputParserException",
                str(
                    OutputParserException(
                        "Invalid JSON output: Expected the output value to start with '```json'"
                    )
                ),
            )
        )
        cases.append(
            _entry(
                "wrong_tool_called",
                "ToolException",
                str(ToolException("Tool 'web_search' not found in the provided tools list")),
            )
        )
    except ImportError as e:
        print(f"  SKIP langchain_core: {e}", file=sys.stderr)

    try:
        from langgraph.errors import GraphInterrupt

        cases.append(
            _entry(
                "unknown",
                "GraphInterrupt",
                str(GraphInterrupt("Interrupted at node: agent")),
            )
        )
    except (ImportError, Exception) as e:
        print(f"  SKIP langgraph.errors: {e}", file=sys.stderr)

    return [c for c in cases if c is not None]


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_a.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    labels = ("schema_mismatch", "external_fault", "timeout", "wrong_tool_called", "unknown")
    for label_val in labels:
        n = sum(1 for c in corpus if c["label"] == label_val)
        print(f"  {label_val}: {n}")
