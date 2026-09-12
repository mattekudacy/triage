# LangGraph Example

Demo wrapping a LangGraph ReAct agent with `wrap_langgraph()`. Shows a tool call failing with a schema error, triage classifying it as `SCHEMA_MISMATCH`, and a retry with a corrective hint succeeding.

**Source:** [`examples/langgraph_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/langgraph_agent.py)

## Requirements

```bash
pip install "triage-agent[langgraph]" langchain-openai
export OPENAI_API_KEY=sk-...
```

## What it demonstrates

1. A LangGraph agent (`create_react_agent`) is built with one calculator tool, then wrapped with `wrap_langgraph()` — no changes to the graph itself.
2. `wrap_langgraph()` streams per-step events via `astream_events(..., version="v2")`, so triage observes every tool call and LLM turn without the graph calling `record_step()` itself.
3. The tool's first call raises a deliberate validation error; `RulesClassifier` matches it as `SCHEMA_MISMATCH` and `retry_with_tool_manifest()` retries with a corrective hint. The second attempt succeeds.
4. `auto_checkpoint=True` is passed through to `Agent.__init__` — adapters accept the same optional kwargs as `triage.Agent` (`classifier`, `checkpoint_store`, `max_recovery_attempts`, `auto_checkpoint`).

See [Adapters → LangGraph](../adapters/langgraph.md) for how `wrap_langgraph()` itself works.

## Run

```bash
python examples/langgraph_agent.py
```
