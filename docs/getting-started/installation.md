# Installation

## Requirements

- Python 3.10 or higher
- Core dependencies: `anyio>=4.0`, `pydantic>=2.0`

## Core install

```bash
pip install triage-agent
```

The core package includes `RulesClassifier`, `InMemoryCheckpointStore`, all built-in strategies, and the `Agent` class. No API keys required.

## Optional extras

Install only what you need:

```bash
# LLM-based classifier (Anthropic / Claude)
pip install "triage-agent[anthropic]"

# LLM-based classifier (OpenAI-compatible: Ollama, Groq, HuggingFace, etc.)
pip install openai  # just the openai package — no extra needed

# Durable checkpoint storage
pip install "triage-agent[sqlite]"   # SQLiteCheckpointStore
pip install "triage-agent[redis]"    # RedisCheckpointStore

# Framework adapters
pip install "triage-agent[langgraph]"      # LangGraph
pip install "triage-agent[crewai]"         # CrewAI
pip install "triage-agent[openai-agents]"  # OpenAI Agents SDK
pip install "triage-agent[langchain]"      # LangChain
```

## BYOK — provider env vars

`LLMClassifier` reads these environment variables so you can switch providers without changing code:

| Variable | Purpose |
|---|---|
| `TRIAGE_LLM_BASE_URL` | Base URL for any OpenAI-compatible API (Ollama, Groq, HuggingFace, etc.) |
| `TRIAGE_LLM_MODEL` | Model name override |
| `TRIAGE_LLM_API_KEY` | API key (if not using the provider's own env var) |

```bash
# Anthropic (uses ANTHROPIC_API_KEY automatically)
LLMClassifier()

# Ollama — local, no key needed
TRIAGE_LLM_BASE_URL=http://localhost:11434/v1 TRIAGE_LLM_MODEL=llama3.2 python agent.py

# Groq
TRIAGE_LLM_BASE_URL=https://api.groq.com/openai/v1 \
TRIAGE_LLM_API_KEY=gsk_... \
TRIAGE_LLM_MODEL=llama-3.1-8b-instant python agent.py
```
