"""
triage.classifier.llm
~~~~~~~~~~~~~~~~~~~~~
Semantic failure classifier using Claude (Anthropic) as the LLM backend.

Uses the synchronous Anthropic client so it works inside a running async
event loop without calling asyncio.run(). Called only on failure — not in
the per-step hot path — so the ~100-400ms blocking latency is acceptable.

Install: pip install triage-agent[anthropic]
"""

from __future__ import annotations

from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

try:
    import anthropic as _anthropic
except ImportError as exc:
    raise ImportError(
        "LLMClassifier requires 'anthropic'. "
        "Install it with: pip install triage-agent[anthropic]"
    ) from exc

_FAILURE_TYPE_VALUES = [ft.value for ft in FailureType]

_SYSTEM_PROMPT = (
    "You are a failure classifier for AI agents. "
    "Given a trajectory of steps and a task description, classify the failure "
    "into exactly one of these categories: "
    + ", ".join(_FAILURE_TYPE_VALUES)
    + ". Respond with only the category name (e.g. \"wrong_tool_called\"), nothing else."
)


class LLMClassifier:
    """Semantic failure classifier using Claude as the LLM backend.

    Satisfies the ``Classifier`` protocol (synchronous ``classify`` method).
    Uses ``anthropic.Anthropic`` (sync client) — safe to call from inside a
    running asyncio event loop.

    Falls back to ``FailureType.UNKNOWN`` on any error (network, parse, rate limit).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_trajectory_steps: int = 10,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_trajectory_steps = max_trajectory_steps
        self._client: _anthropic.Anthropic | None = None

    def _get_client(self) -> "_anthropic.Anthropic":
        if self._client is None:
            self._client = _anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _build_prompt(self, trajectory: Trajectory, task: str) -> str:
        steps = trajectory.last_n_steps(self._max_trajectory_steps)
        lines = [f"Task: {task}", "", "Recent steps:"]
        for step in steps:
            lines.append(f"[{step.index}] {step.action}")
            if step.tool_called:
                lines.append(f"  tool: {step.tool_called}")
            if step.error:
                lines.append(f"  error: {step.error}")
            if step.llm_output:
                lines.append(f"  llm_output: {step.llm_output[:200]}")
        lines.append("")
        lines.append("Classify the failure type:")
        return "\n".join(lines)

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        try:
            prompt = self._build_prompt(trajectory, task)
            message = self._get_client().messages.create(
                model=self._model,
                max_tokens=32,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip().lower()
            for ft in FailureType:
                if ft.value == raw:
                    return ft
        except Exception:
            pass
        return FailureType.UNKNOWN
