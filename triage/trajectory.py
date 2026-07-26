"""
triage.trajectory
~~~~~~~~~~~~~~~~~
Ordered, appendable record of agent steps with replay support.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from triage.taxonomy import Step

logger = logging.getLogger("triage")


class Trajectory:
    """Holds the ordered sequence of Steps for a single agent run."""

    def __init__(self, steps: list[Step] | None = None) -> None:
        self._steps: list[Step] = list(steps) if steps else []

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    def append(self, step: Step) -> None:
        # Step.index is caller-supplied, not auto-assigned — a non-monotonic
        # index usually means the caller mismanaged its own counter (e.g.
        # reused an index across retries). Warn rather than raise: index is
        # informational (used in log lines and LLM prompts, not by any
        # internal invariant), and existing agents that don't track it
        # carefully should not suddenly break on upgrade.
        #
        # Strict less-than (<) rather than less-than-or-equal (<=) is
        # intentional: equal indices are allowed because Step.partial=True
        # steps legitimately share an index with the completed step that
        # follows them (the partial and its successor represent the same
        # logical step, just at different stages of completion).
        if self._steps and step.index < self._steps[-1].index:
            logger.warning(
                "[triage] Trajectory.append() received non-monotonic index "
                "(step.index=%r, previous=%r) — Step.index should increase "
                "with each recorded step",
                step.index,
                self._steps[-1].index,
                extra={"triage_event": "non_monotonic_step_index"},
            )
        self._steps.append(step)

    def replay_from(self, index: int) -> Trajectory:
        """Return a new Trajectory pre-populated with steps[index:].

        Does not mutate self. Raises IndexError if index is out of range
        for a non-empty trajectory.
        """
        if self._steps and not (0 <= index <= len(self._steps)):
            raise IndexError(
                f"index {index} out of range for trajectory of length {len(self._steps)}"
            )
        return Trajectory(steps=self._steps[index:])

    def last_n_steps(self, n: int) -> list[Step]:
        if n <= 0:
            return []
        return self._steps[-n:]

    @classmethod
    def from_steps(cls, steps: list[Step]) -> Trajectory:
        return cls(steps=steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def __getitem__(self, index: int) -> Step:
        return self._steps[index]

    def __repr__(self) -> str:
        return f"Trajectory({len(self._steps)} steps)"
