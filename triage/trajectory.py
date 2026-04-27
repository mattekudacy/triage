"""
triage.trajectory
~~~~~~~~~~~~~~~~~
Ordered, appendable record of agent steps with replay support.
"""

from __future__ import annotations

from triage.taxonomy import Step


class Trajectory:
    """Holds the ordered sequence of Steps for a single agent run."""

    def __init__(self, steps: list[Step] | None = None) -> None:
        self._steps: list[Step] = list(steps) if steps else []

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    def append(self, step: Step) -> None:
        self._steps.append(step)

    def replay_from(self, index: int) -> "Trajectory":
        """Return a new Trajectory pre-populated with steps[index:].

        Does not mutate self. Raises IndexError if index is out of range
        for a non-empty trajectory.
        """
        if self._steps and not (0 <= index <= len(self._steps)):
            raise IndexError(f"index {index} out of range for trajectory of length {len(self._steps)}")
        return Trajectory(steps=self._steps[index:])

    def last_n_steps(self, n: int) -> list[Step]:
        if n <= 0:
            return []
        return self._steps[-n:]

    @classmethod
    def from_steps(cls, steps: list[Step]) -> "Trajectory":
        return cls(steps=steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, index):
        return self._steps[index]

    def __repr__(self) -> str:
        return f"Trajectory({len(self._steps)} steps)"
