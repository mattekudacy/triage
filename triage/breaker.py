"""
triage.breaker
~~~~~~~~~~~~~~
Circuit breaker for cross-run failure-rate tracking.

Unlike per-run caps (``max_recovery_attempts``, ``max_recovery_seconds``),
a ``CircuitBreaker`` is **shared across runs** — it trips open after too many
failures accumulate in a sliding time window, then stays open during a cooldown
period before allowing probe attempts again.

Usage::

    from triage.breaker import CircuitBreaker
    from triage.strategies.circuit_breaker import circuit_breaker
    from triage.strategies.retry import backoff_and_retry

    breaker = CircuitBreaker(failure_threshold=5, window_seconds=60, cooldown_seconds=30)

    policy = FailurePolicy(
        EXTERNAL_FAULT=circuit_breaker(breaker, backoff_and_retry(max_attempts=3)),
    )

States
------
CLOSED  — normal operation; failures are counted within the window.
OPEN    — circuit is tripped; all attempts are immediately blocked (escalated).
HALF_OPEN — cooldown elapsed; one probe attempt is allowed through to test recovery.

Transitions
-----------
CLOSED  → OPEN       when failure count >= failure_threshold within window_seconds
OPEN    → HALF_OPEN  when cooldown_seconds have elapsed since the breaker opened
HALF_OPEN → CLOSED   when the probe attempt succeeds (``record_success()`` called)
HALF_OPEN → OPEN     when the probe attempt fails (``record_failure()`` called)

Thread safety
-------------
All state mutations are guarded with a ``threading.Lock``. This is intentional:
``classify()`` may run inside ``anyio.to_thread.run_sync()``, and the breaker
must be visible across concurrent async tasks and threads alike. A ``ContextVar``
would *not* work here — a ContextVar mutated inside a thread dispatch is invisible
to the caller once the call returns.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Shared, thread-safe circuit breaker.

    Parameters
    ----------
    failure_threshold:
        Number of failures within ``window_seconds`` required to trip the breaker.
    window_seconds:
        Sliding window duration in seconds. Failures older than this are ignored.
    cooldown_seconds:
        How long to stay OPEN before allowing a single probe attempt (HALF_OPEN).
    """

    failure_threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 30.0

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _failure_times: list[float] = field(default_factory=list, init=False, repr=False)
    _state: BreakerState = field(default=BreakerState.CLOSED, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")

    # ── public read ──────────────────────────────────────────────────────────

    def state(self, *, _now: float | None = None) -> BreakerState:
        """Current breaker state (re-evaluates OPEN → HALF_OPEN transition).

        ``_now`` is injectable for testing without real-clock dependency.
        """
        with self._lock:
            return self._evaluate_state(_now)

    def failure_count(self, *, _now: float | None = None) -> int:
        """Number of failures within the current window.

        ``_now`` is injectable for testing without real-clock dependency.
        """
        with self._lock:
            self._evict_old_failures(_now)
            return len(self._failure_times)

    # ── public write ─────────────────────────────────────────────────────────

    def record_failure(self, *, _now: float | None = None) -> BreakerState:
        """Record a failure. Returns the state after this failure.

        ``_now`` is injectable for testing without real-clock dependency.
        """
        now = _now if _now is not None else time.monotonic()
        with self._lock:
            self._evict_old_failures(now)
            self._failure_times.append(now)

            if self._state == BreakerState.HALF_OPEN:
                # Probe failed — re-open immediately
                self._state = BreakerState.OPEN
                self._opened_at = now
            elif self._state == BreakerState.CLOSED:
                if len(self._failure_times) >= self.failure_threshold:
                    self._state = BreakerState.OPEN
                    self._opened_at = now

            return self._state

    def record_success(self, *, _now: float | None = None) -> BreakerState:
        """Record a success. Closes the breaker when called in HALF_OPEN state.

        ``_now`` is injectable for testing.
        """
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.CLOSED
                self._failure_times.clear()
                self._opened_at = None
            return self._state

    def is_open(self, *, _now: float | None = None) -> bool:
        """Return True when the breaker is OPEN (calls should be blocked).

        Returns False for CLOSED and HALF_OPEN (probe allowed in HALF_OPEN).
        """
        now = _now if _now is not None else time.monotonic()
        with self._lock:
            return self._evaluate_state(now) == BreakerState.OPEN

    def allow_request(self, *, _now: float | None = None) -> bool:
        """Return True when a call should be allowed through.

        CLOSED  → True (normal operation)
        OPEN    → False (blocked)
        HALF_OPEN → True (one probe allowed; caller must call record_success/failure)
        """
        now = _now if _now is not None else time.monotonic()
        with self._lock:
            return self._evaluate_state(now) != BreakerState.OPEN

    def reset(self) -> None:
        """Fully reset the breaker to CLOSED with no recorded failures."""
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failure_times.clear()
            self._opened_at = None

    # ── internals ────────────────────────────────────────────────────────────

    def _evict_old_failures(self, now: float | None = None) -> None:
        """Remove failures outside the current window. Must be called under lock."""
        t = now if now is not None else time.monotonic()
        cutoff = t - self.window_seconds
        self._failure_times = [f for f in self._failure_times if f > cutoff]

    def _evaluate_state(self, now: float | None = None) -> BreakerState:
        """Re-evaluate OPEN → HALF_OPEN transition. Must be called under lock."""
        if self._state == BreakerState.OPEN and self._opened_at is not None:
            t = now if now is not None else time.monotonic()
            if (t - self._opened_at) >= self.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
        return self._state
