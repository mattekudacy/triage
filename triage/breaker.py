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

Persistent state (multi-worker / serverless)
--------------------------------------------
Pass a ``store=`` to share OPEN/HALF_OPEN state across processes::

    import redis
    from triage.breaker_store import RedisBreakerStore

    r = redis.Redis.from_url("redis://localhost:6379")
    store = RedisBreakerStore(r, key_prefix="myapp:breaker")
    breaker = CircuitBreaker(failure_threshold=5, window_seconds=60,
                             cooldown_seconds=30, store=store)

When a store is attached all state reads/writes go through it.  Note that
``time.monotonic()`` is process-local; the store path switches to
``time.time()`` (wall-clock UTC seconds) automatically.

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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass


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
    store:
        Optional persistence backend. When supplied all state reads/writes go
        through the store so the breaker is shared across processes. Omit for
        the default in-process-only behaviour.
    """

    failure_threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    store: Any = field(default=None, repr=False)  # BreakerStore | None

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # In-memory state — used only when store is None
    _failure_times: list[float] = field(default_factory=list, init=False, repr=False)
    _state: BreakerState = field(default=BreakerState.CLOSED, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)
    _probe_in_flight: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")

    # ── clock helper ─────────────────────────────────────────────────────────

    def _now_default(self) -> float:
        """Return the appropriate 'now' based on whether a store is attached.

        monotonic() is fine in-process; wall clock is required when comparing
        timestamps across processes via a shared store.
        """
        return time.time() if self.store is not None else time.monotonic()

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
            if self.store is not None:
                snap = self.store.load()
                return len(snap.failure_times)
            return len(self._failure_times)

    # ── public write ─────────────────────────────────────────────────────────

    def record_failure(self, *, _now: float | None = None) -> BreakerState:
        """Record a failure. Returns the state after this failure.

        ``_now`` is injectable for testing without real-clock dependency.
        """
        now = _now if _now is not None else self._now_default()
        with self._lock:
            if self.store is not None:
                return self._record_failure_store(now)
            return self._record_failure_mem(now)

    def record_success(self, *, _now: float | None = None) -> BreakerState:
        """Record a success. Closes the breaker when called in HALF_OPEN state.

        ``_now`` is injectable for testing.
        """
        with self._lock:
            if self.store is not None:
                return self._record_success_store()
            return self._record_success_mem()

    def is_open(self, *, _now: float | None = None) -> bool:
        """Return True when the breaker is OPEN (calls should be blocked).

        Returns False for CLOSED and HALF_OPEN (probe allowed in HALF_OPEN).
        """
        now = _now if _now is not None else self._now_default()
        with self._lock:
            return self._evaluate_state(now) == BreakerState.OPEN

    def allow_request(self, *, _now: float | None = None) -> bool:
        """Return True when a call should be allowed through.

        CLOSED    → True (normal operation)
        OPEN      → False (blocked)
        HALF_OPEN → True for exactly one concurrent caller; subsequent callers get
                    False until the in-flight probe records its outcome. This enforces
                    a single-probe policy when the breaker is shared across agents.
        """
        now = _now if _now is not None else self._now_default()
        with self._lock:
            state = self._evaluate_state(now)
            if state == BreakerState.OPEN:
                return False
            if state == BreakerState.HALF_OPEN:
                if self.store is not None:
                    if self.store.get_probe_in_flight():
                        return False
                    self.store.set_probe_in_flight(True)
                else:
                    if self._probe_in_flight:
                        return False
                    self._probe_in_flight = True
            return True

    def reset(self) -> None:
        """Fully reset the breaker to CLOSED with no recorded failures."""
        with self._lock:
            if self.store is not None:
                from triage.breaker_store import BreakerSnapshot

                self.store.evict_before(float("inf"))
                self.store.save(BreakerSnapshot())
            else:
                self._state = BreakerState.CLOSED
                self._failure_times.clear()
                self._opened_at = None
                self._probe_in_flight = False

    # ── in-memory internals ───────────────────────────────────────────────────

    def _evict_old_failures(self, now: float | None = None) -> None:
        """Remove failures outside the current window. Must be called under lock."""
        if self.store is not None:
            t = now if now is not None else self._now_default()
            cutoff = t - self.window_seconds
            self.store.evict_before(cutoff)
            return
        t = now if now is not None else time.monotonic()
        cutoff = t - self.window_seconds
        self._failure_times = [f for f in self._failure_times if f > cutoff]

    def _evaluate_state(self, now: float | None = None) -> BreakerState:
        """Re-evaluate OPEN → HALF_OPEN transition. Must be called under lock."""
        if self.store is not None:
            snap = self.store.load()
            state = snap.state
            opened_at = snap.opened_at
            if state == BreakerState.OPEN and opened_at is not None:
                t = now if now is not None else self._now_default()
                if (t - opened_at) >= self.cooldown_seconds:
                    state = BreakerState.HALF_OPEN
                    snap.state = state
                    self.store.save(snap)
            return cast(BreakerState, state)
        # in-memory path
        if self._state == BreakerState.OPEN and self._opened_at is not None:
            t = now if now is not None else time.monotonic()
            if (t - self._opened_at) >= self.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
        return self._state

    def _record_failure_mem(self, now: float) -> BreakerState:
        """In-memory record_failure. Must be called under lock."""
        self._evict_old_failures(now)
        self._failure_times.append(now)
        self._probe_in_flight = False

        if self._state == BreakerState.HALF_OPEN:
            self._state = BreakerState.OPEN
            self._opened_at = now
        elif self._state == BreakerState.CLOSED:
            if len(self._failure_times) >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = now
        return self._state

    def _record_failure_store(self, now: float) -> BreakerState:
        """Store-backed record_failure. Must be called under lock."""
        cutoff = now - self.window_seconds
        self.store.evict_before(cutoff)
        failure_times = self.store.add_failure(now)
        self.store.set_probe_in_flight(False)

        snap = self.store.load()
        state = snap.state

        if state == BreakerState.HALF_OPEN:
            state = BreakerState.OPEN
            snap.state = state
            snap.opened_at = now
            snap.probe_in_flight = False
            self.store.save(snap)
        elif state == BreakerState.CLOSED:
            if len(failure_times) >= self.failure_threshold:
                state = BreakerState.OPEN
                snap.state = state
                snap.opened_at = now
                snap.probe_in_flight = False
                self.store.save(snap)
        return cast(BreakerState, state)

    def _record_success_mem(self) -> BreakerState:
        """In-memory record_success. Must be called under lock."""
        self._probe_in_flight = False
        if self._state == BreakerState.HALF_OPEN:
            self._state = BreakerState.CLOSED
            self._failure_times.clear()
            self._opened_at = None
        return self._state

    def _record_success_store(self) -> BreakerState:
        """Store-backed record_success. Must be called under lock."""
        snap = self.store.load()
        snap.probe_in_flight = False
        if snap.state == BreakerState.HALF_OPEN:
            snap.state = BreakerState.CLOSED
            snap.failure_times = []
            snap.opened_at = None
            self.store.evict_before(float("inf"))
        self.store.save(snap)
        return cast(BreakerState, snap.state)
