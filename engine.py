"""Core engine for ActionGate."""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar, overload

from .store import MemoryStore, Store
from .core import (
    BlockReason,
    Decision,
    Gate,
    Mode,
    Policy,
    Result,
    Status,
    StoreErrorMode,
)

P = ParamSpec("P")
T = TypeVar("T")


class Blocked(RuntimeError):
    """Raised when action is blocked in HARD mode."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.message or f"Blocked: {decision.reason}")
        self.decision = decision


class Engine:
    """ActionGate engine for rate limiting agent actions.
    
    Example:
        engine = Engine()
        
        # Simple decorator (raises on block in HARD mode)
        @engine.guard(Gate("api", "search"), Policy(max_calls=10, window=60))
        def search(query: str) -> list[str]:
            return db.search(query)
        
        # Result-wrapped decorator (never raises, returns Result[T])
        @engine.guard_result(Gate("api", "fetch"), Policy(max_calls=5, mode=Mode.SOFT))
        def fetch(url: str) -> dict:
            return requests.get(url).json()
        
        result = fetch("https://api.example.com")
        if result.ok:
            print(result.value)
        else:
            print(f"Blocked: {result.decision.message}")
    """

    __slots__ = ("_store", "_clock", "_policies", "_listeners", "_errors")

    def __init__(
        self,
        store: Store | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store or MemoryStore()
        self._clock = clock or time.monotonic
        self._policies: dict[Gate, Policy] = {}
        self._listeners: list[Callable[[Decision], None]] = []
        self._errors = 0

    # ─────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────

    def register(self, gate: Gate, policy: Policy) -> None:
        """Register a policy for a gate."""
        self._policies[gate] = policy

    def policy_for(self, gate: Gate) -> Policy:
        """Get policy for gate (default if not registered)."""
        return self._policies.get(gate, Policy())

    def on_decision(self, listener: Callable[[Decision], None]) -> None:
        """Add a listener for decisions (for logging/metrics)."""
        self._listeners.append(listener)

    @property
    def listener_errors(self) -> int:
        """Count of listener exceptions (never block execution)."""
        return self._errors

    # ─────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────

    def check(self, gate: Gate, policy: Policy | None = None) -> Decision:
        """Check if action is allowed and reserve a slot if so."""
        now = self._clock()
        policy = policy or self.policy_for(gate)

        try:
            count, allowed, last_age = self._store.check_and_reserve(gate, now, policy)
        except Exception as e:
            # Handle store error based on policy
            if policy.on_store_error == StoreErrorMode.FAIL_OPEN:
                return self._decide(
                    gate, policy,
                    status=Status.ALLOW,
                    reason=BlockReason.STORE_ERROR,
                    message=f"Store error (fail-open): {e}",
                )
            else:
                return self._decide(
                    gate, policy,
                    status=Status.BLOCK,
                    reason=BlockReason.STORE_ERROR,
                    message=f"Store error (fail-closed): {e}",
                )

        if allowed:
            return self._decide(
                gate, policy,
                status=Status.ALLOW,
                calls_in_window=count,
                time_since_last=last_age,
            )

        # Determine block reason
        if policy.cooldown > 0 and last_age is not None and last_age < policy.cooldown:
            reason, msg = BlockReason.COOLDOWN, f"Cooldown: {last_age:.1f}s < {policy.cooldown}s"
        else:
            reason, msg = BlockReason.RATE_LIMIT, f"Rate limit: {count} >= {policy.max_calls}"

        return self._decide(
            gate, policy,
            status=Status.BLOCK,
            reason=reason,
            message=msg,
            calls_in_window=count,
            time_since_last=last_age,
        )

    def enforce(self, decision: Decision) -> None:
        """Raise Blocked if decision is blocked in HARD mode."""
        if decision.blocked and decision.policy.mode == Mode.HARD:
            raise Blocked(decision)

    def clear(self, gate: Gate) -> None:
        """Clear history for a gate."""
        self._store.clear(gate)

    def clear_all(self) -> None:
        """Clear all history."""
        self._store.clear_all()

    # ─────────────────────────────────────────────────────────────
    # Decorator API
    # ─────────────────────────────────────────────────────────────

    def guard(
        self,
        gate: Gate,
        policy: Policy | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator that returns T directly.
        
        - HARD mode (default): raises Blocked on limit
        - SOFT mode: raises Blocked on limit (use guard_result for no-raise)
        
        Example:
            @engine.guard(Gate("api", "search"), Policy(max_calls=5))
            def search(query: str) -> list[str]:
                return db.search(query)
            
            results = search("hello")  # Returns list[str] or raises Blocked
        """
        if policy is not None:
            self.register(gate, policy)

        def decorator(fn: Callable[P, T]) -> Callable[P, T]:
            @wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                decision = self.check(gate)
                if decision.blocked:
                    raise Blocked(decision)
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def guard_result(
        self,
        gate: Gate,
        policy: Policy | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, Result[T]]]:
        """Decorator that returns Result[T] (never raises).
        
        Use this when you want to handle blocks gracefully without exceptions.
        
        Example:
            @engine.guard_result(Gate("api", "fetch"), Policy(max_calls=5, mode=Mode.SOFT))
            def fetch(url: str) -> dict:
                return requests.get(url).json()
            
            result = fetch("https://api.example.com")
            data = result.unwrap_or({"error": "rate limited"})
        """
        if policy is not None:
            self.register(gate, policy)

        def decorator(fn: Callable[P, T]) -> Callable[P, Result[T]]:
            @wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T]:
                decision = self.check(gate)

                if decision.blocked:
                    return Result(decision=decision, value=None)

                value = fn(*args, **kwargs)
                return Result(decision=decision, value=value)

            return wrapper
        return decorator

    # ─────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────

    def _decide(
        self,
        gate: Gate,
        policy: Policy,
        *,
        status: Status,
        reason: BlockReason | None = None,
        message: str | None = None,
        calls_in_window: int = 0,
        time_since_last: float | None = None,
    ) -> Decision:
        decision = Decision(
            status=status,
            gate=gate,
            policy=policy,
            reason=reason,
            message=message,
            calls_in_window=calls_in_window,
            time_since_last=time_since_last,
        )
        self._emit(decision)
        return decision

    def _emit(self, decision: Decision) -> None:
        for listener in self._listeners:
            try:
                listener(decision)
            except Exception:
                self._errors += 1
