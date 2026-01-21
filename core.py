"""Core types for ActionGate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Mode(Enum):
    """Enforcement mode for blocked actions."""
    HARD = auto()  # Raise exception on block
    SOFT = auto()  # Return blocked decision (caller handles fallback)


class StoreErrorMode(Enum):
    """Behavior when store backend fails."""
    FAIL_CLOSED = auto()  # Block action (safe default)
    FAIL_OPEN = auto()    # Allow action (availability over safety)


class Status(Enum):
    """Decision outcome."""
    ALLOW = auto()
    BLOCK = auto()


class BlockReason(Enum):
    """Why an action was blocked."""
    RATE_LIMIT = auto()   # Exceeded max_calls in window
    COOLDOWN = auto()     # Too soon since last call
    STORE_ERROR = auto()  # Backend failure (behavior depends on policy)


@dataclass(frozen=True, slots=True)
class Gate:
    """Identifies a rate-limited action stream.
    
    Examples:
        Gate("billing", "refund", "user:123")     # per-user
        Gate("support", "escalate", "agent:42")   # per-agent  
        Gate("api", "search", "global")           # global limit
        Gate("chat", "send", "session:abc")       # per-session
    """
    namespace: str
    action: str
    principal: str = "global"

    def __str__(self) -> str:
        return f"{self.namespace}:{self.action}@{self.principal}"
    
    @property
    def key(self) -> str:
        """Redis-friendly key string."""
        return f"ag:{self.namespace}:{self.action}:{self.principal}"


@dataclass(frozen=True, slots=True)
class Policy:
    """Rate limiting policy.
    
    Args:
        max_calls: Maximum calls allowed in window (0 = always block)
        window: Rolling window in seconds (None = infinite)
        cooldown: Minimum seconds between calls (0 = no cooldown)
        mode: HARD raises on block, SOFT returns decision
        on_store_error: FAIL_CLOSED blocks, FAIL_OPEN allows
    """
    max_calls: int = 3
    window: float | None = 60.0
    cooldown: float = 0.0
    mode: Mode = Mode.HARD
    on_store_error: StoreErrorMode = StoreErrorMode.FAIL_CLOSED

    def __post_init__(self) -> None:
        if self.max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        if self.window is not None and self.window <= 0:
            raise ValueError("window must be > 0 or None")
        if self.cooldown < 0:
            raise ValueError("cooldown must be >= 0")


@dataclass(frozen=True, slots=True)
class Decision:
    """Result of evaluating an action against its policy."""
    status: Status
    gate: Gate
    policy: Policy
    reason: BlockReason | None = None
    message: str | None = None
    calls_in_window: int = 0
    time_since_last: float | None = None

    @property
    def allowed(self) -> bool:
        return self.status == Status.ALLOW

    @property
    def blocked(self) -> bool:
        return self.status == Status.BLOCK

    def __bool__(self) -> bool:
        """Truthy = allowed."""
        return self.allowed


@dataclass(frozen=True, slots=True)
class Result[T]:
    """Wrapper for guarded function results."""
    decision: Decision
    value: T | None = None

    @property
    def ok(self) -> bool:
        return self.decision.allowed

    def unwrap(self) -> T:
        """Get value or raise if blocked."""
        if self.value is None:
            raise ValueError(f"No value: {self.decision.message or 'blocked'}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default if blocked."""
        return self.value if self.value is not None else default
