"""ActionGate: Deterministic, pre-execution gating for semantic actions (e.g. tool calls) in agent systems.

Reference implementation with policy enforcement, cooldowns, and distributed support.

Example:
    from actiongate import Engine, Gate, Policy, Mode

    engine = Engine()

    @engine.guard(Gate("api", "search", "user:123"), Policy(max_calls=5, window=60))
    def search(query: str) -> list[str]:
        return api.search(query)

    try:
        results = search("hello")  # Returns list[str]
    except Blocked as e:
        print(f"Rate limited: {e.decision.message}")

    # Or use guard_result for no-exception handling:
    @engine.guard_result(Gate("api", "fetch"), Policy(max_calls=5, mode=Mode.SOFT))
    def fetch(url: str) -> dict:
        return requests.get(url).json()

    result = fetch("https://api.example.com")
    data = result.unwrap_or({"error": "rate limited"})
"""

from .engine import Blocked, Engine
from .store import MemoryStore, RedisStore, Store
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

__all__ = [
    # Core
    "Engine",
    "Blocked",
    # Types
    "Gate",
    "Policy", 
    "Mode",
    "StoreErrorMode",
    "Status",
    "BlockReason",
    "Decision",
    "Result",
    # Storage
    "Store",
    "MemoryStore",
    "RedisStore",
]

__version__ = "0.1.0"
