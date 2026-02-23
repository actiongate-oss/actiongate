"""State storage backends for ActionGate."""

from __future__ import annotations

import secrets
import threading
from typing import TYPE_CHECKING, Protocol

from .core import Gate, Policy

if TYPE_CHECKING:
    from redis import Redis


class Store(Protocol):
    """Storage backend protocol.
    
    Implementations must provide atomic check-and-reserve semantics
    for correct behavior under concurrency.
    """

    def check_and_reserve(
        self, gate: Gate, now: float, policy: Policy
    ) -> tuple[int, bool, float | None]:
        """Atomically evaluate and (if allowed) reserve a slot.
        
        Returns:
            (count_in_window, allowed, seconds_since_last)
        """
        ...

    def clear(self, gate: Gate) -> None:
        """Clear history for a specific gate."""
        ...

    def clear_all(self) -> None:
        """Clear all history."""
        ...


class MemoryStore:
    """Thread-safe in-memory store with per-gate locking.
    
    Suitable for single-process deployments. For distributed systems,
    use RedisStore instead.
    """

    __slots__ = ("_events", "_locks", "_global_lock")

    def __init__(self) -> None:
        self._events: dict[Gate, list[float]] = {}
        self._locks: dict[Gate, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, gate: Gate) -> threading.Lock:
        with self._global_lock:
            if gate not in self._locks:
                self._locks[gate] = threading.Lock()
            return self._locks[gate]

    def _prune(self, events: list[float], now: float, window: float | None) -> list[float]:
        if window is None:
            return sorted(events)
        cutoff = now - window
        return sorted(t for t in events if t >= cutoff)

    def check_and_reserve(
        self, gate: Gate, now: float, policy: Policy
    ) -> tuple[int, bool, float | None]:
        with self._get_lock(gate):
            events = self._events.get(gate, [])
            pruned = self._prune(events, now, policy.window)
            last_age = (now - pruned[-1]) if pruned else None

            # Check cooldown first
            if policy.cooldown > 0 and last_age is not None and last_age < policy.cooldown:
                self._events[gate] = pruned
                return len(pruned), False, last_age

            # Check rate limit
            if len(pruned) >= policy.max_calls:
                self._events[gate] = pruned
                return len(pruned), False, last_age

            # Allowed - reserve slot
            pruned.append(now)
            self._events[gate] = pruned
            return len(pruned), True, last_age

    def clear(self, gate: Gate) -> None:
        with self._get_lock(gate):
            self._events.pop(gate, None)

    def clear_all(self) -> None:
        with self._global_lock:
            self._events.clear()
            self._locks.clear()


# ─────────────────────────────────────────────────────────────────
# Redis Store (Lua-based atomic operations)
# ─────────────────────────────────────────────────────────────────

# Lua script for atomic check-and-reserve
# Uses ZSET with score=timestamp, member="{timestamp}:{nonce}" for uniqueness
_LUA_CHECK_AND_RESERVE = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = ARGV[2]
local cooldown = tonumber(ARGV[3])
local max_calls = tonumber(ARGV[4])
local member = ARGV[5]

local window_num = (window ~= "none") and tonumber(window) or nil

-- Prune expired entries if window is set
if window_num then
    local cutoff = now - window_num
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
end

-- Get current events
local events = redis.call('ZRANGE', key, 0, -1, 'WITHSCORES')
local count = #events / 2
local last_ts = nil

if count > 0 then
    last_ts = tonumber(events[#events])
end

local last_age = nil
if last_ts then
    last_age = now - last_ts
end

-- Check cooldown
if cooldown > 0 and last_age and last_age < cooldown then
    return {count, 0, tostring(last_age)}
end

-- Check rate limit
if max_calls >= 0 and count >= max_calls then
    return {count, 0, last_age and tostring(last_age) or "nil"}
end

-- Allowed - reserve slot
redis.call('ZADD', key, now, member)

-- Set TTL to max(window, cooldown) * 1.5 to prevent key leaks
-- Cooldown-only policies (window=none) still need TTL based on cooldown
local ttl_base = 0
if window_num then
    ttl_base = window_num
end
if cooldown > ttl_base then
    ttl_base = cooldown
end
if ttl_base > 0 then
    redis.call('EXPIRE', key, math.ceil(ttl_base * 1.5))
end

return {count + 1, 1, last_age and tostring(last_age) or "nil"}
"""


class RedisStore:
    """Redis-backed store using ZSET + Lua for atomic operations.
    
    Suitable for distributed deployments. Requires redis-py client.
    
    Example:
        import redis
        from actiongate import Engine, RedisStore
        
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        engine = Engine(store=RedisStore(client))
    
    Implementation notes:
        - Uses ZSET with score=timestamp for range queries
        - Member format: "{timestamp}:{nonce}" prevents collision under concurrency
        - Lua script ensures atomicity across check + reserve
        - Auto-expires keys at max(window, cooldown) * 1.5 to prevent leaks
    """

    __slots__ = ("_client", "_script", "_prefix")

    def __init__(self, client: "Redis", prefix: str = "actiongate") -> None:
        """
        Args:
            client: Redis client instance (redis-py)
            prefix: Key prefix for namespacing (default: "actiongate")
        """
        self._client = client
        self._prefix = prefix
        self._script = self._client.register_script(_LUA_CHECK_AND_RESERVE)

    def _key(self, gate: Gate) -> str:
        return f"{self._prefix}:{gate.namespace}:{gate.action}:{gate.principal}"

    def check_and_reserve(
        self, gate: Gate, now: float, policy: Policy
    ) -> tuple[int, bool, float | None]:
        key = self._key(gate)
        
        # Generate unique member: timestamp + random nonce
        nonce = secrets.token_hex(4)
        member = f"{now}:{nonce}"
        
        window_arg = str(policy.window) if policy.window is not None else "none"
        
        result = self._script(
            keys=[key],
            args=[
                str(now),
                window_arg,
                str(policy.cooldown),
                str(policy.max_calls),
                member,
            ]
        )
        
        count = int(result[0])
        allowed = bool(int(result[1]))
        last_age_str = result[2]
        last_age = float(last_age_str) if last_age_str != "nil" else None
        
        return count, allowed, last_age

    def clear(self, gate: Gate) -> None:
        self._client.delete(self._key(gate))

    def clear_all(self) -> None:
        """Clear all actiongate keys.
        
        Warning: Uses SCAN, may be slow on large keyspaces.
        """
        pattern = f"{self._prefix}:*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=pattern, count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break
