"""Tests for ActionGate."""

import pytest
from actiongate import (
    MISSING,
    Engine,
    Gate,
    Policy,
    Mode,
    Blocked,
    Decision,
    Result,
    Status,
    BlockReason,
    StoreErrorMode,
)
from actiongate.core import _Missing


# ═══════════════════════════════════════════════════════════════
# Core types
# ═══════════════════════════════════════════════════════════════


class TestGate:
    """Gate identity model."""

    def test_basic_fields(self):
        g = Gate("api", "search", "user:123")
        assert g.namespace == "api"
        assert g.action == "search"
        assert g.principal == "user:123"

    def test_default_principal(self):
        g = Gate("api", "search")
        assert g.principal == "global"

    def test_str(self):
        g = Gate("api", "search", "user:123")
        assert str(g) == "api:search@user:123"

    def test_key(self):
        g = Gate("api", "search", "user:123")
        assert g.key == "ag:api:search:user:123"

    def test_equality(self):
        a = Gate("api", "search", "user:123")
        b = Gate("api", "search", "user:123")
        c = Gate("api", "search", "user:456")
        assert a == b
        assert a != c

    def test_hashable(self):
        g = Gate("api", "search")
        d = {g: "value"}
        assert d[Gate("api", "search")] == "value"

    def test_frozen(self):
        g = Gate("api", "search")
        with pytest.raises(AttributeError):
            g.namespace = "other"  # type: ignore[misc]


class TestPolicy:
    """Policy validation."""

    def test_defaults(self):
        p = Policy()
        assert p.max_calls == 3
        assert p.window == 60.0
        assert p.cooldown == 0.0
        assert p.mode == Mode.HARD
        assert p.on_store_error == StoreErrorMode.FAIL_CLOSED

    def test_zero_max_calls_allowed(self):
        p = Policy(max_calls=0)
        assert p.max_calls == 0

    def test_negative_max_calls_raises(self):
        with pytest.raises(ValueError, match="max_calls must be >= 0"):
            Policy(max_calls=-1)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window must be > 0"):
            Policy(window=-1)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown must be >= 0"):
            Policy(cooldown=-1)

    def test_none_window_allowed(self):
        p = Policy(window=None)
        assert p.window is None


class TestDecision:
    """Decision structure."""

    def test_allowed(self):
        d = Decision(
            status=Status.ALLOW,
            gate=Gate("api", "search"),
            policy=Policy(),
        )
        assert d.allowed is True
        assert d.blocked is False
        assert bool(d) is True

    def test_blocked(self):
        d = Decision(
            status=Status.BLOCK,
            gate=Gate("api", "search"),
            policy=Policy(),
            reason=BlockReason.RATE_LIMIT,
        )
        assert d.allowed is False
        assert d.blocked is True
        assert bool(d) is False

    def test_defaults(self):
        d = Decision(
            status=Status.ALLOW,
            gate=Gate("api", "search"),
            policy=Policy(),
        )
        assert d.reason is None
        assert d.message is None
        assert d.calls_in_window == 0
        assert d.time_since_last is None


class TestMissingSentinel:
    """MISSING sentinel for Result[T]."""

    def test_repr(self):
        assert repr(MISSING) == "<MISSING>"

    def test_identity(self):
        assert isinstance(MISSING, _Missing)


class TestResult:
    """Result wrapper with MISSING sentinel."""

    def test_allowed_with_value(self):
        d = Decision(
            status=Status.ALLOW,
            gate=Gate("api", "search"),
            policy=Policy(),
        )
        r: Result[str] = Result(decision=d, _value="hello")
        assert r.ok is True
        assert r.has_value is True
        assert r.value == "hello"
        assert r.unwrap() == "hello"
        assert r.unwrap_or("default") == "hello"

    def test_blocked_missing(self):
        d = Decision(
            status=Status.BLOCK,
            gate=Gate("api", "search"),
            policy=Policy(),
            reason=BlockReason.RATE_LIMIT,
        )
        r: Result[str] = Result(decision=d)
        assert r.ok is False
        assert r.has_value is False
        assert r.value is None
        with pytest.raises(ValueError, match="No value"):
            r.unwrap()
        assert r.unwrap_or("fallback") == "fallback"

    def test_blocked_with_explicit_missing(self):
        d = Decision(
            status=Status.BLOCK,
            gate=Gate("api", "search"),
            policy=Policy(),
            reason=BlockReason.RATE_LIMIT,
        )
        r: Result[str] = Result(decision=d, _value=MISSING)
        assert r.ok is False
        assert r.has_value is False
        assert r.value is None

    def test_none_as_legitimate_value(self):
        """CRITICAL: None return from guarded function must not be confused with blocked."""
        d = Decision(
            status=Status.ALLOW,
            gate=Gate("api", "search"),
            policy=Policy(),
        )
        r: Result[None] = Result(decision=d, _value=None)
        assert r.ok is True
        assert r.has_value is True
        assert r.value is None
        assert r.unwrap() is None  # Must not raise!
        assert r.unwrap_or("default") is None  # Must return None, not default!


# ═══════════════════════════════════════════════════════════════
# Engine: basic gating
# ═══════════════════════════════════════════════════════════════


class TestBasicGating:
    """Core rate limiting behavior."""

    def test_allows_up_to_max_calls(self):
        engine = Engine()
        gate = Gate("test", "action")
        policy = Policy(max_calls=3, window=60, mode=Mode.HARD)

        for _ in range(3):
            assert engine.check(gate, policy).allowed

    def test_blocks_after_max_calls(self):
        engine = Engine()
        gate = Gate("test", "action")
        policy = Policy(max_calls=2, window=60, mode=Mode.HARD)

        engine.check(gate, policy)
        engine.check(gate, policy)
        decision = engine.check(gate, policy)

        assert decision.blocked
        assert decision.reason == BlockReason.RATE_LIMIT

    def test_hard_mode_raises(self):
        engine = Engine()
        gate = Gate("test", "action")
        policy = Policy(max_calls=1, mode=Mode.HARD)

        engine.check(gate, policy)
        decision = engine.check(gate, policy)

        with pytest.raises(Blocked) as exc:
            engine.enforce(decision)
        assert exc.value.decision.reason == BlockReason.RATE_LIMIT

    def test_soft_mode_no_exception(self):
        engine = Engine()
        gate = Gate("test", "action")
        policy = Policy(max_calls=1, mode=Mode.SOFT)

        engine.check(gate, policy)
        decision = engine.check(gate, policy)

        engine.enforce(decision)  # Should not raise
        assert decision.blocked


# ═══════════════════════════════════════════════════════════════
# Engine: cooldown
# ═══════════════════════════════════════════════════════════════


class TestCooldown:
    """Cooldown enforcement."""

    def test_cooldown_blocks_rapid_calls(self):
        clock = MockClock(1000)
        engine = Engine(clock=clock)
        gate = Gate("test", "action")
        policy = Policy(max_calls=100, cooldown=10, mode=Mode.HARD)

        assert engine.check(gate, policy).allowed

        clock.advance(5)
        decision = engine.check(gate, policy)

        assert decision.blocked
        assert decision.reason == BlockReason.COOLDOWN

    def test_cooldown_allows_after_wait(self):
        clock = MockClock(1000)
        engine = Engine(clock=clock)
        gate = Gate("test", "action")
        policy = Policy(max_calls=100, cooldown=10)

        assert engine.check(gate, policy).allowed

        clock.advance(15)
        assert engine.check(gate, policy).allowed

    def test_cooldown_does_not_reset_count(self):
        clock = MockClock(1000)
        engine = Engine(clock=clock)
        gate = Gate("test", "action")
        policy = Policy(max_calls=2, window=60, cooldown=5, mode=Mode.HARD)

        assert engine.check(gate, policy).allowed
        clock.advance(10)
        assert engine.check(gate, policy).allowed

        clock.advance(10)
        decision = engine.check(gate, policy)

        assert decision.blocked
        assert decision.reason == BlockReason.RATE_LIMIT


# ═══════════════════════════════════════════════════════════════
# Engine: window expiry
# ═══════════════════════════════════════════════════════════════


class TestWindowExpiry:
    """Window-based count expiry."""

    def test_window_expiry_resets_count(self):
        clock = MockClock(1000)
        engine = Engine(clock=clock)
        gate = Gate("test", "action")
        policy = Policy(max_calls=2, window=60)

        engine.check(gate, policy)
        engine.check(gate, policy)

        clock.advance(70)
        assert engine.check(gate, policy).allowed


# ═══════════════════════════════════════════════════════════════
# Decorator: guard
# ═══════════════════════════════════════════════════════════════


class TestGuardDecorator:
    """@engine.guard decorator (returns T, raises on block)."""

    def test_returns_value_directly(self):
        engine = Engine()
        gate = Gate("test", "greet")

        @engine.guard(gate, Policy(max_calls=2))
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = greet("World")
        assert result == "Hello, World!"

    def test_raises_on_block(self):
        engine = Engine()
        gate = Gate("test", "limited")

        @engine.guard(gate, Policy(max_calls=1, mode=Mode.HARD))
        def limited() -> str:
            return "success"

        assert limited() == "success"
        with pytest.raises(Blocked):
            limited()

    def test_preserves_function_metadata(self):
        engine = Engine()

        @engine.guard(Gate("test", "action"), Policy())
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


# ═══════════════════════════════════════════════════════════════
# Decorator: guard_result
# ═══════════════════════════════════════════════════════════════


class TestGuardResultDecorator:
    """@engine.guard_result decorator (returns Result[T], never raises)."""

    def test_returns_result_wrapper(self):
        engine = Engine()
        gate = Gate("test", "greet")

        @engine.guard_result(gate, Policy(max_calls=2))
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = greet("World")
        assert result.ok
        assert result.has_value
        assert result.value == "Hello, World!"

    def test_blocked_returns_missing(self):
        engine = Engine()
        gate = Gate("test", "limited")

        @engine.guard_result(gate, Policy(max_calls=1, mode=Mode.SOFT))
        def limited() -> str:
            return "success"

        assert limited().value == "success"
        result = limited()

        assert not result.ok
        assert not result.has_value
        assert result.value is None

    def test_none_return_not_confused_with_blocked(self):
        """CRITICAL: Function returning None must not look like a block."""
        engine = Engine()

        @engine.guard_result(Gate("test", "void"), Policy(max_calls=10))
        def void_op() -> None:
            return None

        result = void_op()
        assert result.ok is True
        assert result.has_value is True
        assert result.value is None
        assert result.unwrap() is None

    def test_unwrap_success(self):
        engine = Engine()

        @engine.guard_result(Gate("test", "action"), Policy(max_calls=1))
        def action() -> int:
            return 42

        assert action().unwrap() == 42

    def test_unwrap_blocked_raises(self):
        engine = Engine()

        @engine.guard_result(Gate("test", "action"), Policy(max_calls=1, mode=Mode.SOFT))
        def action() -> int:
            return 42

        action()
        with pytest.raises(ValueError):
            action().unwrap()

    def test_unwrap_or_default(self):
        engine = Engine()

        @engine.guard_result(Gate("test", "action"), Policy(max_calls=1, mode=Mode.SOFT))
        def action() -> int:
            return 42

        action()
        assert action().unwrap_or(0) == 0


# ═══════════════════════════════════════════════════════════════
# Store error modes
# ═══════════════════════════════════════════════════════════════


class TestStoreErrorModes:
    """Store failure handling."""

    def test_fail_closed_blocks_on_error(self):
        engine = Engine(store=FailingStore())
        gate = Gate("test", "action")
        policy = Policy(max_calls=10, on_store_error=StoreErrorMode.FAIL_CLOSED)

        decision = engine.check(gate, policy)

        assert decision.blocked
        assert decision.reason == BlockReason.STORE_ERROR
        assert "fail-closed" in decision.message

    def test_fail_open_allows_on_error(self):
        engine = Engine(store=FailingStore())
        gate = Gate("test", "action")
        policy = Policy(max_calls=10, on_store_error=StoreErrorMode.FAIL_OPEN)

        decision = engine.check(gate, policy)

        assert decision.allowed
        assert decision.reason == BlockReason.STORE_ERROR
        assert "fail-open" in decision.message


# ═══════════════════════════════════════════════════════════════
# Principal scoping
# ═══════════════════════════════════════════════════════════════


class TestPrincipalScoping:
    """Verify gates are scoped by principal."""

    def test_different_principals_independent(self):
        engine = Engine()
        policy = Policy(max_calls=1, mode=Mode.HARD)

        gate_a = Gate("ns", "action", "user:A")
        gate_b = Gate("ns", "action", "user:B")

        engine.check(gate_a, policy)
        engine.check(gate_b, policy)

        assert engine.check(gate_a, policy).blocked
        assert engine.check(gate_b, policy).blocked


# ═══════════════════════════════════════════════════════════════
# Listeners
# ═══════════════════════════════════════════════════════════════


class TestListeners:
    """Decision listeners for observability."""

    def test_listener_receives_decisions(self):
        decisions = []
        engine = Engine()
        engine.on_decision(decisions.append)

        gate = Gate("test", "action")
        engine.check(gate, Policy(max_calls=2))
        engine.check(gate, Policy(max_calls=2))

        assert len(decisions) == 2
        assert all(d.status == Status.ALLOW for d in decisions)

    def test_listener_errors_dont_break_execution(self):
        def bad_listener(d):
            raise RuntimeError("oops")

        engine = Engine()
        engine.on_decision(bad_listener)

        gate = Gate("test", "action")
        decision = engine.check(gate, Policy())

        assert decision.allowed
        assert engine.listener_errors == 1


# ═══════════════════════════════════════════════════════════════
# Clear
# ═══════════════════════════════════════════════════════════════


class TestClear:
    """History clearing."""

    def test_clear_resets_gate(self):
        engine = Engine()
        gate = Gate("test", "action")
        policy = Policy(max_calls=1)

        engine.check(gate, policy)
        assert engine.check(gate, policy).blocked

        engine.clear(gate)
        assert engine.check(gate, policy).allowed

    def test_clear_all(self):
        engine = Engine()
        gate1 = Gate("ns", "a")
        gate2 = Gate("ns", "b")
        policy = Policy(max_calls=1)

        engine.check(gate1, policy)
        engine.check(gate2, policy)

        engine.clear_all()

        assert engine.check(gate1, policy).allowed
        assert engine.check(gate2, policy).allowed


# ─────────────────────────────────────────────────────────────────
# Test Utilities
# ─────────────────────────────────────────────────────────────────

class MockClock:
    """Controllable clock for testing."""

    def __init__(self, start: float = 0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FailingStore:
    """Store that always raises (for testing error handling)."""

    def check_and_reserve(self, gate, now, policy):
        raise ConnectionError("Redis connection failed")

    def clear(self, gate):
        raise ConnectionError("Redis connection failed")

    def clear_all(self):
        raise ConnectionError("Redis connection failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
