"""Tests for ActionGate."""

import pytest
from actiongate import (
    Engine, Gate, Policy, Mode, Blocked, Status, BlockReason, StoreErrorMode
)


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
        assert result.value == "Hello, World!"

    def test_blocked_returns_none_value(self):
        engine = Engine()
        gate = Gate("test", "limited")

        @engine.guard_result(gate, Policy(max_calls=1, mode=Mode.SOFT))
        def limited() -> str:
            return "success"

        assert limited().value == "success"
        result = limited()

        assert not result.ok
        assert result.value is None


class TestResult:
    """Result wrapper methods."""

    def test_unwrap_success(self):
        engine = Engine()
        gate = Gate("test", "action")

        @engine.guard_result(gate, Policy(max_calls=1))
        def action() -> int:
            return 42

        assert action().unwrap() == 42

    def test_unwrap_blocked_raises(self):
        engine = Engine()
        gate = Gate("test", "action")

        @engine.guard_result(gate, Policy(max_calls=1, mode=Mode.SOFT))
        def action() -> int:
            return 42

        action()
        with pytest.raises(ValueError):
            action().unwrap()

    def test_unwrap_or_default(self):
        engine = Engine()
        gate = Gate("test", "action")

        @engine.guard_result(gate, Policy(max_calls=1, mode=Mode.SOFT))
        def action() -> int:
            return 42

        action()
        assert action().unwrap_or(0) == 0


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
