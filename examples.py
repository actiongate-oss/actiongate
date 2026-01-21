"""ActionGate usage examples with observability patterns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from actiongate import Engine, Gate, Policy, Mode, Decision, Status, Blocked


# ═══════════════════════════════════════════════════════════════════
# Basic Usage
# ═══════════════════════════════════════════════════════════════════

def demo_basic():
    """Simple rate limiting with guard decorator."""
    print("\n=== Basic Usage ===")
    
    engine = Engine()

    @engine.guard(
        Gate("api", "search", "user:123"),
        Policy(max_calls=3, window=60)
    )
    def search(query: str) -> str:
        return f"Results for: {query}"

    for i in range(4):
        try:
            result = search(f"query-{i}")
            print(f"  ✓ {result}")
        except Blocked as e:
            print(f"  ✗ Blocked: {e.decision.message}")


# ═══════════════════════════════════════════════════════════════════
# Result-based API (no exceptions)
# ═══════════════════════════════════════════════════════════════════

def demo_guard_result():
    """Using guard_result for graceful degradation."""
    print("\n=== Guard Result (No Exceptions) ===")
    
    engine = Engine()

    @engine.guard_result(
        Gate("billing", "refund", "agent:42"),
        Policy(max_calls=2, window=300, cooldown=5, mode=Mode.SOFT)
    )
    def process_refund(order_id: str) -> str:
        return f"Refunded {order_id}"

    for i in range(4):
        result = process_refund(f"ORDER-{i}")
        if result.ok:
            print(f"  ✓ {result.value}")
        else:
            fallback = result.unwrap_or("Refund queued for retry")
            print(f"  ✗ {result.decision.message} → {fallback}")


# ═══════════════════════════════════════════════════════════════════
# Observability: Prometheus/StatsD/Datadog Patterns
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MetricsCollector:
    """Example metrics collector (replace with your actual client)."""
    
    counters: dict[str, int] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=dict)
    
    def incr(self, name: str, tags: dict[str, str] | None = None) -> None:
        key = f"{name}:{tags}" if tags else name
        self.counters[key] = self.counters.get(key, 0) + 1
    
    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        key = f"{name}:{tags}" if tags else name
        self.histograms.setdefault(key, []).append(value)
    
    def summary(self) -> None:
        print("  Counters:", dict(self.counters))
        print("  Histograms:", {k: f"{len(v)} samples" for k, v in self.histograms.items()})


def create_metrics_listener(metrics: MetricsCollector) -> Callable[[Decision], None]:
    """Create a listener that emits metrics on every decision.
    
    Emits:
        - actiongate.decision (counter) with status/reason tags
        - actiongate.calls_in_window (histogram)
        - actiongate.time_since_last (histogram)
    """
    def listener(decision: Decision) -> None:
        tags = {
            "namespace": decision.gate.namespace,
            "action": decision.gate.action,
            "status": decision.status.name.lower(),
        }
        if decision.reason:
            tags["reason"] = decision.reason.name.lower()
        
        metrics.incr("actiongate.decision", tags)
        
        if decision.calls_in_window > 0:
            metrics.histogram("actiongate.calls_in_window", decision.calls_in_window, tags)
        
        if decision.time_since_last is not None:
            metrics.histogram("actiongate.time_since_last", decision.time_since_last, tags)
    
    return listener


def demo_observability():
    """Observability with metrics collection."""
    print("\n=== Observability ===")
    
    metrics = MetricsCollector()
    engine = Engine()
    engine.on_decision(create_metrics_listener(metrics))
    
    # Also add a simple logger
    engine.on_decision(
        lambda d: print(f"  [{d.status.name}] {d.gate} (calls={d.calls_in_window})")
    )
    
    gate = Gate("demo", "action")
    policy = Policy(max_calls=3, window=60)
    
    for _ in range(5):
        engine.check(gate, policy)
    
    print("\n  --- Metrics Summary ---")
    metrics.summary()


# ═══════════════════════════════════════════════════════════════════
# Production Pattern: Structured Logging
# ═══════════════════════════════════════════════════════════════════

def create_json_logger() -> Callable[[Decision], None]:
    """Structured logging listener (for production use with JSON logs)."""
    import json
    
    def listener(decision: Decision) -> None:
        log_entry = {
            "event": "actiongate.decision",
            "gate": str(decision.gate),
            "status": decision.status.name,
            "reason": decision.reason.name if decision.reason else None,
            "calls_in_window": decision.calls_in_window,
            "time_since_last": decision.time_since_last,
            "policy": {
                "max_calls": decision.policy.max_calls,
                "window": decision.policy.window,
                "cooldown": decision.policy.cooldown,
            }
        }
        print(f"  LOG: {json.dumps(log_entry)}")
    
    return listener


def demo_structured_logging():
    """Structured JSON logging for production."""
    print("\n=== Structured Logging ===")
    
    engine = Engine()
    engine.on_decision(create_json_logger())
    
    gate = Gate("api", "expensive_call", "user:456")
    policy = Policy(max_calls=2, window=60)
    
    for _ in range(3):
        engine.check(gate, policy)


# ═══════════════════════════════════════════════════════════════════
# Run All Demos
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_basic()
    demo_guard_result()
    demo_observability()
    demo_structured_logging()
