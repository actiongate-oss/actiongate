"""Benchmark: ActionGate gate evaluation latency.

Measures check(), guard(), and guard_result() paths against MemoryStore.
Run: python bench/bench.py
"""

from __future__ import annotations

import statistics
import time

from actiongate import Decision, Engine, Gate, Mode, MemoryStore, Policy, Status


def _report(name: str, timings_ns: list[int]) -> dict[str, float]:
    timings_us = [t / 1_000 for t in timings_ns]
    n = len(timings_us)
    p50 = statistics.median(timings_us)
    p95 = sorted(timings_us)[int(n * 0.95)]
    p99 = sorted(timings_us)[int(n * 0.99)]
    mean = statistics.mean(timings_us)
    mn = min(timings_us)
    mx = max(timings_us)

    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")
    print(f"  Iterations : {n:,}")
    print(f"  Mean       : {mean:>10.2f} µs")
    print(f"  Median p50 : {p50:>10.2f} µs")
    print(f"  p95        : {p95:>10.2f} µs")
    print(f"  p99        : {p99:>10.2f} µs")
    print(f"  Min        : {mn:>10.2f} µs")
    print(f"  Max        : {mx:>10.2f} µs")

    return {"mean": mean, "p50": p50, "p95": p95, "p99": p99}


def bench_check_allow(n: int = 10_000) -> list[int]:
    """Benchmark Engine.check() on the ALLOW path."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "check", "user:0")
    policy = Policy(max_calls=n + 1000, window=60.0)
    engine.register(gate, policy)

    # Warmup
    for _ in range(200):
        engine.check(gate)

    engine.clear(gate)

    timings: list[int] = []
    for _ in range(n):
        start = time.perf_counter_ns()
        decision = engine.check(gate)
        elapsed = time.perf_counter_ns() - start
        assert decision.allowed
        timings.append(elapsed)

    return timings


def bench_check_block(n: int = 10_000) -> list[int]:
    """Benchmark Engine.check() on the BLOCK path (early return)."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "block", "user:0")
    policy = Policy(max_calls=1, window=60.0)
    engine.register(gate, policy)

    # Fill the gate
    engine.check(gate)

    timings: list[int] = []
    for _ in range(n):
        start = time.perf_counter_ns()
        decision = engine.check(gate)
        elapsed = time.perf_counter_ns() - start
        assert decision.blocked
        timings.append(elapsed)

    return timings


def bench_guard_decorator(n: int = 10_000) -> list[int]:
    """Benchmark @engine.guard() decorator overhead (ALLOW path)."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "guard", "user:0")
    policy = Policy(max_calls=n + 1000, window=60.0)

    @engine.guard(gate, policy)
    def noop() -> int:
        return 42

    # Warmup
    for _ in range(200):
        noop()

    engine.clear(gate)

    timings: list[int] = []
    for _ in range(n):
        start = time.perf_counter_ns()
        result = noop()
        elapsed = time.perf_counter_ns() - start
        assert result == 42
        timings.append(elapsed)

    return timings


def bench_guard_result_decorator(n: int = 10_000) -> list[int]:
    """Benchmark @engine.guard_result() decorator overhead (ALLOW path)."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "guard_result", "user:0")
    policy = Policy(max_calls=n + 1000, window=60.0)

    @engine.guard_result(gate, policy)
    def noop() -> int:
        return 42

    # Warmup
    for _ in range(200):
        noop()

    engine.clear(gate)

    timings: list[int] = []
    for _ in range(n):
        start = time.perf_counter_ns()
        result = noop()
        elapsed = time.perf_counter_ns() - start
        assert result.ok
        assert result.unwrap() == 42
        timings.append(elapsed)

    return timings


def main() -> None:
    print("ActionGate Benchmark")
    print(f"Python perf_counter_ns resolution: ~{time.get_clock_info('perf_counter').resolution * 1e9:.0f} ns")

    results: dict[str, dict[str, float]] = {}

    timings = bench_check_allow()
    results["check (ALLOW)"] = _report("Engine.check() — ALLOW path", timings)

    timings = bench_check_block()
    results["check (BLOCK)"] = _report("Engine.check() — BLOCK path", timings)

    timings = bench_guard_decorator()
    results["guard"] = _report("@engine.guard() — ALLOW path", timings)

    timings = bench_guard_result_decorator()
    results["guard_result"] = _report("@engine.guard_result() — ALLOW path", timings)

    print(f"\n{'=' * 64}")
    print("  Summary")
    print(f"{'=' * 64}")
    for name, r in results.items():
        print(f"  {name:<25s}  p50={r['p50']:.2f}µs  p99={r['p99']:.2f}µs")


if __name__ == "__main__":
    main()
