#!/usr/bin/env python3
"""Reproducible benchmark suite for ActionGate.

Proves sub-millisecond check/enforce over 10K iterations using
time.perf_counter_ns for nanosecond-precision timing.

Usage:
    python benchmarks/bench_actiongate.py
    python benchmarks/bench_actiongate.py -n 50000
    python benchmarks/bench_actiongate.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass

from actiongate import Engine, Gate, Mode, Policy
from actiongate.store import AsyncMemoryStore, MemoryStore

# ── Constants ────────────────────────────────────────────────────

DEFAULT_ITERATIONS = 10_000
WARMUP_ITERATIONS = 1_000
SUB_MS_NS = 1_000_000  # 1ms in nanoseconds
HIGH_LIMIT = 1_000_000  # avoid blocking during bench
WINDOW_SECONDS = 3600.0
GATE_ROTATION_COUNT = 100


# ── Percentile helper ────────────────────────────────────────────

def percentile(sorted_data: list[int], p: float) -> int:
    """Percentile on pre-sorted nanosecond data."""
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return int(sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f]))


# ── Result type ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BenchResult:
    """Timing results for a single benchmark."""

    name: str
    iterations: int
    total_ns: int
    p50_ns: int
    p95_ns: int
    p99_ns: int

    @property
    def p50_us(self) -> float:
        return self.p50_ns / 1_000

    @property
    def p95_us(self) -> float:
        return self.p95_ns / 1_000

    @property
    def p99_us(self) -> float:
        return self.p99_ns / 1_000

    @property
    def ops_per_sec(self) -> float:
        return self.iterations / (self.total_ns / 1e9)

    @property
    def is_sub_ms(self) -> bool:
        return self.p99_ns < SUB_MS_NS

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_ms": round(self.total_ns / 1e6, 2),
            "p50_us": round(self.p50_us, 2),
            "p95_us": round(self.p95_us, 2),
            "p99_us": round(self.p99_us, 2),
            "ops_per_sec": round(self.ops_per_sec),
            "sub_ms": self.is_sub_ms,
        }

    def __str__(self) -> str:
        tag = "PASS" if self.is_sub_ms else "FAIL"
        return (
            f"  {self.name} [{tag}]\n"
            f"    iterations : {self.iterations:,}\n"
            f"    total      : {self.total_ns / 1e6:.2f} ms\n"
            f"    p50        : {self.p50_us:.2f} us\n"
            f"    p95        : {self.p95_us:.2f} us\n"
            f"    p99        : {self.p99_us:.2f} us\n"
            f"    throughput : {self.ops_per_sec:,.0f} ops/sec"
        )


# ── Runner ───────────────────────────────────────────────────────

def run_bench(
    name: str,
    fn: callable,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> BenchResult:
    """Time fn() over iterations using perf_counter_ns."""
    for _ in range(warmup):
        fn()

    latencies: list[int] = [0] * iterations
    start = time.perf_counter_ns()

    for i in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        latencies[i] = time.perf_counter_ns() - t0

    total = time.perf_counter_ns() - start
    latencies.sort()

    return BenchResult(
        name=name,
        iterations=iterations,
        total_ns=total,
        p50_ns=percentile(latencies, 50),
        p95_ns=percentile(latencies, 95),
        p99_ns=percentile(latencies, 99),
    )


# ── Benchmarks ───────────────────────────────────────────────────

def bench_check(n: int) -> BenchResult:
    """engine.check() — single gate, high limit."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "check", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS)

    return run_bench("check", lambda: engine.check(gate, policy), n)


def bench_enforce(n: int) -> BenchResult:
    """engine.enforce() on a pre-computed ALLOW decision."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "enforce", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS, mode=Mode.HARD)
    decision = engine.check(gate, policy)

    return run_bench("enforce", lambda: engine.enforce(decision), n)


def bench_check_and_enforce(n: int) -> BenchResult:
    """check + enforce round-trip — the real hot path."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "full", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS, mode=Mode.HARD)

    def op() -> None:
        d = engine.check(gate, policy)
        engine.enforce(d)

    return run_bench("check+enforce", op, n)


def bench_guard(n: int) -> BenchResult:
    """@engine.guard decorator overhead."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "guard", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS)

    @engine.guard(gate, policy)
    def noop() -> None:
        pass

    return run_bench("guard", noop, n)


def bench_guard_result(n: int) -> BenchResult:
    """@engine.guard_result decorator overhead."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "guard_result", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS, mode=Mode.SOFT)

    @engine.guard_result(gate, policy)
    def noop() -> int:
        return 42

    return run_bench("guard_result", noop, n)


def bench_async_check(n: int) -> BenchResult:
    """async_check() — single gate, high limit, event loop overhead."""
    engine = Engine(async_store=AsyncMemoryStore())
    gate = Gate("bench", "async_check", "user:1")
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS)
    loop = asyncio.new_event_loop()

    async def run() -> list[int]:
        latencies: list[int] = [0] * n
        for i in range(n):
            t0 = time.perf_counter_ns()
            await engine.async_check(gate, policy)
            latencies[i] = time.perf_counter_ns() - t0
        return latencies

    # warmup
    loop.run_until_complete(engine.async_check(gate, policy))

    start = time.perf_counter_ns()
    latencies = loop.run_until_complete(run())
    total = time.perf_counter_ns() - start
    loop.close()

    latencies.sort()
    return BenchResult(
        name="async_check",
        iterations=n,
        total_ns=total,
        p50_ns=percentile(latencies, 50),
        p95_ns=percentile(latencies, 95),
        p99_ns=percentile(latencies, 99),
    )


def bench_many_gates(n: int) -> BenchResult:
    """check() rotating across 100 gates — lock contention stress."""
    engine = Engine(store=MemoryStore())
    policy = Policy(max_calls=HIGH_LIMIT, window=WINDOW_SECONDS)
    gates = [
        Gate("bench", "multi", f"user:{i}")
        for i in range(GATE_ROTATION_COUNT)
    ]
    counter = [0]

    def op() -> None:
        engine.check(gates[counter[0] % GATE_ROTATION_COUNT], policy)
        counter[0] += 1

    return run_bench("check (100 gates)", op, n)


# ── Main ─────────────────────────────────────────────────────────

ALL_BENCHMARKS = [
    bench_check,
    bench_enforce,
    bench_check_and_enforce,
    bench_guard,
    bench_guard_result,
    bench_async_check,
    bench_many_gates,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="ActionGate benchmark suite")
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"iterations per benchmark (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output results as JSON",
    )
    args = parser.parse_args()

    results: list[BenchResult] = []
    for bench_fn in ALL_BENCHMARKS:
        results.append(bench_fn(args.iterations))

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print("=" * 56)
        print("ActionGate Benchmark Suite")
        print(f"  iterations: {args.iterations:,}  |  timer: perf_counter_ns")
        print("=" * 56)
        for r in results:
            print(r)
            print()

    is_all_sub_ms = all(r.is_sub_ms for r in results)

    if not args.json:
        verdict = "ALL SUB-MILLISECOND" if is_all_sub_ms else "REGRESSION DETECTED"
        print(f"Result: {verdict}")
        print("=" * 56)

    return 0 if is_all_sub_ms else 1


if __name__ == "__main__":
    sys.exit(main())
