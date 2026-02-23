#!/usr/bin/env python3
"""Benchmark script for ActionGate.

Measures p50/p95/p99 latencies for MemoryStore and RedisStore.

Usage:
    # As module (recommended)
    python -m actiongate.bench
    
    # Include Redis benchmarks
    python -m actiongate.bench --redis localhost:6379
    
    # Custom iterations
    python -m actiongate.bench -n 50000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

# Use relative imports when run as module, absolute when run directly
try:
    from . import Engine, Gate, Policy, MemoryStore
except ImportError:
    from actiongate import Engine, Gate, Policy, MemoryStore


@dataclass
class BenchResult:
    """Benchmark results."""
    name: str
    iterations: int
    total_ms: float
    p50_us: float
    p95_us: float
    p99_us: float
    ops_per_sec: float

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations:,}\n"
            f"  Total:      {self.total_ms:.2f}ms\n"
            f"  p50:        {self.p50_us:.2f}μs\n"
            f"  p95:        {self.p95_us:.2f}μs\n"
            f"  p99:        {self.p99_us:.2f}μs\n"
            f"  Throughput: {self.ops_per_sec:,.0f} ops/sec"
        )


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile."""
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def benchmark(
    name: str,
    fn: Callable[[], None],
    iterations: int = 10000,
    warmup: int = 1000,
) -> BenchResult:
    """Run benchmark and collect timing statistics."""
    
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Benchmark
    latencies: list[float] = []
    start = time.perf_counter()
    
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000_000)  # Convert to microseconds
    
    total = time.perf_counter() - start
    latencies.sort()
    
    return BenchResult(
        name=name,
        iterations=iterations,
        total_ms=total * 1000,
        p50_us=percentile(latencies, 50),
        p95_us=percentile(latencies, 95),
        p99_us=percentile(latencies, 99),
        ops_per_sec=iterations / total,
    )


def bench_memory_store(iterations: int) -> BenchResult:
    """Benchmark in-memory store."""
    engine = Engine(store=MemoryStore())
    gate = Gate("bench", "action", "user:test")
    policy = Policy(max_calls=1_000_000, window=3600)  # High limit to avoid blocking
    
    def op():
        engine.check(gate, policy)
    
    return benchmark("MemoryStore", op, iterations)


def bench_memory_store_many_gates(iterations: int) -> BenchResult:
    """Benchmark with many different gates (lock contention test)."""
    engine = Engine(store=MemoryStore())
    policy = Policy(max_calls=1_000_000, window=3600)
    counter = [0]
    
    def op():
        gate = Gate("bench", "action", f"user:{counter[0] % 100}")
        counter[0] += 1
        engine.check(gate, policy)
    
    return benchmark("MemoryStore (100 gates)", op, iterations)


def bench_redis_store(host: str, port: int, iterations: int) -> BenchResult | None:
    """Benchmark Redis store."""
    try:
        import redis
    except ImportError:
        print("  ⚠ redis-py not installed, skipping Redis benchmarks")
        print("    Install with: pip install actiongate[redis]")
        return None
    
    try:
        from . import RedisStore
    except ImportError:
        from actiongate import RedisStore
    
    try:
        client = redis.Redis(host=host, port=port, decode_responses=True)
        client.ping()
    except redis.ConnectionError:
        print(f"  ⚠ Cannot connect to Redis at {host}:{port}, skipping")
        return None
    
    store = RedisStore(client, prefix="bench")
    engine = Engine(store=store)
    gate = Gate("bench", "action", "user:test")
    policy = Policy(max_calls=1_000_000, window=3600)
    
    # Cleanup before benchmark
    store.clear(gate)
    
    def op():
        engine.check(gate, policy)
    
    result = benchmark("RedisStore", op, iterations)
    
    # Cleanup after benchmark
    store.clear(gate)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="ActionGate benchmarks")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=10000,
        help="Number of iterations (default: 10000)"
    )
    parser.add_argument(
        "--redis",
        type=str,
        default=None,
        help="Redis host:port (e.g., localhost:6379)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ActionGate Benchmarks")
    print("=" * 60)
    
    # Memory benchmarks
    print("\n📊 MemoryStore Benchmarks\n")
    
    result = bench_memory_store(args.iterations)
    print(result)
    print()
    
    result = bench_memory_store_many_gates(args.iterations)
    print(result)
    
    # Redis benchmarks
    if args.redis:
        print("\n📊 RedisStore Benchmarks\n")
        
        host, port = args.redis.split(":")
        result = bench_redis_store(host, int(port), min(args.iterations, 5000))
        if result:
            print(result)
    else:
        print("\n💡 Run with --redis localhost:6379 to include Redis benchmarks")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
