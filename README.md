# ActionGate

Deterministic, pre-execution gating for semantic actions (e.g. tool calls) in agent systems.

## Source of Truth

The canonical source is [github.com/actiongate-oss/actiongate](https://github.com/actiongate-oss/actiongate). PyPI distribution is a convenience mirror—verify against the repo if provenance matters to you.

## Install

```bash
pip install actiongate            # PyPI
pip install actiongate[redis]     # with Redis support
pip install -e .                  # from source
```

**Vendoring encouraged.** This is a small, stable primitive. Copy it into your codebase, fork it, reimplement it in Rust/Go/whatever. See [SEMANTICS.md](SEMANTICS.md) for the behavioral contract if you reimplement.

---

## Quick Start

```python
from actiongate import Engine, Gate, Policy, Blocked

engine = Engine()

@engine.guard(
    Gate("api", "search", "user:123"),
    Policy(max_calls=5, window=60)
)
def search(query: str) -> list[str]:
    return api.search(query)

try:
    results = search("hello")
except Blocked as e:
    print(f"Rate limited: {e.decision.message}")
```

## Core Concepts

### Gate

Identifies what's being rate-limited:

```python
Gate(namespace, action, principal)

Gate("billing", "refund", "user:123")    # per-user
Gate("support", "escalate", "agent:42")  # per-agent  
Gate("api", "search", "global")          # global limit
```

### Policy

```python
Policy(
    max_calls=5,        # allow N calls per window
    window=60,          # rolling window (seconds)
    cooldown=2,         # min seconds between calls
    mode=Mode.HARD,     # HARD raises, SOFT returns Result
    on_store_error=StoreErrorMode.FAIL_CLOSED
)
```

### Two Decorator Styles

```python
@engine.guard(gate, policy)        # returns T, raises Blocked
@engine.guard_result(gate, policy) # returns Result[T], never raises
```

---

## Scope & Non-Goals

**ActionGate does:**
- Pre-execution rate limiting (allow N, block N+1)
- Cooldown enforcement (min time between calls)
- Atomic check-and-reserve for correctness under concurrency
- Deterministic decisions with full explainability

**ActionGate does not:**
- Make LLM calls or consume tokens
- Do semantic/intent analysis
- Manage cost, budgets, or billing
- Orchestrate multi-agent workflows
- Provide authentication or authorization
- Replace circuit breakers, retries, or backpressure

See [SEMANTICS.md](SEMANTICS.md) for the formal behavioral contract.

---

## Distributed: Redis Backend

```python
import redis
from actiongate import Engine, RedisStore

client = redis.Redis(host='localhost', port=6379, decode_responses=True)
engine = Engine(store=RedisStore(client))
```

- Atomic via Lua script
- Keys auto-expire at `max(window, cooldown) × 1.5`
- Member format `{timestamp}:{nonce}` prevents collision

---

## Observability

```python
engine.on_decision(lambda d: logger.info(f"{d.status}: {d.gate}"))
```

Every decision includes: status, reason, gate, policy, calls_in_window, time_since_last. See [examples.py](examples.py) for Prometheus/StatsD patterns.

---

## Benchmarks

```bash
python -m actiongate.bench
python -m actiongate.bench --redis localhost:6379
```

Results from local benchmark on Apple M1, Python 3.12. Run to reproduce on your hardware.

| Store | p50 | p95 | p99 | Throughput |
|-------|-----|-----|-----|------------|
| MemoryStore | ~2μs | ~3μs | ~5μs | ~400k ops/s |
| RedisStore (localhost) | ~200μs | ~300μs | ~500μs | ~4k ops/s |

---

## API Reference

| Type | Purpose |
|------|---------|
| `Engine` | Core gating logic |
| `Gate` | Action identity tuple |
| `Policy` | Rate limit configuration |
| `Decision` | Evaluation result with full context |
| `Result[T]` | Wrapper for `guard_result` |
| `Blocked` | Exception from `guard` |
| `MemoryStore` | Single-process backend |
| `RedisStore` | Distributed backend |

| Enum | Values |
|------|--------|
| `Mode` | `HARD`, `SOFT` |
| `StoreErrorMode` | `FAIL_CLOSED`, `FAIL_OPEN` |
| `Status` | `ALLOW`, `BLOCK` |
| `BlockReason` | `RATE_LIMIT`, `COOLDOWN`, `STORE_ERROR` |

---

## Support

This project is published as-is under the MIT license. No support, SLA, or maintenance is guaranteed.

## License

MIT
