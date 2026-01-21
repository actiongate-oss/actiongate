# ActionGate Semantics

This document defines the normative behavior of ActionGate. Any implementation claiming compatibility must conform to these semantics.

Version: 0.1

---

## 1. Purpose

ActionGate is a **deterministic, pre-execution action gate** for AI agents. It prevents action spam by enforcing rate limits and cooldowns before execution occurs.

It is not a policy engine, cost manager, orchestrator, or authorization system.

---

## 2. Action Identity

An action is identified by a **Gate**, a 3-tuple:

```
Gate = (namespace: string, action: string, principal: string)
```

| Field | Purpose | Examples |
|-------|---------|----------|
| `namespace` | Domain or subsystem | `"billing"`, `"api"`, `"support"` |
| `action` | Operation name | `"refund"`, `"search"`, `"send_email"` |
| `principal` | Scope of enforcement | `"user:123"`, `"agent:42"`, `"global"` |

Two gates are equal iff all three fields are equal. Enforcement state is **not shared** across distinct gates.

---

## 3. Policy

A **Policy** defines enforcement parameters:

| Parameter | Type | Meaning |
|-----------|------|---------|
| `max_calls` | int ≥ 0 | Maximum allowed calls within `window` |
| `window` | float > 0 \| null | Rolling window in seconds; null = unbounded |
| `cooldown` | float ≥ 0 | Minimum seconds between consecutive calls |
| `mode` | HARD \| SOFT | HARD raises on block; SOFT returns decision |
| `on_store_error` | FAIL_CLOSED \| FAIL_OPEN | Behavior when storage backend fails |

---

## 4. Enforcement Rules

### 4.1 Pre-Execution

All enforcement occurs **before** the guarded action executes. If blocked, the action **must not** execute.

### 4.2 Decision Logic

Given a gate G and policy P, at time T:

1. **Prune**: Remove all recorded events older than `T - P.window` (if window is set)
2. **Cooldown check** (if `P.cooldown > 0`):
   - Let `last` = timestamp of most recent event
   - If `T - last < P.cooldown` → **BLOCK** (reason: COOLDOWN)
3. **Rate limit check**:
   - Let `count` = number of events remaining after prune
   - If `count >= P.max_calls` → **BLOCK** (reason: RATE_LIMIT)
4. Otherwise → **ALLOW**

### 4.3 Reservation

If and only if the decision is ALLOW, the implementation **must atomically record** an event at time T before returning. This prevents concurrent calls from both passing.

### 4.4 Cooldown Does Not Reset Count

Waiting out a cooldown does not reduce `count`. Only window expiry removes events from the count.

---

## 5. Atomicity

The check-and-reserve operation **must be atomic** with respect to concurrent callers on the same gate. Implementations using shared storage (Redis, database) must use atomic primitives (e.g., Lua scripts, transactions).

A non-atomic implementation may allow more than `max_calls` executions under concurrency. This is a conformance violation.

---

## 6. Failure Semantics

When the storage backend is unavailable or errors:

| `on_store_error` | Behavior |
|------------------|----------|
| `FAIL_CLOSED` | Return BLOCK with reason STORE_ERROR |
| `FAIL_OPEN` | Return ALLOW with reason STORE_ERROR |

The decision **must** include the `STORE_ERROR` reason to distinguish from normal blocks/allows.

---

## 7. Decision Structure

Every evaluation **must** return a Decision containing at minimum:

| Field | Type | Meaning |
|-------|------|---------|
| `status` | ALLOW \| BLOCK | Outcome |
| `gate` | Gate | The evaluated gate |
| `policy` | Policy | The policy used |
| `reason` | BlockReason \| null | Why blocked (null if allowed) |
| `calls_in_window` | int | Event count at decision time |
| `time_since_last` | float \| null | Seconds since last event (null if none) |

This enables full observability and auditability of every decision.

---

## 8. Out of Scope

ActionGate **does not** and **must not**:

- Make LLM or model inference calls
- Perform semantic or intent analysis
- Manage costs, budgets, or billing
- Provide authentication or authorization
- Implement circuit breakers, retries, or backpressure
- Orchestrate multi-agent workflows
- Make decisions based on action content or arguments

ActionGate is a **stateless-per-request, stateful-per-gate** primitive. It examines only the gate identity and timing, never the payload.

---

## 9. Compatibility

An implementation is **ActionGate-compatible** iff:

1. It implements the Gate identity model (§2)
2. It implements the Policy parameters (§3)
3. It follows the enforcement rules exactly (§4)
4. Check-and-reserve is atomic (§5)
5. Failure modes match the specification (§6)
6. Decisions include all required fields (§7)
7. It does not extend scope beyond §8

Compatible implementations may:
- Use any storage backend
- Be written in any language
- Add non-normative fields to Decision
- Provide additional observability hooks

Compatible implementations must not:
- Change the decision logic
- Make blocking conditional on payload content
- Skip reservation on ALLOW

---

## 10. Reference Implementation

The canonical reference implementation is at:

```
https://github.com/yourname/actiongate
```

When this specification and the reference implementation conflict, **this specification governs**.

---

## Changelog

- **0.1** (2025-01): Initial specification
