# 03 — The Risk Gate

Pure. Stdlib only. **No LLM, no I/O, no clock read, no network.** This is the
layer that makes the project's claim true, so it is the layer with the least
freedom.

## D1 — The contract

```python
def evaluate(
    proposal: TradeProposal,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    as_of: datetime,
) -> Verdict
```

One pure function. Everything it needs is an argument — including the time.

## D2 — Input: what the agent proposes

```python
@dataclass(frozen=True, slots=True)
class TradeProposal:
    structure: OptionStructure
    risk: StructureRisk
    quantity: int
    intent: Intent              # OPEN | CLOSE | ROLL
    rationale: str              # the model's reasoning, carried for the journal
    proposed_by: str            # model id, e.g. "claude-opus-5"
    proposal_id: str            # stable identity; feeds the idempotency key (04 D4)
    risk_as_of: datetime        # the instant `risk` was computed against
```

`rationale` is **evidence, not input**. The Gate never parses it. It exists so
the journal ([06](06-journal.md)) can show why a trade was proposed next to
whether it was allowed.

`risk_as_of` exists because a model call sits between perception and this Gate.
`StructureRisk.quote_age_seconds` was true when the agent did the arithmetic; by
the time the verdict is written the quotes may be minutes older. The Gate ages
them forward — `quote_age_seconds + (as_of - risk_as_of)` — so `fresh_quotes`
is a check on the quotes rather than on how recently somebody computed with
them. Inside a single tick, where perception and gating share one `as_of`, the
elapsed term is zero and the number is exactly what `compute_risk` produced.

## D3 — Output: a verdict, and only the Gate can construct approval

```python
class Verdict:  # sealed: Approved | Vetoed
    ...

@dataclass(frozen=True, slots=True)
class Approved:
    order: GatedOrder           # only constructible inside risk.gate
    checks: tuple[CheckResult, ...]

@dataclass(frozen=True, slots=True)
class Vetoed:
    reasons: tuple[VetoReason, ...]   # non-empty
    checks: tuple[CheckResult, ...]
```

`execution/` accepts a `GatedOrder` and nothing else. There is no public
constructor, no `force`, no override. This is Rule 2 of [01](01-architecture.md),
and it is enforced by a boundary test, not by discipline.

**Every check runs, always.** The Gate does not short-circuit on first veto —
it returns the complete `checks` tuple. A veto with one reason and a veto with
five are different situations and the journal should show which.

## D4 — The checks

Each is a pure predicate over `(proposal, portfolio, limits, as_of)`. All are
`CheckResult(name, passed, detail, observed, limit)` so the dashboard can render
the near-misses, not just the failures.

**Structural — cannot be configured away:**

| Check | Veto when |
| --- | --- |
| `defined_risk` | `risk.max_loss` is not finite and positive |
| `known_greeks` | `intent is OPEN` and `risk.net_greeks is None` |
| `fresh_quotes` | quote age at `as_of` exceeds `limits.max_quote_age` |

**Budgeted — configurable, but never disabled:**

| Check | Veto when |
| --- | --- |
| `per_trade_loss` | `max_loss * quantity` > `limits.max_trade_loss` |
| `portfolio_heat` | total open `max_loss` after this fill > `limits.max_portfolio_loss` |
| `position_count` | open structures ≥ `limits.max_open_structures` |
| `underlying_concentration` | exposure to one underlying > `limits.max_per_underlying` |
| `net_delta_budget` | portfolio net delta after fill outside `limits.delta_band` |
| `net_vega_budget` | portfolio net vega after fill outside `limits.vega_band` |
| `liquidity` | `risk.worst_spread_pct` > `limits.max_spread_pct` |
| `expiry_window` | `days_to_expiry` outside `limits.dte_range` |
| `drawdown_killswitch` | `portfolio.drawdown_pct` ≥ `limits.max_drawdown` |
| `daily_trade_cap` | fills today ≥ `limits.max_daily_trades` |

`drawdown_killswitch` is the one that matters for the P&L criterion. Once it
trips, `intent is OPEN` is vetoed unconditionally until manually re-armed;
`CLOSE` always remains permitted. **The Gate must never block an exit.**

The kill switch latch is state the *caller* carries in
(`PortfolioSnapshot.killswitch_tripped`), not state the Gate keeps: a pure
function cannot remember yesterday. Re-arming is a human clearing that flag
before the next snapshot is built, which is why it cannot happen by accident.

"Never block an exit" is implemented as a waiver rather than a skip. For
`intent is CLOSE` every check still *runs* and still reports what it observed —
the dashboard has to be able to say the exit went out over a 90% spread — but a
failure is recorded as passed-with-waiver instead of becoming a `VetoReason`.
A skipped check would leave the tape silent about the very situation that most
needs explaining afterwards.

`ROLL` counts as an open for every budget: the near leg goes away, but a new
position with its own maximum loss arrives. `known_greeks` is the one exception,
scoped to `intent is OPEN` exactly as written above — an unknown net on a roll
is caught by the delta and vega budgets, which refuse an unknown outright.

## D5 — Defaults (competition configuration)

Chosen for the four-day window and the "defensible P&L" goal in [00](00-brief.md).
These live in one module, are logged at startup, and are shown in the dashboard.

```
max_trade_loss        1.0% of equity
max_portfolio_loss    5.0% of equity
max_open_structures   8
max_per_underlying    2.0% of equity
delta_band            (-0.30, +0.30) per $1k equity, normalised
vega_band             (-50, +50) per $1k equity
max_spread_pct        5%
dte_range             (3, 21)      # no 0DTE; see 07 D5
max_drawdown          5%
max_daily_trades      15
max_quote_age         60s
```

`dte_range` is a *strategy* claim as much as a risk one, and it is now bounded
on both sides. Excluding 0DTE: over four days it is the fastest way to turn the
P&L score into a coin flip — the 2025 SPX 0DTE sample cited in
[07](07-strategy.md) D0 has skewness 4.47. Capping at 21 rather than 60: a
45-DTE position cannot round-trip inside the scored window, so it would be
graded purely on mark-to-market. See [07](07-strategy.md) D5.

## D6 — Determinism

Same `(proposal, portfolio, limits, as_of)` → same `Verdict`, byte for byte,
including the order of `checks` and `reasons`. Checks run in a fixed declared
order. No set iteration, no dict ordering assumptions, no `now()`.

## Test plan (RED first)

1. Each check in isolation: one passing fixture, one vetoing fixture, one exactly
   at the boundary. **A value exactly at its limit passes.** The exceptions are
   the two counters (`position_count`, `daily_trade_cap`), where being at the cap
   means the cap is used up, and `drawdown_killswitch`, which trips *at* the
   threshold rather than past it.
2. All checks run even when the first vetoes; `checks` length is constant.
3. `Approved` cannot be constructed outside `risk.gate` (boundary test).
4. `execution` refuses anything that is not a `GatedOrder` (boundary test).
5. Kill switch: trips at threshold, blocks OPEN, permits CLOSE, stays tripped.
6. Determinism: same inputs 100× → identical verdict (hypothesis).
7. `risk/` imports nothing outside stdlib + `alphagate.core` + `alphagate.options`
   (boundary test, modelled on the core's existing one).
