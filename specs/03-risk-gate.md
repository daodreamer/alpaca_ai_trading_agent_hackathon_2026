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

## D6 — Sleeves: two strategies, one account

This system runs two strategies against one Alpaca account. The equity book
([09](09-equity-execution.md)) holds a ranked, fully-invested portfolio; the
options agent ([07](07-strategy.md)) sells defined-risk credit spreads. They
answer different questions and hold different invariants, and until this section
existed they **budgeted against the same number**.

A limit scaled off `account.equity` is not a risk control here, and a kill
switch watching `account.equity` is not either. Two failures follow from it
directly:

**The equity book latches the options kill switch.** D5 sets the options
drawdown limit at 5%. A $95,000 stock sleeve falls 5% of the account on an 8%
market move, so the options agent — having lost nothing and possibly holding
nothing — trips and stays tripped until a human clears it. The switch fires for
a reason that has nothing to do with the risk it watches.

**Each strategy's budgets move with the other's mark-to-market.** How much
premium the options agent may sell would depend on what stocks did overnight.
That is one strategy's noise arriving as another strategy's constraint.

### The rule

> Every budgeted limit is a fraction of the **sleeve** the strategy was
> allocated, and every kill switch measures the drawdown of **its own sleeve**.
> The account does not appear in either arithmetic.

`risk.sleeve.Sleeve` is the whole mechanism, and it is deliberately almost
nothing:

```
sleeve.equity = max(0, allocation + realised + unrealised)
```

`allocation` is **fixed in configuration, not a fraction of live equity**. A
fraction would reintroduce exactly the coupling this removes — the other
strategy's overnight mark would resize this strategy's budgets. The operator
splits the account once and the split does not move because the market did.

`realised` is summed from the journal's `outcome.realised_pl` amendments, the
same field the dashboard renders, so the number the kill switch measures and the
number a reader sees have one source. `unrealised` is the broker's own mark on
this sleeve's open positions. They are carried separately rather than summed on
the way in, because [07](07-strategy.md) D7 reports them separately and a type
that has already added them cannot.

The zero floor is not decoration. Every limit is a *fraction* of this number, so
a negative equity would flip the sign of each one and permit more the further
underwater the sleeve got. Defined risk makes that unreachable today; a limit
that depends on a strategy invariant to stay sane is one that breaks the day the
strategy changes.

### The competition split

| Sleeve | Allocation | Limits | Measured |
| --- | --- | --- | --- |
| Options agent | $5,000 | `SLEEVE_LIMITS` | bottom-up: allocation + its own P&L |
| Equity book | $95,000 | `equity/policy.py` | residual: account − the options sleeve |

**One is measured bottom-up and the other as a residual, and the asymmetry is
deliberate.** The options sleeve's P&L is separately identifiable — its
positions are option contracts, its round-trips carry `realised_pl` — and
[07](07-strategy.md) D7 requires realised and mark-to-market reported apart. The
equity book is then everything the account holds beyond it. That makes the split
an identity rather than an estimate:

```
account_equity = equity_sleeve.equity + options_sleeve.equity
```

which is what makes the isolation exact in both directions. An options loss of
$1,000 lowers the account to $99,000 *and* the options sleeve to $4,000, so the
residual is still $95,000 — the equity book does not absorb a drawdown it did
not take. An $8,000 fall in the stock book leaves the options sleeve at exactly
$5,000.

### The equity book's scale

Its targets are 95% of the account, and moving to that scale **forces no
rebalance**: the no-trade band is 20% of the position with a $25 floor
([09](09-equity-execution.md) D3), and a 5% shift clears neither on any of the
87 names in the current book. The positions converge at the next scheduled
rebalance, when the weights are recomputed anyway.

**The 39% cash the book holds is not spare capital.** Its gross is 0.61, so the
reserve is the strategy's own, for rebalancing, and it is a fraction of whatever
base it is given. Taking the options sleeve out of the *base* rather than out of
the reserve is what keeps that reserve proportionally intact; the alternative
spends cash the strategy was relying on.

The cost, stated rather than hidden: **the equity book runs at 95% of the scale
`ai_quant_researcher` validated.** For a weight-based book that is linear —
returns scale by 0.95, the character does not change — but it is a deviation
from the backtest and the submission says so.

### Migrating the high-water marks

A peak measured against the account and a peak measured against a sleeve are not
a bigger and a smaller number, they are different measurements. `SessionState`
therefore persists a `basis`, and a state file read under a different one has
its peak **discarded and the discard printed** — not rescaled. Rescaling means
subtracting what the other sleeve was worth at that moment, which the file does
not record; it happens to equal the allocation today only because the options
agent has not traded, and a migration correct only while some fact holds is one
that goes wrong silently later. Dropping the mark can only understate a
drawdown, and only until the sleeve makes a new high.

`SLEEVE_LIMITS` is best read as a diff against D5's `DEFAULT_LIMITS`. Two
absolute figures are held where they were and three are deliberately moved:

| Limit | Was (5% of $100k account) | Now (fraction of $5k sleeve) | Why |
| --- | --- | --- | --- |
| per trade | $1,000 | $1,000 | held — `agent/candidates.py` was tuned live at this size |
| book heat | $5,000 | $4,000 | for defined risk, heat *is* capital deployed; 0.80 leaves the last trade to be refused by a budget rather than by a broker |
| per underlying | $2,000 | $4,000 | equal to heat: [07](07-strategy.md) D2 gives this strategy one underlying, so anything tighter is a second heat cap wearing a concentration limit's name, and it would cap the sleeve at a quarter of its allocation while `max_portfolio_loss_pct` read as the binding number. Set equal rather than removed, so the check is already correct the day a second name is added |
| net delta | ±30 | ±6 | ±30 delta is $19,500 of SPY notional against $5,000 of capital |
| kill switch | $5,000, of the account | $1,000, of the sleeve | 5% of the sleeve is $250, which one spread reaching its stop produces — that is a trade going wrong, not a strategy failing |

**±6 is the number most likely to need a live adjustment.** It is the only limit
here calibrated by arithmetic rather than by an observed run, and the failure it
produces is quiet: candidates outside the band are dropped before the model sees
them (`agent/candidates.py`), so a band set too tight looks like a market with no
setups rather than like a misconfiguration. `preflight` prints the whole sleeve
in absolute dollars for that reason — a $50 per-trade budget against a $100
spread never trades, and from the outside that is indistinguishable from a quiet
afternoon.

### What a sleeve is not

**It is not a broker sub-account.** Alpaca holds one pool of buying power and it
has never heard of this file. A sleeve constrains what each strategy will *ask*
for; it cannot stop the other strategy from having already spent the cash. Two
consequences follow and both are the operator's to hold:

* the allocations must sum to no more than the account;
* a strategy can still be refused by the broker for buying power it was, by its
  own sleeve, entitled to. That is a rejection with a journalled reason, which
  is the failure mode we want, but it is not prevented here.

## D7 — Determinism

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
6b. Sleeve (D6): equity is allocation plus its own P&L and floors at zero; a
   drawdown is measured against the sleeve's high-water mark, not the account's;
   an account fall driven entirely by the equity book leaves the options
   drawdown at zero; and `Sleeve` takes no account argument — asserted
   structurally, because an edit that added one would restore the coupling
   without failing any arithmetic test.
7. `risk/` imports nothing outside stdlib + `alphagate.core` + `alphagate.options`
   (boundary test, modelled on the core's existing one).
