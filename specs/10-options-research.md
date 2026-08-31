# 10 — Options research: exploring option strategies the way equities are explored

Where this lives: **`ai_quant_researcher/`**, alongside the equity search, under
[CLAUDE.md](../CLAUDE.md) §2b. It holds no money, places no orders and imports
no `alphagate`. The same three sentences that make the equity side safe make
this side safe, and for the same reasons.

What it is *for*: [07-strategy.md](07-strategy.md) asserts a strategy —
IV-rank-conditioned defined-risk credit spreads on SPY — on the strength of
published evidence. This spec builds the apparatus that can measure that claim
on data rather than cite it, and that can look for others.

```
option chain (EOD) ──┐
volatility history ──┼─→ option features ─→ OptionSpec ─→ options engine ─→ BacktestResult
SPY bars (1D, 1h) ───┘        (new)          (new)          (new)            (the SAME type)
                                                                                │
                          walk-forward · robustness · residual alpha · overfitting · evaluator
                                        ── all reused, unchanged ──
```

The one architectural commitment: **the options engine emits the same
`BacktestResult` the equity engines emit.** Everything downstream of a backtest
in `ai_quant_researcher/` is written against that type and against `Metrics` /
`ResidualAlpha`, not against an engine. That is the whole of "explore options
the way equities are explored", and it is a type-level statement rather than an
aspiration.

---

## D0 — What the data is, measured

Every decision below follows from this table. It was measured on the cache, not
read off a vendor page.

```
data-options/option_chain/SPY.csv            99,876 rows   753 sessions  2019-02-09 → 2024-08-30
data-options/volatility_history/SPY.csv         747 rows   732 usable    2019-02-09 → 2024-08-30
data-options-underlying/1D/SPY.csv            1,677 bars                 2018-01-02 → 2024-08-30
data-options-underlying/1h/SPY.csv           10,062 bars                 2018-01-02 → 2024-08-30
data-options-sealed/option_chain/SPY.csv    177,890 rows 1,260 sessions  2019-02-09 → 2026-08-28
```

**One snapshot per session. There is no time column.** `option_chain` is keyed
`(date, expiration, strike, call_put)`. Intraday option research is not
possible here and is not possible on any free source; it starts around $40 a
month elsewhere. A strategy that trades options several times a day cannot be
priced by this system, so this system does not offer the vocabulary to write
one.

**The session grid is irregular, and it is not daily.**

| Year | Sessions | Cadence |
| --- | --- | --- |
| 2019 | 48 | one Saturday snapshot a week |
| 2020–2023 | ~151/yr | Monday / Wednesday / Friday |
| 2024 (to 08-30) | mixed; Tuesdays and Thursdays begin appearing |
| 2025–2026 | 258 / 167 | daily — **but that is the sealed window** |

753 sessions across 5.55 years covers about 54% of trading days. The engine is
therefore event-driven over sessions and must never index by bar position: a
rule expressed in "bars" would mean a week in 2019 and a day in 2025.

**Three expirations per session, sampled at rolling ~14 / ~28 / ~49 DTE
targets.** DTE ranges 11–66, median 28. No 0DTE, no short weeklies, no LEAPS.
The expiries shift with the observation date rather than being fixed listed
dates.

**About 24 strikes per expiry, resampled around the money each session.**
Spacing runs 8–15 points — a sample of the listed ladder, not the ladder. The
delta ladder near the money is dense: a 16-delta put at 25–45 DTE is available
on **742 of 753** sessions.

**A specific contract is almost never re-quoted later.** Sampling entries at
~16-delta across three DTE buckets and counting quotes on subsequent sessions
before expiry:

| Entry bucket | Later sessions that quote the same contract |
| --- | --- |
| ~14 DTE | 1.3% |
| ~28 DTE | 3.3% |
| ~49 DTE | 3.1% |

This is the single most consequential fact in the file. **Mark-to-market is not
available.** Neither is a stop, a profit target, a signal exit, a roll, or any
management rule that needs to know what a position is worth before it expires.

**The spread is large and is measured, not modelled.** Relative bid-ask:
median 1.5%, p75 4.4%, p90 40%. 4.7% of rows have a zero bid.

**The underlying must be pulled raw, not adjusted.** Option strikes are set in
raw terms and do not move for an ordinary dividend, so a dividend-adjusted close
compared against a strike reports a moneyness the trade never had. Alpaca's
default is `adjustment=all`, which is correct for equity research and wrong
here: SPY's real close on 2019-11-22 was 311.02 and the adjusted series says
282.10. `data-options-underlying/` is pulled with `--adjustment raw`, the choice
is recorded in the dataset version (`alpaca:sip:raw:...`), and put-call parity
checks the two caches against each other — see D2a.

**A chain session is not always a trading day.** 82 of 753 sessions have no
same-day SPY bar: 47 are the 2019 Saturday snapshots, 1 a Sunday, and 34 are
market holidays. Every one maps back to a trading day within 4 calendar days, so
the underlying reference for a session is **the close of the most recent trading
day at or before it, refused beyond 4 days**. Every settlement date, by
contrast, has a bar — 742 of 742 — so no position is ever left unable to close.

---

## D1 — The unit of research is a structure held to expiry

Entry is priced from the chain. Exit is **settlement against the underlying's
close on the expiration date**, computed from `data-options-underlying/1D`.

This is not a simplification chosen for speed. It is the only exit the data can
price, per D0. The alternative — re-deriving a mid from that session's
surface — replaces a missing number with a modelled one and then reports the
model's opinion as a backtest result. A system whose entire justification is
that it refuses to lie about out-of-sample evidence does not get to do that in
its exit logic.

The consequence is stated rather than hidden: **this system cannot evaluate
managed strategies.** "Manage at 50% of max profit", the single most common
retail rule and the one [07](07-strategy.md) D6 specifies, is unmeasurable
here. A verdict from this apparatus is a verdict on the *unmanaged* version of
the rule, and every artefact it writes says so.

**Early assignment is not modelled.** SPY options are American and the ETF pays
dividends, so an ITM short leg can be assigned before expiry, most likely
around an ex-dividend date. Settlement here is European. The bias is toward
optimism on short ITM legs and is recorded with every result.

### D1a — The money is marked to market; the curve is marked to model

These are two different questions and they get two different answers.

**Realised P&L uses quoted prices only.** Entry at the session's bid/ask,
settlement at the underlying's close. No model touches a number that appears in
a trade's P&L, and every trade's result could be reconciled against a broker
statement.

**The equity curve between entry and expiry is a Black-Scholes valuation**, on
the daily trading-day grid, holding the contract's entry IV constant and
advancing only spot and time.

Marking is not optional, and cash accounting is not the conservative choice it
looks like. A book that only moves on cash flows has, by construction, a beta of
zero and no drawdown between entry and expiry. Feeding that into
[`alpha.py`](../ai_quant_researcher/src/aqr/backtest/alpha.py) would attribute
the entire return of a short put spread to alpha — the exact failure that module
was written to prevent, arriving through the back door. A short premium book
*has* beta, it *has* drawdown, and a curve that cannot show either is not
conservative. It is wrong in the direction that flatters the strategy.

The assumption is named rather than buried: **IV is frozen at entry**, so the
curve carries the position's delta and gamma and does not carry its vega. A
volatility spike shows up in the mark as the spot move only. That understates
the drawdown of a short-premium book in exactly the episodes that matter, and
every result says so.

The split is the whole of it: **a model may never move money, and cash
accounting may never be mistaken for risk.**

---

## D2 — The fill convention, and the one rule that makes it honest

> **Decide from information available at session `t−1`. Select and fill the
> contract from session `t`'s chain, buying at its ask and selling at its bid.**

This is [the equity engine's rule](../ai_quant_researcher/src/aqr/backtest/engine.py)
translated, and it has the same property: nothing at or after the fill can
influence the decision, because features are computed over the whole series up
front and read only at index `t−1`.

There is no configuration flag that relaxes it. A look-ahead switch is a
look-ahead bug with a rationale attached.

**Crossing the spread is charged in full, always.** Buy the ask, sell the bid,
per leg, with no mid-price option and no partial-cross parameter. The quotes
are in the data; a fill assumption more favourable than the quoted book would
be inventing liquidity that the p90 spread of 40% says is not there. Commission
is charged per contract per leg on top (D7).

**A row with a zero bid cannot be sold.** 4.7% of rows have one. Selling into a
zero bid is a fill at zero, which is not a trade; it is an entry the engine
must refuse.

### D2a — The chain states its own spot, and it is checked against the bars

`C − P = S − K` at zero rates, so any expiry that carries both a call and a put
at one strike implies an underlying price. Measured on the research cache, that
implied spot agrees with the raw bar close to a median of **0.14%**, p99 0.61%,
never worse than 2%. Under the dividend-adjusted series it was out by ten
percent.

This is asserted by a test rather than trusted, because the failure it catches
already happened here and it is silent by construction: the backtest ran to
completion, produced a full set of plausible numbers, and reported a 58% win
rate on a 16-delta short put spread that should win about 84% of the time.
Nothing raised, because nothing was wrong with the arithmetic — only with which
two numbers were being compared. A cross-check between two independently pulled
caches is the only thing that could have noticed.

---

## D3 — An entry whose expiry crosses the embargo is refused

Settlement reads the underlying's close on the expiration date. A structure
entered on 2024-08-30 at 28 DTE settles on 2024-09-27 — inside the reserved
window. Filling it would read embargoed prices to produce a research result,
which is precisely the failure [`aqr.seal`](../ai_quant_researcher/src/aqr/seal.py)
exists to make impossible.

**Invariant: the engine refuses an entry whose `expiration >= EMBARGO_START`**
(2024-09-01 in the research phase; the sealed root's own end in the sealed
phase). Not a warning, not a truncation applied afterwards — a refusal at
selection time, tested.

The cost is small and measured: **9 of 753 sessions** lose their ~28 DTE entry
this way. The research window's effective end is one max-DTE before the embargo
boundary, and that is correct rather than unfortunate.

---

## D4 — Defined risk in the type, not in review

The proposable structures, and nothing else:

```
long_call · long_put                     debit, risk = premium paid
put_credit_spread · call_credit_spread   credit, risk = width − credit
put_debit_spread · call_debit_spread     debit, risk = premium paid
iron_condor                              credit, risk = wider wing − credit
```

There is no `custom`, no single short leg, and no kind whose maximum loss is
unbounded. Same mechanism as [02-options-domain.md](02-options-domain.md) D3 on
the execution side, reached independently: **if it cannot be constructed, it
cannot be proposed, and it cannot be measured into existence.**

`covered_call` and `cash_secured_put` are deliberately absent. Both need a stock
position or a cash reserve to be defined risk, which makes them a portfolio
statement rather than a structure, and the accounting for that is not in this
phase.

Maximum loss must be computable at construction from the fill prices alone.
Sizing is a fraction of equity risked against *that* number — never a notional,
never a delta budget.

---

## D5 — What the DSL may say, and what it deliberately cannot

```yaml
name: iv_rank_put_credit_spread_v1
hypothesis: when SPY's IV rank is high, a 28-day 16-delta put spread is paid a positive risk premium
underlying: SPY

entry: iv_rank(252) > 50 and close > sma(200)

structure:
  type: put_credit_spread
  dte: {target: 28, tolerance: 10}
  anchor: {delta: 0.16, tolerance: 0.06}   # the leg the rule names
  width_delta: 0.06                       # the protective leg, by its own delta

sizing:
  type: fixed_risk
  risk_per_trade: 0.01                     # of equity, against max loss
  max_concurrent: 3

cadence:
  min_sessions_between_entries: 5
```

`entry` is parsed by the existing
[`dsl/expr.py`](../ai_quant_researcher/src/aqr/dsl/expr.py) — same tokenizer,
same whitelist, same `feature_keys` walk, same content-addressed fingerprint.
Only the feature table changes.

**There is no `exit:` block, and its absence is the point.** No stop, no target,
no `signal_exit`, no roll, no `max_holding_bars`. D0 says the data cannot price
any of them. A field that exists but whose semantics are fabricated is worse
than a missing field, because the LLM will fill it in and every downstream
number will silently be about a strategy nobody can trade.

`OptionSpec` is a **new type in `options/spec.py`**, not new fields on
`StrategySpec`. The equity spec's `stop_loss`, `take_profit` and `max_positions`
are meaningless here and its `mode: portfolio` has no cross-section to rank;
widening one type to cover both would make every validator branch on which half
of itself was in use.

**The width is named by delta, not by points.** Measured against a 16-delta
short put on the research window: a delta-selected wing resolves on **98%** of
sessions, an exact 10-point wing on **23%**. The listed widths below a 16-delta
strike are 8, 9, 10, 18, 25, 35 or 45 points depending on the session, because
the cache samples about 24 rungs from a ladder that lists hundreds. "Ten points
wide" is not a rule this data can express, and a search told to use one would be
selecting on the vendor's sampling rather than on the market. `width_points`
remains available for a rule that means a literal distance, and it refuses
rather than widening quietly.

**Selection is deterministic and total.** `anchor: {delta: 0.16}`
must resolve to exactly one contract on a given session or refuse — ties broken
by the lower strike, then by the earlier expiry, then not at all. Two runs of
the same spec on the same cache produce the same trades, byte for byte, or the
engine is broken.

---

## D6 — The feature vocabulary

| Feature | Source | Note |
| --- | --- | --- |
| `iv_rank(n)` | volatility_history | `(iv_current − iv_year_low) / (iv_year_high − iv_year_low)`; the vendor supplies the year extremes |
| `iv_hv_spread()` | volatility_history | `iv_current − hv_current` — the variance risk premium, directly |
| `iv_change(n)` | volatility_history | `week_ago` / `month_ago` columns are in the table |
| `atm_iv(dte)` | option_chain | interpolated within a DTE bucket |
| `term_slope()` | option_chain | ~49 DTE ATM IV − ~14 DTE ATM IV |
| `skew_25d()` | option_chain | 25-delta put IV − 25-delta call IV |
| `close`, `sma`, `ema`, `rsi`, `atr`, `realized_vol`, … | SPY bars | **the existing registry, unchanged** |

Intraday features from `data-options-underlying/1h` are derived into
session-level values (prior-day range, close-to-close gap, last-hour momentum)
and read at `t−1` like everything else. They raise the information the decision
uses; they do not raise the trading frequency, which D0 caps at one entry per
session.

**Forward-fill is explicit and bounded.** The two grids do not align: 22 chain
sessions have no same-day IV row, and 15 vendor rows have blank extremes. An IV
value is carried forward at most **5 calendar days** and the feature is `NaN`
beyond that. Without the bound, the 2019 weekly era would quietly evaluate
rules on IV that is a fortnight old.

---

## D7 — Costs: the spread is the cost, and it is already in the data

Charged per leg, per contract:

1. **The full spread**, per D2. This dominates. A 10-point put credit spread
   collecting $1.50 crosses two spreads to open; at the median 1.5% relative
   spread on each leg that is real money, and at the p90 it is the trade.
2. **Commission**, from a named schedule in `backtest/costs.py`, alongside the
   existing equity schedules: `ALPACA_OPTIONS` ($0/contract plus regulatory
   fees) and `IBKR_OPTIONS` ($0.65/contract). The schedule travels with the
   verdict, as the equity schedules already do.
3. **Assignment and exercise fees** at settlement when a leg finishes ITM.

Cost retention is a fatal gate on the equity side and stays one here.

---

## D8 — What "out of sample" means when there are 71 cycles

This is the section that keeps the apparatus honest, and it is the reason the
options search must be smaller than the equity search rather than larger.

**Independent cycles in the research window, measured:**

| Program | Non-overlapping cycles in 5.55 years |
| --- | --- |
| ~14 DTE | 144 |
| ~28 DTE | **71** |
| ~49 DTE | 33 |

And conditioning matters enormously. SPY's IV rank over these sessions has
median 18.5 (p75 39.3, p90 67.2) and exceeds 50 on **17.8%** of them. So
[07](07-strategy.md)'s `iv_rank > 50` filter on a 28 DTE program leaves on the
order of **twelve independent bets in five and a half years.** No amount of
engine correctness makes twelve bets into evidence.

Four consequences, all mechanical:

**Trades are not the sample; non-overlapping cycles are.** Overlapping entries
produce correlated returns. `MIN_TRADES = 30` in the evaluator is close to
meaningless here — thirty overlapping trades can be eight independent bets. The
options evaluator gates on **independent cycles ≥ 25**, and every result reports
the count next to the trade count.

**`asset_robustness` is undefined on one underlying and must not be faked.** It
is replaced, not zero-filled, by two things the data can support: leave-one-year-out
(6 folds) and **DTE-bucket agreement** — does the rule keep its sign across the
~14 / ~28 / ~49 buckets? A rule that only works at one DTE target found a DTE
target, not a premium.

**The search budget is capped at 20 hypotheses**, and the counter is kept
separately from the equity side's. The equity search spent 414 hypotheses and
deflated its Sharpe to 0.74 for it; running 400 trials against 71 cycles would
produce a number with no information in it. The multiplicity bar must not be
computed against a denominator that mixes the two searches.

**The sealed window can refute and nothing more.** 2024-09-01 → 2026-08-28 is
about 25 independent 28-DTE cycles. It is entitled to say "this stopped
working". It is not entitled to say "this works", and no artefact this system
writes may word it that way.

---

## D9 — What is reused, unchanged

Reused as-is, because it is written against types rather than engines:

`seal.py` and the whole embargo apparatus · pre-registration and the one-shot
sealed run · `registry/db.py` · `backtest/metrics.py` · `backtest/alpha.py`
(residual alpha against SPY buy-and-hold) · `validation/walkforward.py` ·
`validation/overfitting.py` · `dsl/expr.py` · the existing bar feature registry
· the provider and repair loop in `agent/`.

New, under `src/aqr/options/`:

`chain.py` (the indexed, queryable chain) · `pricing.py` (Black-Scholes, for
the curve and nothing else — D1a) · `features.py` · `spec.py` · `engine.py` ·
an `ALPACA_OPTIONS` / `IBKR_OPTIONS` schedule in the existing `costs.py` · an
option proposer and prompt in `agent/`.

Changed:

`evaluator/score.py` gains an options profile per D8 — independent-cycle gate,
`asset_robustness` replaced by year and DTE-bucket robustness.

`options/` joins `PURE_LAYERS` in
[`tests/test_boundaries.py`](../ai_quant_researcher/tests/test_boundaries.py):
no network, no filesystem, no clock, no database. The loaders stay in `data/`,
where `OptionChain.__post_init__` already reports every session it holds to the
seal.

---

## D10 — Not in this spec

**The handoff to AlphaGate.** An option book is a list of legs, not a vector of
weights, so `target_book.py` does not describe one and must not be stretched to.
When it is built it is a separate artefact with its own schema, and the seam
stays a file in both directions ([09](09-equity-execution.md) D0).

**Intraday options.** D0. Not a scope decision — a data one.

**More than one underlying.** The chain cache holds SPY alone and
`data-options/_raw/` has been cleared, so adding underlyings costs a multi-hour
re-download. It is the one change that would materially raise statistical power
— a cross-section of IV rank across ~600 names turns 71 cycles into 71 × N — and
it is the right first move after the hackathon, not during it.

**Managed exits.** D1. Requires data this project does not have.

---

## Test plan (RED first)

`options/` and the engine are the correctness surface. Tests come first there,
in this order:

**Chain indexing**
1. `select_by: delta` resolves to exactly one contract, deterministically, with
   ties broken by the stated order.
2. A requested delta or DTE outside tolerance **raises**; it never returns the
   nearest thing available.
3. A zero-bid row cannot be selected for a short leg.

**No look-ahead**
4. Truncating the chain after session `t` does not change any decision made at
   or before `t`.
5. The features a spec touches are all readable at `t−1`; a feature that needs
   session `t` fails the build.

**The embargo**
6. An entry whose `expiration >= EMBARGO_START` is refused, in the research
   phase, with the expiry named in the error.
7. Running the whole engine over the research root leaves the seal clean, and
   the load ledger contains no session past the embargo.

**The underlying reference**
8. A session with no same-day bar resolves to the most recent trading day's
   close; one more than 4 days stale refuses. 82 of 753 sessions take this
   path and none of them fails.
8a. The chain's parity-implied spot and the bar close agree to within 2% on
   every session (D2a). This is the test that catches a cache pulled with the
   wrong price adjustment, which nothing else can see.

**Structure and money**
9. Every `StructureKind` computes a finite maximum loss at construction; a
   naked short leg does not construct.
10. Entry cost equals the full crossed spread plus commission, per leg, checked
    against a hand-computed fixture.
11. Settlement arithmetic, per kind, against hand-computed expiry payoffs
    including exactly-at-strike and both-legs-ITM.

**The mark, and the wall around it (D1a)**
12. No trade's realised P&L changes when the pricing model changes. Asserted by
    running a backtest twice with different volatility inputs to the mark and
    comparing the trade list, which must be identical.
13. A short put spread's marked curve carries a negative beta to the underlying
    and a drawdown greater than zero — the properties cash accounting destroys.

**Determinism**
14. Two runs of the same spec on the same cache produce identical trades and an
    identical equity curve.

**The seam that matters**
15. The engine's result satisfies everything `Metrics`, `ResidualAlpha`,
    `run_walkforward` and `evaluate_strategy` require — asserted by running the
    real downstream functions on a real option backtest, not by checking the
    type.

**The measured claims in this spec**
16. D0's table is regenerated from the cache by a test, so a re-pull that
    changes the data fails the build instead of silently invalidating the
    reasoning above.
