# 07 — Strategy

## D0 — What the evidence actually says

This spec starts with the negative result, because it determines everything else.

**There is no publicly documented options strategy with reliable alpha.** The
search was run; this is the finding, not a failure to find. What the last two
years of evidence shows:

**The volatility risk premium — the best-documented premium in options — has
decayed to zero alpha.** Dew-Becker and Giglio, *The Decline of the Variance
Risk Premium* (Chicago Fed WP 2025-17, September 2025), document a structural
break around 2010. Since then, the CAPM alphas of traded index options have
"converged to zero", and since the traded-vs-synthetic gap is literally a
delta-hedged return, **the alpha of delta-hedged options has gone to zero**. The
premium that generations of option sellers harvested is, on current evidence, a
premium no longer.

**Put-writing still performs, but as beta, not alpha.** Cboe's PUT index
returned 17.8% in 2024 against 25.0% for SPX total return, with beta 0.56 and a
15.06% max drawdown (April 2025, 109 sessions to recover). That is equity
exposure with a capped upside, not an edge.

**The famous retail rulesets are weaker than their reputation.** Independent
backtests of the tastytrade 45-DTE / 16-delta / manage-at-50% short strangle
report roughly 5–9% *total* over an 11-year period.

**0DTE is a lottery with a house edge on the other side.** In a 2025 sample of
199 SPX 0DTE straddles held from 9:35am to settlement, mean P&L was +$1.45 and
median was −$3.28, with skewness 4.47 and kurtosis 33.58. Selling that
distribution has a positive mean and a catastrophic left tail. Over a four-day
scored window it is indistinguishable from a coin flip, which is why
[03](03-risk-gate.md) excludes it.

### What follows

We do not claim an edge, and the submission must not either. Every other entrant
will assert their agent found alpha; the evidence says they did not. Our claim is
narrower and defensible:

> We harvest a well-documented, structurally-explained premium whose alpha has
> decayed, and the system's contribution is risk control and execution
> discipline rather than prediction. Here is the paper that says the premium
> decayed, and here is what we did about it.

That is a better story than a fabricated edge, and it survives a judge who knows
the literature.

## D1 — The strategy

**IV-rank-conditioned defined-risk credit spreads, mechanically managed.**

Premium selling, but only when premium is actually rich, with direction taken
from the deterministic trend engine rather than from a forecast, and with every
position's loss capped at construction.

## D2 — Universe

Liquid optionable underlyings only — the strategy's real enemy over four days is
the bid/ask spread, not the market:

- Index ETFs: `SPY`, `QQQ`, `IWM`
- Large caps with dense chains: three to five, chosen at kickoff for open
  interest, **excluding any with earnings inside the window**

Breadth exists to produce enough independent fills to escape single-trade noise
([00](00-brief.md)), not to diversify a return stream.

## D3 — Regime gate

`iv_rank` is IV against its own trailing 6-month range, computed in `options/`.

| Regime | Action |
| --- | --- |
| `iv_rank >= 30` | Sell a defined-risk credit spread |
| `iv_rank < 30` | **Stand aside.** No selling into cheap premium. |

The threshold encodes the one part of the VRP literature that survived: the
premium is largest when volatility is elevated. When it is not elevated, there
is nothing to harvest and the correct action is to do nothing — which
[05](05-agent.md) D6 already makes a first-class outcome.

Standing aside is not a missing feature. On a low-IV day the journal will show
`NO_SETUP` across the whole universe, and that is the system working.

## D4 — Direction

From `MarketRead`, deterministically — this is the [adr/0001](../adr/0001-core-reuse.md)
reuse dividend, and it is the input the model does not have to guess at:

| Trend + confluence | Structure |
| --- | --- |
| Up, timeframes agree | **Put credit spread** below support |
| Down, timeframes agree | **Call credit spread** above resistance |
| Timeframes disagree, or trend is `RANGE` | **Iron condor**, both wings outside the range |

Short strikes are placed beyond a `level_engine` level, not merely at a delta.
Selling a strike that sits *behind* a tested support is the reuse paying for
itself; a pure-delta screen cannot see the level.

## D5 — Strike and expiry

- **Short leg ≈ 0.16 delta** (the ~1σ convention), adjusted to the nearest strike
  that is also beyond the relevant level (D4).
- **Long leg** one to two strikes further out. Width sized so `max_loss` fits the
  per-trade budget at quantity ≥ 1; a candidate that cannot is dropped before the
  model sees it ([05](05-agent.md) D4).
- **DTE 3–14, targeting ~7.** This revises [03](03-risk-gate.md) D5, which had
  `(7, 60)`. Rationale: over a scored window of roughly four sessions, a 45-DTE
  position never realises anything — it is scored purely on mark-to-market. Short
  weeklies let positions actually round-trip. 0DTE and 1DTE remain excluded.

## D6 — Management (deterministic; the model has no vote)

Evaluated every cycle, per [05](05-agent.md) D8:

| Rule | Action | Built |
| --- | --- | --- |
| Mark ≤ 50% of credit received | **Close.** Take the profit. | `ExitRule.PROFIT_TARGET` |
| Mark ≥ 200% of credit received | **Close.** Cut the loss. | `ExitRule.STOP` |
| DTE ≤ 2 | **Close.** Never hold into expiry gamma. | `ExitRule.DTE_CLOSE` |
| Underlying breaches the short strike's level | **Close.** The thesis is void. | **not implemented** |

The thresholds live in `ExitPolicy` — one place, logged at startup, shown on the
dashboard's Live tab, which renders each position travelling between its stop
and its target.

The fourth rule is specified and absent. It needs a level read per open
position, which is the same machinery D5's strike selection needs and is
deferred with it; the first three are enough to bound holding time, which is
what the rules are for. A position whose short strike has been breached is
still closed by the stop, just later and at a worse price. Nothing silently
substitutes for it: `ExitRule` has no member for it, so it cannot fire and be
mis-attributed.

The 50/200 pair is the tastytrade convention. Its independent evidence is
mediocre (D0), and it is used here for a different reason than edge: it bounds
holding time, which is what produces round-trips inside a four-day window.

## D7 — Reporting P&L honestly

At submission, realised and unrealised are reported **separately**:

- **Realised P&L** — closed round-trips. The number that means something.
- **Mark-to-market** — positions still open at 2026-09-04 15:00 UTC.
- **Fill count**, and the fraction of cycles that produced no trade.
- **Max drawdown** against the 5% limit from [03](03-risk-gate.md) D5.

A four-day return is a sample, not a track record, and the submission says so.
The backtest (spec 08, not yet written) is what carries any claim about the strategy;
the live window is what demonstrates the system.

## Test plan (RED first)

1. `iv_rank` arithmetic: at 0, at 100, with a flat trailing window, with gaps.
2. Regime gate: below, at, and above 30 — document which side is inclusive.
3. Direction mapping, one fixture per trend/confluence combination, including
   disagreement → condor.
4. Strike selection respects both the delta target and the level constraint, and
   reports which one bound it.
5. Width selection produces `max_loss` within budget, or no candidate at all.
6. Each management rule fires at its exact threshold and not one tick before.
7. Management rules are unreachable from any model output (boundary test).
8. Earnings inside the window excludes an underlying from the universe.
