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

**The rule comes from an option book, and the book comes from the research.**

The strategy is not written here any more. It is
`iv_rank_low_sticky_put_credit_spread_v1` — proposed by the LLM search in
`ai_quant_researcher`, walk-forward validated, pre-registered, and measured once
against a sealed two-year window it had never seen. What this spec now defines
is the *frame*: which artefact carries a rule, what makes one executable, and
what the executor supplies that the research never modelled.

```
aqr option-research  →  aqr preregister  →  cli_sealed option-run  →  aqr option-book
                                                                          ↓
                                                          journal/option_books/*.json
                                                                          ↓
                                    alphagate.agent.option_book.load_option_book
```

**The fingerprint is pinned in configuration, with no default.**
`ALPHAGATE_OPTION_FINGERPRINT` in `.env.local`, exactly as
`ALPHAGATE_STRATEGY_FINGERPRINT` is pinned for the equity sleeve. A book names
the rule it describes; if that name were believed, replacing the file would
replace the strategy and nothing would notice. The pin is the only checkable
meaning of "it only trades the rule the research validated".

### The rule, as it stands

| | |
|---|---|
| entry | `iv_rank() < 15` |
| structure | put credit spread |
| expiry | 14 DTE, ±10 |
| short leg | 0.16 delta, ±0.06 |
| long leg | 0.08 delta |
| cadence | at most one entry per session |
| sizing | 2% of equity per structure, 3 concurrent |

**This inverts [D3](#d3--regime-gate) as it was originally written**, and the
inversion is the finding rather than a typo. The hand-written rule sold premium
only when `iv_rank >= 30`, on the reasoning in [D0](#d0--what-the-evidence-actually-says)
that the variance risk premium is largest when volatility is elevated. The
search tested that shape — `iv_rank_call_credit_spread_v3`,
`short_dated_put_skew_harvest_v1`, `iv_rank_extreme_put_credit_spread_v1` and
about a hundred others — and what survived out of sample was the opposite
condition: sell into *cheap* volatility, not rich. The proposed mechanism is
that demand for downside protection is sticky and slow to unwind in calm
regimes, so the premium stays rich relative to a subdued realised path.

We do not claim that mechanism is true. We claim it was pre-registered before
the sealed window was read, and that the window did not refute it. See
[D8](#d8--what-the-sealed-run-does-and-does-not-say).

### When there is no book

The agent runs nothing and says so. There is deliberately no hand-written
fallback rule: a fallback would mean the system trades *something* whatever
happens upstream, and "which rule was live on Tuesday" would stop being
answerable from the pin alone. A missing book, an unspent seal, a refuted rule
or a fingerprint mismatch each produce the same outcome — no orders, and a
journal record naming the refusal.

## D2 — Universe

**`SPY`. One underlying.**

It is the only instrument that satisfies all four requirements this strategy
places on an underlying at once.

**The market is tight enough to trade.** Median relative spread on ~16-delta
puts — the short-leg candidates — measured on live quotes:

| Underlying | Spread | Gate limit is 5% |
| --- | --- | --- |
| **SPY** | **0.53%** | passes |
| QQQ | 2.34% | passes |
| IWM | 4.74% | passes |
| TSLA, NVDA, MSFT | 2.5–4.0% | passes |
| AAPL | 6.23% | refused |
| DIA, XLF, XLE, XLI | 10–23% | refused |
| XLK, XBI, XLU | 22–27% | refused |
| XLV, XLY, XLC, XLB, XLRE | 35–200% | refused |

The sector ETFs are not small versions of SPY and are not available as breadth:
on the strikes this strategy sells, XLRE quotes a 200% relative spread and the
Gate refuses every candidate.

**It has no earnings.** An index trust has no quarter to report, so
`earnings.ETF_UNDERLYINGS` answers `earnings_within_dte` as a fact about the
instrument. Alpaca serves no earnings calendar, so for a single name that field
is `None` and [05](05-agent.md) D6 fail-closes on it — a single-name universe
trades nothing until a human fills in every report date by hand.

**The evidence in [D0](#d0--what-the-evidence-actually-says) is index evidence.**
The Cboe PUT index is SPX, the variance-risk-premium decay paper measures index
options, and the tastytrade 45-DTE convention was studied on SPY, IWM and SPX.
Nothing in that section supports a single-name credit-spread book. Single-name
premium is richer because it compensates earnings gaps and idiosyncratic jump
risk, which is a different trade.

**It is the only underlying with free history to backtest on.** The historical
source ([`OPTIONS_DATA.md`](../ai_quant_researcher/OPTIONS_DATA.md)) carries SPY
back to 2019 with bid, ask, implied volatility and greeks. It does not carry QQQ
or IWM, nor any cash-settled index option. Buying the gap costs about $40/month
and is not bought: seven years of bid/ask on the instrument actually traded is
worth more than three instruments with none.

### What one underlying costs

**Breadth comes from expiries and strikes, not from names.** Concurrent
positions differ by expiry and short strike on the same underlying, so they are
not independent bets — one adverse move in SPY moves all of them. The
[00](00-brief.md) target of roughly thirty fills is still reachable, and the
P&L it produces is a sample of one underlying's four days rather than of a
strategy across a market. [D7](#d7--reporting-pl-honestly) reports it as such.

**`underlying_concentration` stops being a diversification control.** With one
name in the universe, that check and `portfolio_heat` measure the same quantity,
so the tighter of the two is the only one that binds. [03](03-risk-gate.md) D6
sets them equal for this reason, and keeps the check rather than deleting it:
the universe is configuration, and the day a second name is added the check must
already be there and already correct.

## D3 — Regime gate

**The book's `entry` expression is the regime gate.** There is no threshold in
this document to tune, because a threshold here and a threshold in the book
would be two spellings of one number and they would drift.

Today that expression is `iv_rank() < 15`.

`iv_rank` is IV against its own trailing range, 0–100, computed in `options/`
and carried on `MarketRead`. Standing aside is not a missing feature: on a
session the expression does not fire the journal shows `NO_SETUP` with the
clause that failed, and that is the system working.

### The executor's reach is narrower than the researcher's, and that is a refusal

`agent/option_book.py` parses the entry against `MEASURABLE_FEATURES` — the
fields a `MarketRead` actually carries: `iv_rank`, `iv_percentile`, `iv_vs_hv`,
`hv_rank`, `atr_pct`. The researcher had a seven-year vendor volatility history
and could condition on `term_slope()`, `skew_25d()` and `atm_iv(dte)`. This
account cannot compute any of those.

**A book naming one is refused at load, not approximated.** Substituting a
near-enough number would execute a different rule under a fingerprint that
certifies this one, which is precisely the failure the pin exists to prevent.

### Undecidable is not false

`iv_rank` is `None` until the IV store holds `MIN_HISTORY` sessions
(`options/volatility.py`), and — per `agent/iv_store.py` — historical option
bars need a signed OPRA agreement this account does not have, so the history is
accumulated one session at a time.

When a feature the rule names is unmeasured, `EntryRule.decide` returns *false
with a reason that says so*, and the journal must show "could not decide"
rather than "the market was quiet". Those are different states and only one of
them is about the market. **Until the store is seeded, this rule trades
nothing**, and that is the honest outcome rather than a bug to route around.

## D4 — Direction

**From the book's `structure` field, not from the trend engine.**

Today that is `put_credit_spread` on every session the entry fires: the research
selected the structure along with the condition, and the two were measured
together. A rule validated as a put credit spread and executed as an iron condor
because a trend engine disagreed on a Tuesday is not the rule the sealed run
measured.

`load_option_book` accepts `put_credit_spread`, `call_credit_spread` and
`iron_condor` — the structures `agent/candidates.py` can build — and refuses
every other kind at load rather than producing an empty menu each cycle and
looking like a quiet market.

**What the trend engine still does.** It is no longer the direction input, but
`MarketRead` is still assembled in full and still journalled, so the record
shows what the deterministic engines saw on each entry. That is the
[adr/0001](../adr/0001-core-reuse.md) reuse dividend spent on evidence rather
than on the decision.

**What was lost, said plainly.** Short strikes are no longer placed beyond a
`level_engine` level. The research selected by delta because a delta-selected
wing resolves on 98% of sessions in its data and an exact points width on 23%
(specs/10 D5); the level constraint was never in the rule that was measured, so
adding it here would change the rule. `load_option_book` refuses `width_points`
for the same reason.

## D5 — Strike and expiry

**From the book. Nothing in this document sets them.**

- **Short leg** at the book's `anchor.delta` ± its tolerance — currently
  0.16 ± 0.06, the ~1σ convention.
- **Long leg** at the book's `width_delta` — currently 0.08. `load_option_book`
  refuses a book whose width delta is not *strictly* less than its anchor delta:
  a protective leg at or inside the short strike does not cap the loss, whatever
  the `structure` field claims, and that single check is the whole of "defined
  risk" at this boundary (CLAUDE.md rule 6).
- **Width still has to fit the budget.** A structure whose `max_loss` cannot be
  funded at quantity ≥ 1 is dropped before the model sees it
  ([05](05-agent.md) D4). The book chooses the wing; the account decides whether
  it is affordable, and an unaffordable one is journalled as such.
- **DTE from `rule.dte`** — currently 14 ± 10, so a 4–24 day window.
  `OptionRule.dte_window()` floors the low end at 1 whatever the book says, so a
  wide tolerance cannot silently re-admit the 0DTE that [03](03-risk-gate.md)
  excludes.

This supersedes the earlier "DTE 3–14, targeting ~7". That window was chosen so
positions could round-trip inside a four-day scored window; the researched rule
is held to expiry at ~14 days and will not round-trip inside it. See
[D7](#d7--reporting-pl-honestly).

## D6 — Management (deterministic; the model has no vote)

### The executed rule is the *managed* version of a rule measured unmanaged

The book says so itself, in `exit_convention`:

> Held to expiry. There is no stop, no profit target, no roll and no management
> rule, because the research data cannot price one: a specific contract is
> re-quoted on 1–3% of later sessions (specs/10 D0, D1).

We keep the exits below anyway, and the reason is not that the research was
wrong. The research could not *measure* an exit — 67% of contracts in that
vendor cache appear on exactly one session, so a mark-to-market does not exist
to stop against. Absence of measurement is not evidence that unmanaged is
better, and running a live paper account with no stop because a backtest could
not price one would be reasoning backwards from the data's limits to the
account's risk.

**The consequence is a claim we must not make.** Live P&L from this sleeve is
*not* comparable to the sealed measurement in [D8](#d8--what-the-sealed-run-does-and-does-not-say),
because it is a different exit policy on the same entries. The journal records
which exit rule fired on every close, so the difference is auditable rather than
asserted, and no demo artefact may present the live number as a realisation of
the sealed one.

The other thing the research did not model, and the executor owns: **early
assignment**. Settlement upstream is European-style against the underlying's
close; SPY options are American. `consumer_must_supply` in the book names this
explicitly.

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
The sealed run ([D8](#d8--what-the-sealed-run-does-and-does-not-say)) is what
carries any claim about the strategy; the live window is what demonstrates the
system.

**Round-trips will be rare.** At 14 DTE, positions opened this week mostly do
not reach expiry inside the scored window, so most of this sleeve's P&L will be
mark-to-market. That is a consequence of executing the researched rule rather
than the old 3–14 DTE hand-written one, it was chosen knowingly, and the report
separates the two numbers so it cannot be blurred.

## D8 — What the sealed run does and does not say

Spent once, on 2026-09-01, on `cc197008e0deb097`. Pre-registered before any
sealed session was read; the seal is now spent and there is no second run.

| | |
|---|---|
| window | 2024-09-03 → 2026-08-31, 500 sessions, 85 trades |
| strategy | return +8.14%, Sharpe 1.15, max drawdown −2.05% |
| benchmark | return +36.08%, Sharpe 1.02 |
| residual | alpha +2.52%/yr, beta 0.09, **t = +1.11**, IR 0.79 |
| significance bar | **t ≥ 1.96** (Bonferroni, 5% family-wise, 1 look) |

**The rule was not refuted. It was not confirmed either, and it cannot be.**
The window holds about 25 independent 28-DTE cycles, and specs/10 D8 is explicit
that a window that size is entitled to say a rule stopped working and never
entitled to say it works. t = 1.11 against a bar of 1.96 means the alpha is
positive and unmeasurable — which is not evidence.

`SealedOptionRun` carries `refuted` and `can_confirm` as two fields rather than
one verdict, precisely so no consumer can flatten "was not refuted" into
"passed". **No dashboard, README, demo script or submission artefact may word
this as validation.** What it licenses is exactly one sentence: the rule
survived a pre-registered attempt to refute it on data it had never seen.

## Test plan (RED first)

1. `iv_rank` arithmetic: at 0, at 100, with a flat trailing window, with gaps.
2. The entry expression decides correctly at, above and below its threshold, and
   an **unmeasured** feature declines with a reason distinguishable from a
   measured false — `tests/agent/test_option_book.py`.
3. A book naming a feature this account cannot measure is refused at load, and a
   book whose width delta is not strictly less than its anchor delta is refused
   as not defined risk.
4. The pinned fingerprint is the only one that loads; a book for another rule is
   refused by name.
5. An unspent seal, a refuted sealed run and a non-tradeable registry status are
   each refused, and a book wrong in four ways reports four faults at once.
6. The fixture payload matches the shape of the real artefact `aqr option-book`
   writes — the seam between the two projects is a file, and nothing else checks
   that it still fits.
7. Strike selection respects the book's delta target and reports the tolerance
   that bound it. The level constraint is gone with D4 and no test may assert it.
8. Width selection produces `max_loss` within budget, or no candidate at all.
9. Each management rule fires at its exact threshold and not one tick before,
   and the journal names which one closed each position — D6 makes live P&L
   non-comparable to D8, so the record has to carry the difference.
10. Management rules are unreachable from any model output (boundary test).
11. Earnings inside the window excludes an underlying from the universe, and an
   **unmeasured** earnings date excludes it too — the distinction that costs
   money is between `False` and `None` (`agent/earnings.py`), and D2 now rests
   on it entirely.
12. Every watchlist entry is in `earnings.ETF_UNDERLYINGS`. Asserted against
   that set rather than against the string `SPY`, so a name added without a
   report date fails the test rather than passing by not being checked.
13. `tradeable_today()` returning empty is a reported state, not an exception
    and not a silent no-op. Under D2 it is never the expected state, so an empty
    return must be reported rather than idled through.
14. An underlying whose measured spread exceeds the Gate's limit is refused by
    `liquidity`, not by the watchlist. D2 chooses the universe; the Gate is what
    enforces it, and a name slipped into the watchlist must be stopped by the
    check rather than by the list having been correct.
