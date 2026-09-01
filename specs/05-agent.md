# 05 — The agent

The orchestration layer. The only package permitted to import an LLM SDK
([01](01-architecture.md) Rule 1).

## D1 — The cycle

One pass, run on a schedule during regular trading hours:

```
 1. perceive      core engines over fresh bars        → MarketRead      pure
 2. screen        deterministic pre-filter            → Setup | None    pure
 3. enumerate     build candidate structures          → Candidates      pure
 4. propose       the LLM picks one, or picks none    → Choice          impure
 5. size          risk budget → quantity              → TradeProposal   pure
 6. gate          risk.evaluate                       → Verdict         pure
 7. submit        execution.submit, if Approved       → Submission      impure
 8. record        every step, always                  → CycleRecord     impure
```

Steps 4 and 7 are the only impure ones. Everything else is a pure function of
its inputs, which is what makes step 8 a *replayable* record rather than a log.

**The cycle always reaches step 8.** A screen that finds nothing, a model that
declines, a Gate that vetoes — each writes a record. A journal that only contains
trades cannot answer "why didn't it trade?", which is the question a judge asks.

## D2 — What the model is shown

Facts from the deterministic engines, never raw prices to interpret and never
an image:

```python
@dataclass(frozen=True, slots=True)
class MarketRead:
    underlying: Ticker
    as_of: datetime
    trend: TrendState              # core.trend_engine — multi-timeframe
    confluence: ConfluenceReport   # core.confluence — do the timeframes agree
    levels: tuple[Level, ...]      # core.level_engine — nearby support/resistance
    atr_pct: Decimal
    iv_rank: Decimal               # current IV against its own trailing range
    iv_vs_hv: Decimal              # implied over realised
    earnings_within_dte: bool
```

`iv_rank` is the headline input: it is what separates "sell premium" from "buy
premium", and it is computed, not asked about.

**Every field except `spot` is optional, and `None` means unmeasured.** Not
neutral, not average, not zero. This is not defensive typing; it is the fix for
a real failure. The first live cycle passed a mean implied-volatility *level*
(15.79) in a field named `iv_rank`; the model read "IV rank is low (15.79)" and
reasoned faultlessly to a conclusion the input did not support. A rank is a
position inside a trailing range and a level is not, and the same 15.79% is rank
100 against a calm regime and rank 0 against a wild one.

Three consequences, all enforced:

- `MarketRead` **refuses** a value outside 0–100 in a rank field, so a level
  cannot be smuggled into one again.
- The prompt renders any unmeasured field as the literal string `unmeasured`,
  and names its units in its key (`iv_rank_0_100`, `iv_vs_hv_ratio`).
- The proposer is told to prefer declining when something it needs is
  unmeasured — and does. See D6.

### What is actually available

`iv_rank` needs a trailing implied-volatility series, and Alpaca serves none:
historical option bars and trades are gated behind an OPRA agreement that comes
with the paid data plan. On the Basic plan this account holds, verified by
probe:

| Endpoint | Result |
| --- | --- |
| options snapshots, `feed=indicative` | 200 |
| options snapshots, `feed=opra` | 403 `OPRA agreement is not signed` |
| options historical bars / trades | 403 `OPRA agreement is not signed` |
| stock bars, `feed=iex` | 200 |
| stock bars, `feed=sip` | 403 subscription |

Options trading **level 3** is approved and orders fill; that is a trading
permission and is unrelated to the data entitlement. Alpaca also serves no
options data at all before 2024-02-05, which `OPTIONS_HISTORY_FLOOR` encodes so
a longer lookback cannot silently return a short series.

So the read carries three volatility figures instead of one, and says which it
could compute:

- **`iv_vs_hv`** — current at-the-money implied over trailing realised.
  Exactly computable today from the chain's own greeks and the underlying's own
  bars. This is what carries "is premium rich".
- **`hv_rank`** — realised volatility ranked against its own trailing range.
  Needs no options entitlement at all.
- **`iv_rank`** — `None` until `MIN_HISTORY` sessions are observed.
  `IvHistoryStore` accumulates one observation per session, and
  `seed_from_option_bars` back-fills by inverting Black–Scholes over one
  contract's daily closes the moment the entitlement exists.

This is the reuse dividend from [adr/0001](../adr/0001-core-reuse.md). The
median entrant will paste OHLC into a prompt and ask the model what it thinks.
We hand it a trend state machine's output.

## D3 — The model chooses from a menu; it never writes a symbol

**No free-form OCC symbols. Ever.** Step 3 fetches the live chain and builds a
bounded set of fully-formed, already-validated `OptionStructure` candidates —
typically 6 to 12, each with its `StructureRisk` already computed. The model
receives them as an indexed list and returns **an index**, or `null`.

**The menu must not rank the Gate's refusals to the top.** Found by watching the
live agent, twice, and it is the same mistake D4 already forbids for sizing:

- Ranking by raw return on risk puts the strike closest to the money first,
  which is also the highest delta — so index 0 was systematically the one
  candidate `net_delta_budget` would refuse. The model chose it, correctly by
  its own lights, and was vetoed every cycle.
- The closest strike also has the widest market, so fixing delta moved the veto
  to `liquidity` and nothing else changed.

Two fixes, and the second is the interesting one. Candidates that breach the
delta budget at their own size are dropped before the model sees them, exactly
as zero-quantity candidates are. And the ranking is now **net of the crossing
cost** — the bid/ask you must pay to enter is subtracted from the return you are
ranking on. That is the financially correct metric regardless (a 30% return
costing 6% to enter is not better than a 26% return costing 1%), and it stops
the ordering from adversarially selecting whatever the Gate refuses.

A shortlist filter that is looser than the Gate is fine and deliberate. A
*ranking* that is adversarial to the Gate is not, and the difference is worth
stating because the first looks like the second until you watch it run.

```python
@dataclass(frozen=True, slots=True)
class Choice:
    candidate_index: int | None    # None = decline, and that is a valid answer
    rationale: str
    confidence: float              # recorded, never acted on — see D5
```

A hallucinated ticker cannot reach the broker, because the model has no channel
through which to express one. This is the same move as [02](02-options-domain.md)
D3 making naked shorts unconstructible: the failure mode is designed out rather
than validated against.

Enforced by structured output (tool-use schema), and re-validated on receipt:
an index outside the candidate range is a decline, not an error to retry around.

## D4 — The model never sizes

Quantity is a pure function of the risk budget and the chosen structure's
`max_loss`:

```python
quantity = floor(limits.max_trade_loss / structure.max_loss)
```

If that is zero, the candidate is dropped before the model ever sees it. Sizing
is where an LLM's lack of calibration does the most damage per unit of
plausibility, so it does not get a vote.

## D5 — Confidence is recorded, not acted on

The model returns a confidence; the system does not scale position size by it.
Self-reported confidence is not calibrated, and treating it as a probability is
the most common way an agent turns a good structure into a bad bet. It is kept
in the journal so the demo can show whether it *would* have correlated — an
honest observation is worth more than a fake edge.

The field is called **`self_reported_confidence`**, and the length is the
enforcement. A boundary test asserts that no module outside the type that
declares it and the adapter that parses it so much as names it — which was
unenforceable while it was called `confidence`, because `TrendState` and `Level`
each have one too and theirs is a *measured* quantity: how much of the requested
evidence an engine could read. The first draft of the guard flagged the trend
engine, and the right response was to fix the name rather than loosen the guard.
Nobody types `choice.self_reported_confidence` into a sizing formula without
noticing what they are about to do, and that — not a deliberate multiplication —
is the plausible mistake.

## D6 — Fail closed

| Failure | Behaviour |
| --- | --- |
| LLM unavailable, times out, or returns malformed output | **No trade.** Never a default or fallback trade. |
| Chain or quote fetch fails | No candidates, no trade |
| Quotes stale beyond `max_quote_age` | Candidates dropped before the model sees them |
| Perception incomplete — no trend, no IV/HV, no ATR | No setup, no proposal |
| No earnings calendar for the underlying | No setup. `None` is not `False` |
| Gate vetoes | No trade; all check results journalled |
| Kill switch tripped | Opens blocked, closes still permitted ([03](03-risk-gate.md) D4) |

There is no path where a degraded system trades anyway. Doing nothing is always
an available and correct outcome.

The last two rows are the ones that took a live account to get right. Every
field of `MarketRead` except `spot` is `Optional`, `None` means *unmeasured*,
and the screen refuses to build a `Setup` from a read that is missing anything
it needs. `earnings_within_dte` is `bool | None` for exactly this reason: Alpaca
has no earnings calendar, and the distinction that costs money is between "no
event in the window" and "nobody checked" — which is precisely the distinction a
`bool` cannot make.

## D7 — Determinism, given that step 4 is not

The LLM is the one nondeterministic component, so it is fenced:

- **Temperature 0**, fixed model id, fixed prompt template version. All three go
  into the journal record; a prompt edit is a version bump.
- Steps 1–3, 5, 6 are pure and property-tested. Same bars, same candidates, same
  sizing, same verdict.
- **Replay mode** substitutes a `RecordedProposer` that returns the journalled
  choice for a given cycle id. A journalled day replays to the identical order
  set, which is what makes the record evidence rather than narration.
- The backtest (spec 08, not yet written) runs a `DeterministicProposer` — a pure
  rule over the same `MarketRead` — so the strategy can be measured over months
  without months of model calls. **What the backtest measures is the strategy,
  not the model.** The submission must say so plainly rather than implying the
  agent was backtested.

## D8 — Cadence and scope

- Watchlist: the liquid optionable underlyings the strategy may trade. Breadth
  is not the point; enough fills to escape single-trade noise is
  ([00](00-brief.md)).

  [07](07-strategy.md) D2 scopes it to **`SPY` alone**: the tightest option
  market listed, no earnings by construction, and the only underlying with free
  history to backtest on.

  `agent/watchlist.py` holds that one entry. `tradeable_today()` still exists
  and is still called rather than assumed — it returns what the screen can
  actually admit, and under D2 that is always the whole watchlist, so an empty
  return means the earnings calendar has changed shape and the run must say so
  rather than idling quietly.

- One underlying per slot, round-robin across whatever `tradeable_today()`
  returns. A cycle is about one name, and rotating spreads the day's fills
  across the watchlist rather than concentrating them in whichever name came
  first.
- Cycle every 15 minutes during RTH, plus one pass 20 minutes after the open.
  Not on the open itself — the first minutes' quotes are the widest of the day
  and `worst_spread_pct` would veto most candidates anyway.
- Exits are evaluated every cycle and are **not** the model's decision: profit
  target, stop, and DTE-based close are deterministic rules. The Gate never
  blocks an exit.

The concrete entry rules — which `MarketRead` shapes become a `Setup`, and which
structure family each maps to — are [07-strategy.md](07-strategy.md), written
separately so that tuning thresholds does not mean editing the orchestration.
What lives in this layer is the `Screen` protocol and one fail-closed default;
when [07](07-strategy.md) lands it supplies a `Screen` and nothing here changes.

### The schedule is computed, not slept into

`session_slots(open, close)` returns the instants; the runner walks them. A loop
that decides when to run *while* running can only be tested by waiting — this
way a whole trading day, half days included, is tested in microseconds. Session
bounds come from the exchange clock and never from a local one: a 13:00 close is
a real thing three times a year, and a hardcoded 16:00 would have the agent
proposing into a closed market.

One rule the spec did not state and should: **no new positions in the last
twenty minutes.** An order placed at 15:58 that does not fill is a `day` order
that expires, and options support no other time in force ([04](04-execution.md)
D3). Those slots are `EXITS_ONLY` — never *no* slot, because closing at 15:55 is
exactly when you want to be able to.

## Test plan (RED first)

1. The cycle writes a journal entry on every path: no setup, no candidates,
   decline, veto, submit.
2. An out-of-range or non-integer candidate index is treated as a decline.
3. A malformed or absent model response produces no order (D6).
4. Sizing: `max_loss` at, just under, and just over the budget; zero-quantity
   candidates never reach the proposer.
5. Confidence does not appear in any sizing or gating code path (boundary test).
6. `RecordedProposer` replays a journalled day to a byte-identical order set.
7. Steps 1–3 and 5 are deterministic under hypothesis.
8. Stale quotes drop candidates before proposal, not after.
