# 09 — Equity execution: the handoff, and what happens after it

The chain this spec closes:

```
aqr research → walk-forward → pre-register → sealed run → target book (a file)
                                                              │
                                                              ▼
                                            AlphaGate: load → plan → gate → submit
                                                              │
                                                              ▼
                                                journal + status.json → dashboard
```

Spec 08 stays reserved for the options backtest. This is the equity path, and it
exists because [`ai_quant_researcher/`](../ai_quant_researcher/) produces a
validated *equities* strategy and then stops — deliberately. Its
`CONSUMER_MUST_SUPPLY` names six things it will not do, and every one of them is
a decision below.

## D0 — The seam is a file, and it stays a file

`ai_quant_researcher` does not import `alphagate`. `alphagate` does not import
`aqr`. Neither knows the other's package exists. The interface is a target-book
JSON artefact on disk, and `tests/test_boundaries.py` fails the build if either
direction of import appears.

That is not fastidiousness. The two projects answer different questions and hold
different invariants — the researcher holds no money and places no orders, so it
has no `Decimal` rule and no Risk Gate; AlphaGate holds both. An import would
make one project's invariants the other's problem, and
[CLAUDE.md](../CLAUDE.md) §2b already says not to reconcile them.

The file is also what makes the claim auditable. A book carries its own
fingerprint, seal state, sealed-run measurement and both multiplicity
denominators, so a reader who has never seen `aqr` can check where the weights
came from.

## D1 — Only a book that earned it may be loaded

`load_target_book` is pure — it takes a parsed mapping, not a path — and it
refuses in seven ways. All seven are checked, and every fault is reported at
once, because a book that is wrong in three ways should not be fixed three
times.

| Refusal | Why |
| --- | --- |
| `schema_version != 1` | fields may mean something new under the same number |
| `spec_fingerprint` ≠ the pinned one | the pin is what makes "only this strategy" true |
| `provenance.status` not in `{PAPER, LIVE}` | a CANDIDATE has not earned paper money |
| `sealed_look < 1` | the seal is unspent; the rule still owes an out-of-sample verdict |
| `sealed_measurement.refuted` | the sealed window refuted it |
| any weight < 0 | a short leg needs a locate and a different risk shape (D6) |
| `gross` > 1.0 + tolerance | leverage the researcher never measured |

**The fingerprint is pinned in configuration, not read from the book.** A book
names the strategy it describes; if the backend believed that name, swapping the
file would swap the strategy and nothing would notice. So the operator pins
`ALPHAGATE_STRATEGY_FINGERPRINT` once and a book that disagrees is refused by
name. This is the whole of "only the strategy `ai_quant_researcher` provides".

`provenance.status` comes from the registry's `CANDIDATE → PAPER → LIVE` state
machine, which refuses to skip a step. Reading it here means the lifecycle rule
is enforced on the execution side too, rather than trusted.

## D2 — Weights become shares, and the executor owns every step

The book carries weights. Turning one into an order needs four things it
deliberately does not have: equity, a price, what is already held, and a policy.

```
target_notional[s] = weight[s] × equity
target_shares[s]   = target_notional[s] / price[s]
delta_shares[s]    = target_shares[s] − held_shares[s]
```

Rounded per asset: fractional to 4 dp when Alpaca reports the asset
`fractionable`, floored to a whole share otherwise. A whole-share floor that
produces zero is a **skip with a journalled reason**, not a silent drop — on a
$100k account a 0.19% sleeve position is $192, and a non-fractionable $500 name
rounds to nothing. That is a real hole in the book and the record should say so.

`price` is the snapshot mid, or the last trade when there is no two-sided quote.
Never the book's own bars: the book is as of yesterday's close and the order
fills today.

## D3 — The no-trade band, and why daily drift is not a signal

The strategy rebalances every 5 sessions and holds 10 core names. Between
rebalances the target weights do not change, but the *held* weights drift with
price, so a naive diff would trade every name every day and pay costs the
backtest never paid.

So an intent is emitted only when

```
|delta_notional| ≥ max(band_pct × equity, min_order_notional)
```

Default band 0.25% of equity, minimum order $25. Both are executor policy, both
are recorded in the journal beside the plan, and neither is a strategy
parameter — the strategy is what `aqr` validated, and this is the cost of
placing it.

## D4 — Sells before buys, and a deterministic order

The plan is a sequence, and the sequence is part of the contract:

1. **sells first**, largest notional first, ties broken by symbol;
2. **buys second**, largest notional first, ties broken by symbol.

Sells release buying power that buys then spend. Placing them the other way
round on a fully-invested book means the first buys are refused for want of cash
and the last sells leave the account holding it — a rebalance that half-happened
and is worse than either end state.

Ties broken by symbol so the same inputs produce the same sequence. Determinism
is [CLAUDE.md](../CLAUDE.md) §3 rule 7 and it is what lets a plan be replayed.

## D5 — The equity Risk Gate

The options Gate does not apply — its checks are about short strikes, DTE and
defined-risk width, none of which an equity order has. `CONSUMER_MUST_SUPPLY`
says so explicitly. So there is a second gate, with the same shape and the same
guarantees: pure, no clock, no I/O, no model, every check runs, no
short-circuit, `CHECKS` is a tuple and that tuple is the order.

| Check | Refuses |
| --- | --- |
| `book_is_pinned` | a plan whose book fingerprint is not the pinned one |
| `book_is_fresh` | a book older than `max_book_age_days` sessions |
| `symbol_is_tradeable` | an asset the broker reports untradable or halted |
| `price_is_fresh` | a quote older than `max_quote_age` seconds |
| `order_is_material` | a notional under the floor — D3, asserted again here |
| `position_cap` | a resulting position over `max_position_pct` of equity |
| `gross_exposure_cap` | a resulting gross over `max_gross` |
| `buying_power` | a buy whose notional exceeds available buying power |
| `daily_turnover_cap` | today's traded notional over `max_daily_turnover_pct` |
| `daily_order_cap` | more than `max_daily_orders` orders today |
| `no_short_selling` | any sell that would take the position below zero |
| `drawdown_killswitch` | drawdown past the limit, or the latch already set |

Boundaries are inclusive on the safe side: a value exactly at its limit passes.

**A sell that reduces or closes is never blocked by a budget.** Same rule as
[03](03-risk-gate.md) D4's exit waiver, for the same reason: the checks are
computed so the dashboard has the numbers, and a failure on a risk-reducing
order is waived rather than turned into a veto. `no_short_selling` and
`symbol_is_tradeable` are not waivable — one would open a short leg, the other
cannot be filled.

## D6 — No short selling, and the structure that makes it unexpressible

`direction: market_neutral` in the DSL means the *ranking* is cross-sectional.
The book it produced holds 104 long positions and no negative weight. A short
equity leg needs a locate, accrues borrow, and has unbounded loss — a different
risk shape entirely, and one nothing here has measured.

So `EquityOrderIntent` carries `shares: Decimal` and a `side`, and the planner
never emits a sell larger than the held quantity. D1 refuses a negative weight
at load; the planner cannot construct one; the Gate checks it anyway. Three
layers, same as [02](02-options-domain.md) D3's treatment of naked options.

## D7 — The one door, again

```python
def submit_equity(order: GatedEquityOrder, mcp: McpSession) -> Submission
```

`GatedEquityOrder` is minted only inside `risk.equity_gate`, enforced by the
same frame check that guards `GatedOrder` ([03](03-risk-gate.md) D3), and
`tests/test_boundaries.py` guard 5 is extended to cover **both** gated types and
**both** doors. A second execution path added without a second guard is the
failure mode this whole spec is arranged against.

Orders are `market`, `day`, `extended_hours: false`. A limit order on a
rebalance is a rebalance that may not happen: the book is a set of weights to be
holding, not a price to be got, and an unfilled leg leaves the account in a
state neither the backtest nor the plan describes. Fractional quantities require
market/day anyway.

`client_order_id` is derived from `(fingerprint, as_of, symbol, side, trading
day)`. A timeout is resolved by reading it back, never by resending —
[04](04-execution.md) D4, unchanged.

## D8 — The cadence

`equity-run` is a long-lived process:

* **one rebalance pass per session**, at a configured offset after the open
  (default 15 minutes — enough for the opening auction to settle, early enough
  that the book is not a lunchtime book);
* **a heartbeat every 30 seconds** that re-reads account and positions, re-marks
  the book, and rewrites `equity-status.json`, so the dashboard is live even on
  the four days in five when the plan is empty;
* **a reconcile pass** that reads back every order the session submitted and
  amends its journal line with the fill.

The pass is idempotent by day: a restart at 14:00 finds the day's plan already
journalled and does not replay it. Same discipline as the options runner.

## D9 — What lands on disk

| Path | What |
| --- | --- |
| `journal/YYYY-MM-DD.jsonl` | the same file. Equity cycles carry `kind: "equity"` |
| `journal/equity-status.json` | the live equity book, rewritten every heartbeat |
| `journal/books/<fingerprint>-<as_of>.json` | the book as loaded, copied verbatim |

The book is copied rather than referenced. `aqr` regenerates
`runs/target_books/` daily; a journal line pointing at a path whose contents
have since changed is a record of nothing. The copy is hashed and the digest
goes in the journal line, so a book on disk that no longer matches what was
executed can be identified as such.

A cycle line carries the plan, every intent, every intent's verdict with the
full check tape, and the submission. `NO_TRADES` is journalled like any other
outcome — "why did it not rebalance today" is the question the journal exists
for, and on this strategy it is the answer four days in five.

## D10 — The dashboard shows the provenance, not just the positions

The Live tab gains an **Equity** view with four blocks:

* **the strategy** — name, fingerprint, `as_of`, and the sealed-run numbers:
  alpha, beta, `t`, information ratio, the looks count and the significance bar
  it had to clear. This is the block that makes the demo a claim rather than a
  screenshot;
* **the book** — target weight against held weight per symbol, the drift, and
  whether it is inside the no-trade band;
* **today** — every intent, its verdict, and its fill;
* **the gate** — the check tape, tightest-first, same as the options view.

Read from `equity-status.json` and the journal. `alphagate.interface` still
imports neither `execution` nor `live` nor `marketdata`, so there is still no
code path from a browser to an order.

## Test plan (RED first)

1. `load_target_book` refuses each of D1's seven faults, and reports several at once.
2. A book with the wrong fingerprint is refused even though it is otherwise valid.
3. `plan_rebalance` is deterministic: same inputs, same sequence, byte-identical.
4. Sells precede buys, and both are ordered by notional then symbol.
5. A drift inside the band emits no intent; one exactly at the band emits one.
6. A non-fractionable asset whose target rounds to zero produces a skip with a reason.
7. Each Gate check vetoes on its own and passes at its exact boundary.
8. A reducing sell is waived past a budget veto; it is not waived past `no_short_selling`.
9. `GatedEquityOrder` cannot be constructed outside `risk.equity_gate`.
10. `submit_equity` refuses anything that is not a `GatedEquityOrder`.
11. A timeout produces a read-back, never a second `place_stock_order`.
12. The whole chain runs offline from a fixture book and a `RecordedSession`.
