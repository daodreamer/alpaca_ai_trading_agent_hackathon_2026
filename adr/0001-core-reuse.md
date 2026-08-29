# ADR 0001 — Reuse the Personal Market Monitor domain as `alphagate.core`

- Status: accepted
- Date: 2026-08-25 (pre-competition)

## Context

The Alpaca AI Trading Agents Hackathon runs 28 Aug – 4 Sep 2026, solo, seven days.
An options agent needs a perception layer: cleaned bars, indicators, market
structure, price levels, trend state. Written from scratch that is three days of
the seven, and it is the part with the least competitive differentiation — every
entrant needs it and nobody is judged on their RSI.

The author already maintains *Personal Market Monitor* (PMM), whose `pmm.domain`
package is exactly that layer: 9.6k lines, pure standard library, Decimal money,
tz-aware UTC throughout, deterministic by construction, with an 11k-line test
suite and an architecture guard test that enforces its purity.

## Decision

**D1.** Extract `pmm.domain` verbatim into this repository as `alphagate.core`.
The change is a package rename, applied mechanically. No logic is edited.

**D2.** Extract its full test suite as `tests/core`, unmodified. A reused library
that arrives without its tests is a reused liability.

**D3.** Extract the three infrastructure modules the core's own tests depend on —
`clock`, `calendars`, the in-memory stores — into `alphagate.infra`. The pure
layers may not import from it; `tests/test_boundaries.py` enforces that.

**D4.** Do **not** reuse PMM's `application/`, `interface/`, `persistence/` or
provider layers. They are shaped for a monitoring product with a human in the
loop. PMM's Alpaca adapter in particular is market-data only — it targets
`data.alpaca.markets` with a GET-only transport seam — so order execution is new
work regardless.

**D5.** Carry the unused-but-passing subpackages (`alerts`, `alert_engine`,
`news`, `stores`, `operations`, `accounts`) rather than pruning them. Splitting a
green test suite costs competition hours and buys nothing; pruning happens after
4 September.

**D6.** Declare the reuse prominently — in the README, in this ADR, and in the
submission. The reused code predates the competition and is the author's own
open-source work.

## Consequences

- Day one starts at the options domain instead of at `Bar`.
- `alphagate.core` arrives with mypy strict and `filterwarnings = ["error"]`
  already passing. That strictness is kept for `core`, `options` and `risk`,
  and deliberately relaxed for the dashboard — see specs/01-architecture.md.
- The two repositories diverge from this point. No attempt is made to keep them
  in sync; this is a fork for a seven-day event, not a shared dependency.
- Risk: if the organisers read pre-existing code narrowly, this reuse could be
  challenged. Mitigation is D6 plus confirming the rule in the event Discord
  before kickoff. **Open action, must close before 28 Aug.**

## Verification

Extraction is complete and green in this repository:

```
1089 passed, 2 skipped     # the 2 skips are options/ and risk/, not yet written
ruff check .               All checks passed
mypy                       Success: no issues found in 56 source files
```
