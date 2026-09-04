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

> Superseded 2026-09-04 by the pruning in D5: the only tests that needed
> `alphagate.infra` were the ones testing the modules D5 pruned, so the package
> went with them and the dependency on `pandas-market-calendars` went with it.
> Guard 1 still holds, now vacuously — there is no `infra` to reach sideways
> into.

**D4.** Do **not** reuse PMM's `application/`, `interface/`, `persistence/` or
provider layers. They are shaped for a monitoring product with a human in the
loop. PMM's Alpaca adapter in particular is market-data only — it targets
`data.alpaca.markets` with a GET-only transport seam — so order execution is new
work regardless.

**D5.** Carry the unused-but-passing subpackages (`alerts`, `alert_engine`,
`news`, `stores`, `operations`, `accounts`) rather than pruning them. Splitting a
green test suite costs competition hours and buys nothing; pruning happens after
4 September.

> **Done, 2026-09-04.** Pruned on the last day, with the build finished. Gone:
> `alerts`, `alert_engine`, `news`, `stores`, `operations`, `accounts`,
> `level_store`, `market_data`, `normalization`, `aggregation`, `symbol`,
> `clock`, the `UserLevel`/`Priority` types in `levels`, the five account and
> alert id types in `identifiers`, the re-export barrel in `core/__init__.py`,
> and all of `alphagate.infra` — with their tests. About 4k lines of source and
> 356 tests.
>
> The criterion was reachability, not judgement: a module was pruned only if no
> module reachable from `python -m alphagate` imported it by name. The barrel in
> `core/__init__.py` was excluded from that count deliberately — it re-exported
> everything and was itself imported by nothing, so counting it would have kept
> the whole package alive on the strength of a file no consumer used. It is now
> a docstring.
>
> What this buys, on a submission where the reuse is declared: `alphagate.core`
> is now the perception layer the agent actually perceives with, and a reader
> can no longer find a notification-delivery store or a news screener inside a
> trading agent and wonder what it is for. `pandas-market-calendars` and
> `anthropic` left with it — 17 packages including pandas and numpy.
>
> D2's principle survives: every module still here still has its upstream tests.
> 2299 pass, `ruff` and `mypy --strict` are clean over 109 source files.

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
