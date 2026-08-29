# 02 — Options domain model

Pure. Stdlib only. No provider types, no LLM, no I/O.

## D1 — Contract identity

An option contract is identified by four facts, not by a string.

```python
@dataclass(frozen=True, slots=True)
class OptionContract:
    underlying: Ticker          # core.identifiers.Ticker
    expiry: date                # expiration date, exchange-local calendar date
    strike: Decimal            # exact; never float
    right: Right               # CALL | PUT
    multiplier: int = 100      # shares per contract
```

`OccSymbol` is a *rendering* of a contract (`AAPL260918C00150000`), not the
identity. Parsing and formatting live in `options.occ` with a round-trip
property test: `parse(format(c)) == c` for any valid contract.

Invariants (raise `InvariantViolation`):

- `strike > 0`
- `multiplier > 0`
- `strike` must quantise exactly to the contract's strike increment; a strike
  that does not round-trip through OCC's 8-digit thousandths encoding is invalid.

## D2 — Quote and greeks

```python
@dataclass(frozen=True, slots=True)
class OptionQuote:
    contract: OptionContract
    as_of: datetime            # tz-aware, UTC
    bid: Decimal
    ask: Decimal
    ...
    @property
    def mid(self) -> Decimal          # (bid + ask) / 2, quantised to a cent
    @property
    def spread_pct(self) -> Decimal   # (ask - bid) / mid
```

```python
@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float; gamma: float; theta: float; vega: float; rho: float
    iv: float
```

Money is `Decimal`; greeks and IV are `float` (D3 in [01](01-architecture.md)).

**Staleness is explicit.** A quote older than `MAX_QUOTE_AGE` (default 60s) is
stale. The Risk Gate vetoes on stale quotes; the domain only reports the age.

**Missing greeks are `None`, never zero.** A provider that omits delta must not
be silently read as delta-neutral.

## D3 — Structure, not leg

The unit the agent proposes and the Gate judges is a **structure**, never a
bare leg. This is the mechanism that makes "no naked options" enforceable by
construction rather than by review.

```python
class StructureKind(Enum):
    VERTICAL_CREDIT      # short spread, defined risk
    VERTICAL_DEBIT
    IRON_CONDOR
    COVERED_CALL         # requires long stock cover
    CASH_SECURED_PUT     # requires cash cover
```

```python
@dataclass(frozen=True, slots=True)
class Leg:
    contract: OptionContract
    side: Side                  # BUY | SELL
    quantity: int               # > 0

@dataclass(frozen=True, slots=True)
class OptionStructure:
    kind: StructureKind
    legs: tuple[Leg, ...]
```

Construction validates the legs against the kind — a `VERTICAL_CREDIT` with one
leg, or with legs on different underlyings or expiries, does not construct.
There is no `StructureKind.CUSTOM` and no naked-short kind. **If it cannot be
built, it cannot be proposed, and it cannot be sent.**

## D4 — The numbers the Gate needs

Every structure computes, purely, from a quote set:

```python
@dataclass(frozen=True, slots=True)
class StructureRisk:
    net_premium: Decimal        # credit received positive, debit paid negative
    max_loss: Decimal           # ALWAYS finite and > 0. Never None.
    max_profit: Decimal
    breakevens: tuple[Decimal, ...]
    net_greeks: Greeks | None   # None if any leg's greeks are missing
    worst_spread_pct: Decimal   # widest bid/ask across the legs
    days_to_expiry: int
```

**The sign convention is the finance one and it is not Alpaca's.** A credit is
positive here; Alpaca's `limit_price` says the opposite. The flip happens in one
named function in the adapter — see [04-execution.md](04-execution.md) D2. Do not
"fix" the convention in this module to match the wire format; the arithmetic
below reads correctly only this way.

`max_loss` being non-`None` and finite is a **type-level invariant**. A structure
whose loss is unbounded cannot be represented, so the Gate never has to ask.

Cash-secured put: `max_loss = (strike - credit) * multiplier * qty` — assignment
to zero, not "unlimited". Covered call: loss is bounded by the stock leg, which
must be evidenced by a held position at construction time.

## D5 — Determinism

`StructureRisk` is a pure function of `(OptionStructure, quotes, as_of)`.
Same inputs, same outputs, no clock reads, no randomness. Property-tested with
hypothesis, the way the core's engines are.

## Test plan (RED first)

1. OCC round-trip, including fractional and high strikes.
2. Contract invariants: negative strike, zero multiplier, non-quantisable strike.
3. Structure construction: every kind's valid shape, and the rejections —
   mismatched underlying, mismatched expiry, wrong leg count, a lone short leg.
4. `StructureRisk` arithmetic per kind, hand-computed fixtures, exact Decimal.
5. `max_loss` finite **and positive** for every constructible structure
   (hypothesis, all kinds) — `tests/options/test_properties.py`. The generators
   draw a quantity as well as strikes and premiums: every hand-written fixture
   uses one contract, so an arm that dropped the quantity scaling would leave
   the whole suite green while reporting a ten-lot's risk as a one-lot's.
6. Missing greeks propagate as `None`, never as `0.0`.
7. Quote staleness reported against a supplied `as_of`, never `datetime.now()`.
