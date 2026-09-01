"""Executor policy — the numbers that are ours, not the strategy's.

PURE. Stdlib plus `alphagate.core`.

Everything here is a decision about *placing* a book, never about *choosing*
one. The strategy is what `ai_quant_researcher` validated; the band, the floor
and the caps below are the cost of turning its weights into orders, and keeping
them in a separate type from anything the researcher produced is what stops the
two from being confused.

Budgeted limits are fractions of equity for the same reason
[03](../../../../specs/03-risk-gate.md) D5 gives: a dollar cap written for a
$100k paper account silently becomes a different policy on a $30k one.

`DEFAULT_EQUITY_POLICY` is the competition configuration. One module, logged at
startup, rendered in the dashboard — a limit nobody can read is a limit nobody
is enforcing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation

__all__ = ["DEFAULT_EQUITY_POLICY", "EquityPolicy"]


@dataclass(frozen=True, slots=True)
class EquityPolicy:
    """One immutable configuration. Never mutated; replaced wholesale."""

    drift_band_pct: Decimal
    """Minimum drift before a symbol is traded, as a fraction of **that
    position's own size** — not of the account.

    specs/09 D3. The strategy rebalances every five sessions; between them the
    targets do not move but the holdings drift with price, so a naive diff would
    trade every name every day and pay costs no backtest ever charged.

    Proportional rather than a flat fraction of equity, and the first version was
    the flat one and was wrong in a way worth recording. A 0.25% band on a $100k
    account is $253. The book's sleeve positions are 0.192% of equity — $194
    each — so *every sleeve position was permanently inside the band*: the
    hundred names the strategy holds could never be established at all, and the
    ten core names would have been bought into an account that was otherwise
    cash. The book would have been a tenth of the strategy, and nothing would
    have reported an error.

    A band relative to position size cannot have that failure, because the
    threshold for establishing a position is a fraction of the position rather
    than a constant it might be smaller than."""

    min_order_notional: Decimal
    """Absolute floor on one order, in account currency.

    The proportional band alone would admit a $4 order on a $20 position, and a
    $4 order is all spread. This is the floor that makes the band cost-aware, and
    it is the binding constraint on exactly the small sleeve names where it
    should be."""

    max_position_pct: Decimal
    """Ceiling on one resulting position. The book's own largest core weight is
    8%, so this is a guard against a malformed book rather than a constraint on
    a healthy one — which is exactly what a Gate check should be."""

    max_gross: Decimal
    """Total exposure ceiling. 1.0 is fully invested; anything above is
    leverage, and no run upstream measured leverage."""

    max_daily_turnover_pct: Decimal
    """Traded notional in one session, as a fraction of equity. The first
    session builds the whole book, so this is generous by necessity; what it
    stops is a loop that rebuilds it repeatedly."""

    max_daily_orders: int
    """The book holds 104 names, so a full build is 104 orders. This is a
    circuit breaker on a runaway plan, not a limit on a rebalance."""

    max_book_age_days: int
    """Calendar days between the book's `as_of` and today. A book from last
    month describes a portfolio the strategy no longer holds."""

    max_quote_age: float
    """Seconds. Inclusive: a quote exactly this old is still fresh. `float`
    because it is a duration, not money."""

    max_drawdown_pct: Decimal
    """Peak-to-current, and the kill switch's threshold. Latched by the caller —
    a pure policy cannot remember that it tripped yesterday."""

    fractional_places: int
    """Decimal places for a fractional share quantity. Alpaca accepts nine;
    four is well inside that and keeps the journal readable."""

    def __post_init__(self) -> None:
        for name in (
            "drift_band_pct",
            "min_order_notional",
            "max_position_pct",
            "max_gross",
            "max_daily_turnover_pct",
            "max_drawdown_pct",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")
            if value <= 0:
                raise InvariantViolation(
                    f"{name} must be positive, got {value}; a limit of zero is not a "
                    "conservative limit, it is a disabled system"
                )
        for name in ("max_daily_orders", "max_book_age_days", "fractional_places"):
            count = getattr(self, name)
            if count <= 0:
                raise InvariantViolation(f"{name} must be positive, got {count}")
        # NaN fails every comparison silently, so ask explicitly rather than
        # letting a NaN limit disable the check it configures.
        if not math.isfinite(self.max_quote_age):
            raise InvariantViolation(f"max_quote_age must be finite, got {self.max_quote_age}")
        if self.max_quote_age <= 0:
            raise InvariantViolation(f"max_quote_age must be positive, got {self.max_quote_age}")
        if self.fractional_places > 9:
            raise InvariantViolation(
                f"fractional_places is {self.fractional_places}; Alpaca accepts nine"
            )

    # --------------------------------------------------------------- #
    # Absolute limits, derived from the equity they are a fraction of
    # --------------------------------------------------------------- #

    def threshold(self, target_notional: Decimal, held_notional: Decimal) -> Decimal:
        """The drift one symbol must exceed before it is traded, in currency.

        Measured against the **larger** of what is wanted and what is held, which
        is what makes the same rule cover all three cases:

        * *establishing* a position — held is zero, so the threshold is a
          fraction of the target and the full target clears it;
        * *drifting* — the two are close, so it is a fraction of the position;
        * *exiting* — the target is zero, so it is a fraction of the holding, and
          the whole holding clears it.

        Floored by `min_order_notional`, so a tiny position is not rebalanced by
        orders worth less than their spread.
        """
        size = max(abs(target_notional), abs(held_notional))
        return max(self.drift_band_pct * size, self.min_order_notional)

    def max_position(self, equity: Decimal) -> Decimal:
        return self.max_position_pct * equity

    def max_daily_turnover(self, equity: Decimal) -> Decimal:
        return self.max_daily_turnover_pct * equity


EQUITY_SLEEVE_ALLOCATION: Final = Decimal(90_000)
"""The capital assigned to the equity book — specs/03 D6.

90% of a $100,000 account; the other 10% is the options agent's
`OPTIONS_SLEEVE_ALLOCATION`. The two must sum to no more than the account,
because Alpaca holds one pool of buying power and has never heard of sleeves.

**This is not the strategy's cash reserve.** The book's own gross is 0.61, so it
already holds roughly 39% of its base in cash by design, for rebalancing. That
reserve is a fraction of whatever base it is given, and taking the options
sleeve out of the base rather than out of the reserve is what keeps it
proportionally intact — the alternative spends cash the strategy was relying on.

So the idle cash an operator sees in the account is two different things added
together, and reading it as one number is the mistake this note exists to
prevent: roughly 39% of this sleeve is the equity strategy's own rebalancing
reserve, and `OPTIONS_SLEEVE_ALLOCATION` on top of it is capital that was never
the equity book's to spend.

**Reduced from $95,000 when the options sleeve was doubled.** The options rule
could not open a single position at $5,000 — one contract of specs/07 D1's
structure risks $1,389, against a $1,000 per-trade budget — so the split moved
to 90/10 to make the researched sizing executable. The equity book is a
weight-based book, so the cost is exactly linear: it now runs at 90% rather than
95% of the scale `ai_quant_researcher` validated, returns scale by 0.90, and the
strategy's character does not change. It is still a deviation from the backtest
and the submission still says so.

**Changing this constant does not force a rebalance.** Every target is a weight
times this number, so a 95,000 -> 90,000 move shrinks each one by about 5.3% —
and the no-trade band is 20% of the position with a $25 floor (specs/09 D3),
which a 5.3% drift does not clear on any of the 87 names in the current book.
The positions converge at the next scheduled rebalance, when the weights are
recomputed anyway. `equity-plan` prints what would be ordered without ordering
it, which is the way to check that rather than assume it.
"""


DEFAULT_EQUITY_POLICY: Final = EquityPolicy(
    # A fifth of the position. On a $8,276 core name that is $1,655 of drift
    # before an order — roughly a 20% move, which a five-session rebalance
    # period will usually correct anyway. On a $194 sleeve name it is $39, and
    # the $25 floor below is what stops it going lower.
    drift_band_pct=Decimal("0.20"),
    min_order_notional=Decimal(25),
    # Comfortably above the book's 8.2% largest name and far below anything that
    # would read as a concentrated bet.
    max_position_pct=Decimal("0.15"),
    max_gross=Decimal("1.00"),
    # A first-session build is one turnover of the whole account. 1.2 leaves
    # room for that plus a rebalance, and stops a third pass.
    max_daily_turnover_pct=Decimal("1.20"),
    max_daily_orders=250,
    # Five sessions is the rebalance period; seven calendar days covers it
    # across a weekend without admitting a book from the session before last.
    max_book_age_days=7,
    max_quote_age=90.0,
    max_drawdown_pct=Decimal("0.10"),
    fractional_places=4,
)
