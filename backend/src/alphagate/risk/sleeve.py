"""Capital sleeves — specs/03 D6.

PURE. Stdlib plus `alphagate.core`.

Two strategies share one broker account. The equity book (specs/09) holds a
ranked, fully-invested portfolio; the options agent (specs/07) sells
defined-risk credit spreads. They answer different questions, hold different
invariants, and — until this module existed — **budgeted against the same
number**.

Both gates scaled their limits off `account.equity`, and both watched
`account.equity` for a drawdown. On a $100,000 account that meant:

* the options kill switch, set at 5%, was tripped by a $5,000 fall in the
  account — which a $95,000 equity book produces on an 8% market move, while
  the options sleeve has lost nothing and is then latched shut until a human
  clears it;
* every options budget grew and shrank with the equity book's mark-to-market,
  so how much premium the agent could sell depended on what stocks did
  overnight.

Neither is a risk control. Both are one strategy's noise arriving as another
strategy's constraint.

A `Sleeve` is the fix, and it is deliberately almost nothing: an allocation, and
the P&L of the positions that allocation paid for. **The account does not appear
in the arithmetic**, which is the entire design. `tests/risk/test_sleeve.py`
asserts that structurally rather than trusting it, because an edit that passes
account equity in to make a sleeve "aware of its surroundings" would restore the
coupling without failing any arithmetic test.

What a sleeve is *not* is a broker sub-account. Alpaca holds one pool of buying
power and it does not know about this file. A sleeve constrains what each
strategy will *ask* for; it cannot stop the other strategy from having already
spent the cash. That is why the allocations must sum to no more than the
account, and why `equity-preflight` and `preflight` both report their sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from alphagate.core.errors import InvariantViolation

__all__ = ["Sleeve", "residual_sleeve"]


@dataclass(frozen=True, slots=True)
class Sleeve:
    """One strategy's pool of money, and the P&L it has made on it.

    `equity` is what every percentage limit in `limits.py` and `policy.py` is a
    fraction of, and what the kill switch measures its high-water mark against.
    """

    name: str
    """What the journal, the dashboard and the preflight call this pool.

    Required rather than defaulted: two sleeves reported under the same label
    is a status page that cannot be read, and the label is the only thing
    distinguishing two otherwise identical numbers."""

    allocation: Decimal
    """The capital assigned to this strategy, fixed at configuration.

    **Not a fraction of live account equity.** A fraction would reintroduce
    exactly the coupling this module removes: the other strategy's overnight
    mark would resize this strategy's budgets. The operator splits the account
    once, in configuration, and the split does not move because the market did.
    """

    realised: Decimal = Decimal(0)
    """Closed round-trips in this sleeve, summed from the journal's outcome
    amendments (`outcome.realised_pl`). Positive is profit."""

    unrealised: Decimal = Decimal(0)
    """Mark-to-market on this sleeve's open positions, as the broker reports it.

    Carried separately from `realised` rather than added on the way in, because
    specs/07 D7 reports the two separately and a type that has already summed
    them cannot."""

    def __post_init__(self) -> None:
        if not self.name:
            raise InvariantViolation("sleeve name must not be empty")
        for field_name in ("allocation", "realised", "unrealised"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(
                    f"{field_name} must be Decimal, got {type(value).__name__}; "
                    "money is Decimal end to end (specs/01 Rule 4)"
                )
            if not value.is_finite():
                raise InvariantViolation(f"{field_name} must be finite, got {value}")
        if self.allocation <= 0:
            raise InvariantViolation(
                f"allocation must be positive, got {self.allocation}; an allocation "
                "of zero is not a conservative sleeve, it is a disabled system"
            )

    @property
    def equity(self) -> Decimal:
        """What this sleeve is worth now. Never negative.

        The floor is not defensive decoration. Every budget in `limits.py` is a
        *fraction* of this number, so a negative equity would flip the sign of
        each one and permit more the further underwater the sleeve got. Defined
        risk makes that unreachable today; a limit that depends on a strategy
        invariant to stay sane is one that breaks the day the strategy changes.
        """
        return max(Decimal(0), self.allocation + self.realised + self.unrealised)

    def drawdown(self, *, peak: Decimal | None) -> Decimal:
        """Peak-to-current, as a fraction of the peak — the kill switch's input.

        `peak` is the caller's to carry across days, exactly as it was when this
        measured the account: a pure function cannot remember yesterday, and a
        high-water mark that resets each morning is a kill switch that cannot
        latch across the days that matter.

        `None` means nobody has a history yet and the answer is zero — a first
        run must not fire the switch.
        """
        if peak is None or peak <= 0:
            return Decimal(0)
        current = self.equity
        if current >= peak:
            return Decimal(0)
        return (peak - current) / peak


def residual_sleeve(
    name: str,
    *,
    allocation: Decimal,
    account_equity: Decimal,
    others: tuple[Sleeve, ...],
) -> Sleeve:
    """The sleeve that owns everything the named sleeves do not.

    One strategy in the account can be measured bottom-up — its allocation plus
    the P&L of positions we can identify as its own. Whatever is left over is,
    by definition, the other one:

        residual.equity = account_equity - sum(other.equity for other in others)

    That identity is what makes the isolation exact rather than approximate. If
    the options sleeve loses $1,000, the account falls $1,000 *and* the options
    sleeve's own equity falls $1,000, so the residual is unchanged — the equity
    book does not absorb a loss it did not take. The same holds in reverse: an
    $8,000 fall in the stock book leaves every `other` untouched, so the whole
    of it lands here, where it belongs.

    **Why the equity book is the residual and the options agent is not.** The
    options sleeve's realised and unrealised are separately identifiable — its
    positions are option contracts and its round-trips carry `realised_pl` — and
    specs/07 D7 requires them reported apart. The residual cannot split its own
    P&L from the account alone, so it reports one figure. The sleeve whose split
    matters is the one computed bottom-up, which is the right way round.

    The residual's P&L is carried in `unrealised` because that is the honest of
    the two: it is a mark, it moves every tick, and none of it has been proven
    by a closed round-trip.
    """
    claimed = sum((sleeve.equity for sleeve in others), Decimal(0))
    return Sleeve(
        name=name,
        allocation=allocation,
        unrealised=account_equity - claimed - allocation,
    )
