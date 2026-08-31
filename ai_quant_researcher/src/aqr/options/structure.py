"""Defined-risk structures, and the money they make or lose — specs/10 D4.

The unit this system proposes, sizes and measures is a **structure**, never a
bare leg. That is what makes "no naked options" enforceable by construction
rather than by review, and it is the same decision
[02-options-domain.md](../../specs/02-options-domain.md) D3 reached
independently on the execution side.

Two things here are worth more attention than the rest.

**Maximum loss is computed, not looked up.** A formula per kind would be seven
chances to be wrong with no way to notice, and the number is the denominator of
``fixed_risk`` sizing — get it wrong by the width of a spread and the whole book
is sized wrong by the same factor, invisibly. The payoff of any combination of
options on one expiry is piecewise linear with kinks only at the strikes, so its
minimum is at a strike or at a boundary. Evaluating there is exact, and it is
the same code for every kind including ones nobody has added yet.

**Sign convention, once, everywhere.** Cash in is positive, cash out is
negative. ``entry_cash + settlement_value(spot)`` is the P&L per share. A
flipped sign here turns every loss into a gain, so it is pinned by a test rather
than by this paragraph.

Prices are per share. The 100 multiplier belongs to the position, not to the
price, and is applied once in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from aqr.options.chain import Quote, Right
from aqr.options.pricing import black_scholes, intrinsic

__all__ = ["Leg", "Side", "Structure", "StructureKind"]

Side = Literal["buy", "sell"]

StructureKind = Literal[
    "long_call",
    "long_put",
    "put_credit_spread",
    "call_credit_spread",
    "put_debit_spread",
    "call_debit_spread",
    "iron_condor",
]
"""Every structure the DSL may name.

There is no ``custom`` and no kind with a lone short leg. ``covered_call`` and
``cash_secured_put`` are absent deliberately: both need a stock position or a
cash reserve before they are defined risk, which makes them a statement about
the portfolio rather than about the structure.
"""


@dataclass(frozen=True, slots=True)
class Leg:
    quote: Quote
    side: Side

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1

    @property
    def cash(self) -> float:
        """Signed cash at entry: sell collects the bid, buy pays the ask.

        The full spread, always. D2 allows no mid-price fill and no partial
        cross, because the cache's p90 relative spread is 40% and a fill better
        than the quoted book is an instrument that was not traded.
        """
        return self.quote.bid if self.side == "sell" else -self.quote.ask


@dataclass(frozen=True, slots=True)
class Structure:
    """One defined-risk position, opened at one session's quotes."""

    kind: StructureKind
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError(f"{self.kind}: a structure needs at least one leg")
        if len({leg.quote.expiration for leg in self.legs}) != 1:
            raise ValueError(
                f"{self.kind}: every leg must share one expiry; a calendar spread cannot "
                f"be priced here, because the near leg needs a quote on the far leg's "
                f"expiration date and the cache does not carry one"
            )
        _validate_shape(self.kind, self.legs)
        for leg in self.legs:
            if leg.side == "sell" and not leg.quote.sellable:
                raise ValueError(
                    f"{self.kind}: the short {leg.quote.strike:g} {leg.quote.right} has a "
                    f"zero bid; opening it is a fill at nothing, not a trade"
                )
        if self._unbounded_above():
            raise ValueError(
                f"{self.kind}: net short calls make the loss unbounded above. "
                f"No structure this system can size may have that shape."
            )
        if self.max_loss <= 0:
            raise ValueError(
                f"{self.kind}: computed maximum risk is {self.max_loss:.4f}, which is not "
                f"a positive number. On this cache that means a stale or crossed quote "
                f"rather than free money, and fixed_risk sizing would divide by it."
            )

    # ----------------------------------------------------------------- shape #

    @property
    def expiration(self) -> date:
        return self.legs[0].quote.expiration

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(sorted(leg.quote.strike for leg in self.legs))

    # ------------------------------------------------------------------ money #

    @property
    def entry_cash(self) -> float:
        """Signed cash per share at entry. Positive for a credit structure."""
        return sum(leg.cash for leg in self.legs)

    def settlement_value(self, spot: float) -> float:
        """Signed value per share at expiry, against the underlying's close.

        European: the position is worth its intrinsic value and nothing else.
        Early assignment is not modelled, which biases short ITM legs
        optimistically and is recorded with every result (D1).
        """
        return sum(
            leg.sign * intrinsic(leg.quote.right, spot=spot, strike=leg.quote.strike)
            for leg in self.legs
        )

    @property
    def max_loss(self) -> float:
        """The worst P&L per share this position can produce, as a positive number.

        Evaluated at every kink and both boundaries rather than derived per
        kind. ``0.0`` is returned for a position that cannot lose, and
        ``__post_init__`` refuses to build one.
        """
        top = max(self.strikes)
        candidates = (0.0, *self.strikes, top * 2.0)
        worst = min(self.entry_cash + self.settlement_value(spot) for spot in candidates)
        return max(-worst, 0.0)

    def mark(self, *, spot: float, years: float, rate: float = 0.0) -> float:
        """Signed model value per share — the equity curve only (D1a).

        Never P&L, never an exit price. IV is the one the contract was quoted at
        on entry, held constant, so the mark carries delta and gamma and does
        not carry vega. That understates the drawdown of a short-premium book in
        exactly the episodes that matter, and it is named here and in every
        result rather than left for a reader to discover.
        """
        return sum(
            leg.sign
            * black_scholes(
                leg.quote.right,
                spot=spot,
                strike=leg.quote.strike,
                years=years,
                iv=leg.quote.iv,
                rate=rate,
            )
            for leg in self.legs
        )

    def _unbounded_above(self) -> bool:
        """Net short calls: the payoff falls without limit as spot rises.

        The put side needs no equivalent check — as spot goes to zero the payoff
        converges on a finite number for any combination of puts.
        """
        return sum(leg.sign for leg in self.legs if leg.quote.right == "call") < 0


# --------------------------------------------------------------------------- #
# Shape validation
# --------------------------------------------------------------------------- #


def _validate_shape(kind: StructureKind, legs: tuple[Leg, ...]) -> None:
    if kind in ("long_call", "long_put"):
        right: Right = "call" if kind == "long_call" else "put"
        if len(legs) != 1 or legs[0].side != "buy" or legs[0].quote.right != right:
            raise ValueError(f"{kind}: exactly one bought {right}")
        return

    if kind == "iron_condor":
        if len(legs) != 4:
            raise ValueError("iron_condor: four legs — a put spread and a call spread")
        puts = tuple(leg for leg in legs if leg.quote.right == "put")
        calls = tuple(leg for leg in legs if leg.quote.right == "call")
        if len(puts) != 2 or len(calls) != 2:
            raise ValueError("iron_condor: two puts and two calls")
        _validate_shape("put_credit_spread", puts)
        _validate_shape("call_credit_spread", calls)
        return

    right = "put" if kind.startswith("put_") else "call"
    if len(legs) != 2:
        raise ValueError(f"{kind}: two legs, not {len(legs)}")
    if any(leg.quote.right != right for leg in legs):
        raise ValueError(f"{kind}: both legs must be {right}s")
    sold = [leg for leg in legs if leg.side == "sell"]
    bought = [leg for leg in legs if leg.side == "buy"]
    if len(sold) != 1 or len(bought) != 1:
        raise ValueError(f"{kind}: one leg bought and one sold")
    if sold[0].quote.strike == bought[0].quote.strike:
        raise ValueError(f"{kind}: the two legs must have different strikes")

    # Which side is sold is the whole identity of the structure. A put spread
    # that sells the *lower* strike is a debit spread wearing a credit spread's
    # name: opposite directional bet, different maximum loss, same YAML.
    sells_higher = sold[0].quote.strike > bought[0].quote.strike
    wants_higher = kind in ("put_credit_spread", "call_debit_spread")
    if sells_higher != wants_higher:
        wanted = "higher strike" if wants_higher else "lower strike"
        raise ValueError(
            f"{kind}: must sell the {wanted}; selling the other one is a different "
            f"structure with a different maximum loss"
        )
