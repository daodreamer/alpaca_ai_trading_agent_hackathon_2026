"""Structures — specs/02 D3.

    "The unit the agent proposes and the Gate judges is a structure, never a
    bare leg. This is the mechanism that makes 'no naked options' enforceable by
    construction rather than by review."

Everything in this module serves that sentence. There is no `CUSTOM` kind and no
naked-short kind, so a naked short is not a thing the Gate refuses — it is a
thing with no representation. The validation below is deliberately strict:
every rejection is a position whose loss is unbounded, mislabelled, or
unstateable, and each of those would defeat the Gate's `defined_risk` check
before that check ever ran.

Two subtleties worth naming:

*Credit and debit are structural, not observational.* Which side is short
determines whether a vertical takes in premium, and that is knowable from the
strikes alone. Validating it here means `StructureRisk` can trust the label
rather than re-deriving it from quotes that might be missing.

*Leg order is not identity.* Two orderings of the same legs are the same
structure, so legs are normalised at construction. Otherwise the journal would
show two different fingerprints for one position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.options.contract import OptionContract, Right, Side

__all__ = ["Cover", "Leg", "OptionStructure", "StructureKind"]


class StructureKind(Enum):
    """The complete set. Adding a member is a change to the safety argument."""

    VERTICAL_CREDIT = "vertical_credit"
    VERTICAL_DEBIT = "vertical_debit"
    IRON_CONDOR = "iron_condor"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"


@dataclass(frozen=True, slots=True)
class Leg:
    """One contract, one side, a positive quantity.

    Direction lives in ``side`` and nowhere else. Allowing a negative quantity
    would give two ways to express "short", and two representations of one fact
    eventually disagree.
    """

    contract: OptionContract
    side: Side
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InvariantViolation(
                f"leg quantity must be positive, got {self.quantity}; "
                "direction belongs in `side`"
            )

    @property
    def signed_contracts(self) -> int:
        return self.side.signum * self.quantity

    def _sort_key(self) -> tuple[str, str, int, str]:
        return (
            self.contract.expiry.isoformat(),
            self.contract.right.value,
            self.contract.strike_thousandths,
            self.side.value,
        )


@dataclass(frozen=True, slots=True)
class Cover:
    """Evidence that a short leg is not naked.

    ``basis`` is required for a covered call because without it the maximum loss
    cannot be stated: the position loses the whole value of the stock if the
    underlying goes to zero, and "the whole value" depends on what was paid.
    Refusing to construct is better than reporting a loss figure that is really
    a guess.
    """

    shares: int = 0
    basis: Decimal | None = None
    cash: Decimal | None = None

    def __post_init__(self) -> None:
        if self.shares < 0:
            raise InvariantViolation(f"cover shares must not be negative, got {self.shares}")
        for name in ("basis", "cash"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, Decimal):
                raise InvariantViolation(
                    f"cover {name} must be Decimal, not {type(value).__name__}"
                )
            if not value.is_finite():
                raise InvariantViolation(f"cover {name} must be finite, got {value}")
            if value < 0:
                raise InvariantViolation(f"cover {name} must not be negative, got {value}")


@dataclass(frozen=True, slots=True)
class OptionStructure:
    """A defined-risk options position. Constructible only in a valid shape."""

    kind: StructureKind
    legs: tuple[Leg, ...]
    cover: Cover | None = None
    _sorted: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.legs:
            raise InvariantViolation(f"{self.kind.name} needs at least one leg")

        # Normalise leg order so identity is the set of legs, not their sequence.
        ordered = tuple(sorted(self.legs, key=Leg._sort_key))
        if ordered != self.legs:
            object.__setattr__(self, "legs", ordered)

        underlyings = {leg.contract.underlying for leg in self.legs}
        if len(underlyings) != 1:
            raise InvariantViolation(
                f"{self.kind.name} legs span several underlyings: {sorted(underlyings)}"
            )
        expiries = {leg.contract.expiry for leg in self.legs}
        if len(expiries) != 1:
            raise InvariantViolation(
                f"{self.kind.name} legs span several expiries: {sorted(expiries)}; "
                "a calendar spread has different risk and is not this kind"
            )
        quantities = {leg.quantity for leg in self.legs}
        if len(quantities) != 1:
            raise InvariantViolation(
                f"{self.kind.name} legs have unequal quantities {sorted(quantities)}; "
                "a ratio spread is unbounded on one side"
            )

        validator = {
            StructureKind.VERTICAL_CREDIT: self._validate_vertical_credit,
            StructureKind.VERTICAL_DEBIT: self._validate_vertical_debit,
            StructureKind.IRON_CONDOR: self._validate_iron_condor,
            StructureKind.COVERED_CALL: self._validate_covered_call,
            StructureKind.CASH_SECURED_PUT: self._validate_cash_secured_put,
        }[self.kind]
        validator()

    # ----------------------------------------------------------------- #
    # Shape validation
    # ----------------------------------------------------------------- #

    def _vertical_pair(self, label: str) -> tuple[Leg, Leg]:
        if len(self.legs) != 2:
            raise InvariantViolation(f"{label} needs exactly two legs, got {len(self.legs)}")
        rights = {leg.contract.right for leg in self.legs}
        if len(rights) != 1:
            names = sorted(r.name for r in rights)
            raise InvariantViolation(f"{label} legs must share a right, got {names}")
        sides = {leg.side for leg in self.legs}
        if sides != {Side.BUY, Side.SELL}:
            raise InvariantViolation(f"{label} needs one long leg and one short leg")
        short = next(leg for leg in self.legs if leg.side is Side.SELL)
        long = next(leg for leg in self.legs if leg.side is Side.BUY)
        if short.contract.strike == long.contract.strike:
            raise InvariantViolation(
                f"{label} legs share strike {short.contract.strike}; "
                "zero width is a naked position wearing two legs"
            )
        return short, long

    def _validate_vertical_credit(self) -> None:
        short, long = self._vertical_pair("a vertical credit spread")
        right = short.contract.right
        takes_credit = (
            short.contract.strike < long.contract.strike
            if right is Right.CALL
            else short.contract.strike > long.contract.strike
        )
        if not takes_credit:
            raise InvariantViolation(
                f"these legs form a debit, not a credit: short {right.name} "
                f"{short.contract.strike} against long {long.contract.strike}"
            )

    def _validate_vertical_debit(self) -> None:
        short, long = self._vertical_pair("a vertical debit spread")
        right = long.contract.right
        pays_debit = (
            long.contract.strike < short.contract.strike
            if right is Right.CALL
            else long.contract.strike > short.contract.strike
        )
        if not pays_debit:
            raise InvariantViolation(
                f"these legs form a credit, not a debit: long {right.name} "
                f"{long.contract.strike} against short {short.contract.strike}"
            )

    def _validate_iron_condor(self) -> None:
        if len(self.legs) != 4:
            raise InvariantViolation(
                f"an iron condor needs exactly four legs, got {len(self.legs)}"
            )
        puts = [leg for leg in self.legs if leg.contract.right is Right.PUT]
        calls = [leg for leg in self.legs if leg.contract.right is Right.CALL]
        if len(puts) != 2 or len(calls) != 2:
            raise InvariantViolation("an iron condor needs two puts and two calls")

        put_short = self._one(puts, Side.SELL, "put")
        put_long = self._one(puts, Side.BUY, "put")
        call_short = self._one(calls, Side.SELL, "call")
        call_long = self._one(calls, Side.BUY, "call")

        if put_long.contract.strike >= put_short.contract.strike:
            raise InvariantViolation(
                "iron condor put wing is inverted: the long put must sit below the short put"
            )
        if call_short.contract.strike >= call_long.contract.strike:
            raise InvariantViolation(
                "iron condor call wing is inverted: the long call must sit above the short call"
            )
        if put_short.contract.strike >= call_short.contract.strike:
            raise InvariantViolation(
                "iron condor short strikes cross: the short put must sit below the short call"
            )

    @staticmethod
    def _one(legs: list[Leg], side: Side, label: str) -> Leg:
        matching = [leg for leg in legs if leg.side is side]
        if len(matching) != 1:
            raise InvariantViolation(
                f"an iron condor needs exactly one {side.value} {label}, got {len(matching)}"
            )
        return matching[0]

    def _validate_covered_call(self) -> None:
        if len(self.legs) != 1:
            raise InvariantViolation(f"a covered call is one leg, got {len(self.legs)}")
        leg = self.legs[0]
        if leg.contract.right is not Right.CALL or leg.side is not Side.SELL:
            raise InvariantViolation("a covered call is a short call")
        needed = leg.quantity * leg.contract.multiplier
        if self.cover is None or self.cover.shares < needed:
            held = 0 if self.cover is None else self.cover.shares
            raise InvariantViolation(
                f"covered call needs cover of {needed} shares, holds {held}; "
                "an uncovered short call has unbounded loss"
            )
        if self.cover.basis is None:
            raise InvariantViolation(
                "covered call cover needs a basis: without it the maximum loss cannot be stated"
            )

    def _validate_cash_secured_put(self) -> None:
        if len(self.legs) != 1:
            raise InvariantViolation(f"a cash-secured put is one leg, got {len(self.legs)}")
        leg = self.legs[0]
        if leg.contract.right is not Right.PUT or leg.side is not Side.SELL:
            raise InvariantViolation("a cash-secured put is a short put")
        needed = leg.contract.strike * leg.contract.multiplier * leg.quantity
        held = Decimal(0) if self.cover is None or self.cover.cash is None else self.cover.cash
        if held < needed:
            raise InvariantViolation(
                f"cash-secured put needs cover of {needed}, holds {held}; "
                "assignment must be fundable"
            )

    # ----------------------------------------------------------------- #
    # Derived facts
    # ----------------------------------------------------------------- #

    @property
    def underlying(self) -> Ticker:
        return self.legs[0].contract.underlying

    @property
    def expiry(self) -> date:
        return self.legs[0].contract.expiry

    @property
    def quantity(self) -> int:
        """Contracts per leg. Equal across legs by construction."""
        return self.legs[0].quantity

    @property
    def multiplier(self) -> int:
        return self.legs[0].contract.multiplier

    @property
    def short_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.side is Side.SELL)

    @property
    def long_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.side is Side.BUY)

    @property
    def width(self) -> Decimal:
        """Strike distance that bounds the loss, per share.

        For a condor this is the *wider* wing: only one side can finish in the
        money, so the worse of the two is what the position risks.
        """
        if self.kind is StructureKind.IRON_CONDOR:
            puts = sorted(
                leg.contract.strike for leg in self.legs if leg.contract.right is Right.PUT
            )
            calls = sorted(
                leg.contract.strike for leg in self.legs if leg.contract.right is Right.CALL
            )
            return max(puts[1] - puts[0], calls[1] - calls[0])
        if self.kind in (StructureKind.VERTICAL_CREDIT, StructureKind.VERTICAL_DEBIT):
            strikes = sorted(leg.contract.strike for leg in self.legs)
            return strikes[1] - strikes[0]
        if self.kind is StructureKind.CASH_SECURED_PUT:
            return self.legs[0].contract.strike
        # Covered call: the loss is bounded by the stock, at its basis, which
        # construction guarantees is present.
        if self.cover is None or self.cover.basis is None:  # pragma: no cover
            raise InvariantViolation("covered call without a cover basis")
        return self.cover.basis

    def days_to_expiry(self, as_of: date) -> int:
        return self.legs[0].contract.days_to_expiry(as_of)

    def __str__(self) -> str:
        return f"{self.kind.value} {self.underlying} {self.expiry:%Y-%m-%d} x{self.quantity}"
