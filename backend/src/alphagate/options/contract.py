"""Contract identity and OCC rendering — specs/02 D1.

A contract is four facts: underlying, expiry, strike, right. The OCC symbol is a
*rendering* of those facts, which is why `parse_occ` and `format_occ` are
functions here rather than a `symbol` field on the dataclass. A system that
treats the string as the identity ends up with two contracts that are the same
option and do not compare equal, and discovers it at reconciliation time.

Pure: stdlib plus `alphagate.core`. No provider types, no I/O, no clock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker

__all__ = [
    "STRIKE_PLACES",
    "OptionContract",
    "Right",
    "Side",
    "format_occ",
    "parse_occ",
]

STRIKE_PLACES: Final = 3
"""OCC encodes the strike in thousandths of a dollar, in eight digits."""

_STRIKE_SCALE: Final = Decimal(10) ** STRIKE_PLACES
_MAX_STRIKE_THOUSANDTHS: Final = 99_999_999

_OCC_PATTERN: Final = re.compile(
    r"^(?P<root>[A-Z0-9.\-]{1,6})"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<right>[CP])"
    r"(?P<strike>\d{8})$"
)


class Right(Enum):
    """Which way the option points."""

    CALL = "C"
    PUT = "P"


class Side(Enum):
    """Which way an order goes. Declared here so a leg needs no other import."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def signum(self) -> int:
        """+1 for a long leg, -1 for a short one. Used by premium arithmetic."""
        return 1 if self is Side.BUY else -1


@dataclass(frozen=True, slots=True)
class OptionContract:
    """One option contract. Frozen, hashable, exact."""

    underlying: Ticker
    expiry: date
    strike: Decimal
    right: Right
    multiplier: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.strike, Decimal):
            raise InvariantViolation(
                f"strike must be Decimal, got {type(self.strike).__name__}: money is exact "
                "and Decimal(0.1) is not 0.1"
            )
        # Finiteness before magnitude. `Decimal("NaN") <= 0` raises
        # InvalidOperation rather than answering, and an invariant check that
        # throws a different exception than the one it documents is a check the
        # caller cannot handle.
        if not self.strike.is_finite():
            raise InvariantViolation(f"strike must be finite, got {self.strike}")
        if self.strike <= 0:
            raise InvariantViolation(f"strike must be positive, got {self.strike}")
        if self.multiplier <= 0:
            raise InvariantViolation(f"multiplier must be positive, got {self.multiplier}")

        # The strike must survive the wire format. A contract whose strike cannot
        # be written as an OCC symbol cannot be ordered, so it must not exist.
        scaled = self.strike * _STRIKE_SCALE
        if scaled != scaled.to_integral_value():
            raise InvariantViolation(
                f"strike {self.strike} is finer than a tenth of a cent and does not "
                "round-trip through the OCC thousandths encoding"
            )
        if scaled > _MAX_STRIKE_THOUSANDTHS:
            raise InvariantViolation(
                f"strike {self.strike} exceeds the eight-digit OCC encoding"
            )

    @property
    def strike_thousandths(self) -> int:
        return int(self.strike * _STRIKE_SCALE)

    def days_to_expiry(self, as_of: date) -> int:
        """Calendar days until expiry. Negative once expired.

        Calendar rather than trading days on purpose: an option's time value
        decays over weekends, and the Gate's DTE window is a claim about
        wall-clock exposure.
        """
        return (self.expiry - as_of).days

    def __str__(self) -> str:
        return format_occ(self)


def format_occ(contract: OptionContract) -> str:
    """Render a contract as an OCC symbol, e.g. ``AAPL260918C00150000``."""
    return (
        f"{contract.underlying}"
        f"{contract.expiry:%y%m%d}"
        f"{contract.right.value}"
        f"{contract.strike_thousandths:08d}"
    )


def parse_occ(symbol: str, *, multiplier: int = 100) -> OptionContract:
    """Recover a contract from an OCC symbol.

    ``multiplier`` is a parameter because the symbol does not carry it: adjusted
    contracts share the encoding and differ only in deliverable. Defaulting to
    100 is right for everything this system trades, and being explicit about the
    gap is better than pretending the string said something it did not.
    """
    candidate = symbol.strip().upper()
    match = _OCC_PATTERN.match(candidate)
    if match is None:
        raise InvariantViolation(f"malformed OCC symbol: {symbol!r}")

    try:
        expiry = date(
            2000 + int(match["yy"]), int(match["mm"]), int(match["dd"])
        )
    except ValueError as exc:
        raise InvariantViolation(f"malformed OCC symbol {symbol!r}: {exc}") from exc

    try:
        strike = Decimal(int(match["strike"])).scaleb(-STRIKE_PLACES)
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - regex guards this
        raise InvariantViolation(f"malformed OCC strike in {symbol!r}") from exc

    return OptionContract(
        underlying=ticker(match["root"]),
        expiry=expiry,
        strike=strike,
        right=Right(match["right"]),
        multiplier=multiplier,
    )
