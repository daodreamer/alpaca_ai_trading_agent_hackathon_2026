"""The exact numeric domain — ADR 0005.

Two numeric domains exist in this system:

* **exact** — prices, money, quantities. `Decimal`, quantized to 8 places,
  compared with `==`. That is what this module builds.
* **approximate** — scores, ratios, oscillators, slopes. Plain `float`.
  Nothing here applies to those.

The single most important rule is that a `Decimal` is never constructed from a
`float`: `Decimal(0.1)` is `0.1000000000000000055511151231257827…`, and that
error survives every later operation. Prices arrive from providers as text, so
there is no legitimate reason for a float to be on the path. `from_approximate`
is the one supervised crossing, for converting a computed indicator value back
to the exact domain at a persistence or API boundary.
"""

from __future__ import annotations

import math
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
from typing import Final

from alphagate.core.errors import InvariantViolation

__all__ = [
    "DOMAIN_CONTEXT",
    "PRICE_PLACES",
    "ExactInput",
    "format_exact",
    "from_approximate",
    "is_quantized",
    "price",
    "quantity",
]

PRICE_PLACES: Final = 8
"""Decimal places every exact-domain value is quantized to.

Eight rather than four: sub-penny prints and provider-computed VWAP routinely
carry more than four decimals, and truncating on ingest would destroy the
ability to reproduce a provider's own aggregates.
"""

_EXPONENT: Final = Decimal(1).scaleb(-PRICE_PLACES)

DOMAIN_CONTEXT: Final = Context(
    prec=28,
    rounding=ROUND_HALF_EVEN,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
"""Explicit arithmetic context.

Set explicitly rather than inherited from the process default so that results
never depend on ambient global state — a determinism requirement (CLAUDE.md §11).
"""

type ExactInput = Decimal | int | str
"""What may become an exact value. Note the absence of `float`."""


def _coerce(value: ExactInput, *, field: str) -> Decimal:
    if isinstance(value, float):
        raise TypeError(
            f"{field}: refusing to build a Decimal from a float ({value!r}). "
            "Parse from text, or cross the boundary explicitly with "
            "from_approximate() (ADR 0005 D2)."
        )
    if isinstance(value, bool):
        # bool is a subclass of int; True would silently become 1.00000000.
        raise TypeError(f"{field}: bool is not a numeric value")
    if isinstance(value, Decimal):
        candidate = value
    else:
        try:
            candidate = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise InvariantViolation(f"{field}: not a number: {value!r}") from exc

    if not candidate.is_finite():
        raise InvariantViolation(f"{field}: must be finite, got {value!r}")
    return candidate


def _quantize(value: Decimal, *, field: str) -> Decimal:
    try:
        return value.quantize(_EXPONENT, context=DOMAIN_CONTEXT)
    except (InvalidOperation, Overflow) as exc:
        raise InvariantViolation(f"{field}: out of representable range: {value!r}") from exc


def price(value: ExactInput, *, field: str = "price") -> Decimal:
    """Coerce to an exact price, quantized to `PRICE_PLACES` with banker's rounding.

    Sign is not constrained here — deltas and offsets are prices too. Entities
    that require positivity enforce it themselves.
    """
    return _quantize(_coerce(value, field=field), field=field)


def quantity(value: ExactInput, *, field: str = "quantity") -> Decimal:
    """Coerce to an exact quantity. Same representation as a price.

    Shares are quantized to eight places rather than kept integral because
    fractional-share trading is ordinary and providers report it.
    """
    return _quantize(_coerce(value, field=field), field=field)


def from_approximate(value: float, *, field: str = "value") -> Decimal:
    """Cross from the approximate domain into the exact one, deliberately.

    Goes through `str` so the shortest round-trip representation is used and the
    binary artefacts of the float are not imported.
    """
    if not math.isfinite(value):
        raise InvariantViolation(f"{field}: must be finite, got {value!r}")
    return _quantize(Decimal(str(value)), field=field)


def format_exact(value: Decimal) -> str:
    """Render an exact value as a plain decimal string.

    `str(Decimal)` is not usable for this. A quantized zero prints as `"0E-8"`
    and very large or small magnitudes switch to scientific notation, which is
    not what ADR 0005's "prices serialize as JSON strings" means and not
    something a JS client will parse for display. Always format prices through
    this function at the API boundary.
    """
    return f"{value:.{PRICE_PLACES}f}"


def is_quantized(value: Decimal) -> bool:
    """Whether `value` already carries exactly `PRICE_PLACES` decimal places."""
    if not value.is_finite():
        return False
    return value.as_tuple().exponent == -PRICE_PLACES
