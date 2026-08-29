"""What the equity Gate returns — specs/09 D5 and D7.

PURE. Stdlib plus `alphagate.core` and `alphagate.equity`.

The load-bearing type is `GatedEquityOrder`, and it is guarded exactly the way
`GatedOrder` is: construction is refused unless the calling frame belongs to
`alphagate.risk.equity_gate`. Two doors, two keys, one rule.

That symmetry is the point. The first version of the equity path could have
reused `GatedOrder` and saved a module — and it would have been wrong, because
`GatedOrder` carries an `OptionStructure` and the guarantee that comes with it
(defined risk, four legs, a known expiry). A share order has none of those, and
a type that can mean either is a type that proves neither.

`CheckResult` and `VetoReason` *are* shared, from `verdict.py`. Those describe a
check's finding rather than an order's shape, and the finding is the same shape
whatever was checked — which is what lets the dashboard render both tapes with
one component.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import FrameType
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.equity.plan import EquitySide
from alphagate.risk.verdict import CheckResult, VetoReason

__all__ = [
    "ApprovedEquity",
    "EquityVerdict",
    "GatedEquityOrder",
    "VetoedEquity",
]

_GATE_MODULE: Final = "alphagate.risk.equity_gate"
_MINTING_INTERNALS: Final = frozenset({__name__, "dataclasses"})


def _minted_by_the_gate() -> bool:
    """True when the nearest frame outside this module is the equity Gate."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name not in _MINTING_INTERNALS:
            return bool(name == _GATE_MODULE)
        frame = frame.f_back
    return False  # pragma: no cover - the stack always has a caller


def _refuse_forgery(what: str) -> None:
    if not _minted_by_the_gate():
        raise InvariantViolation(
            f"{what} may only be constructed inside {_GATE_MODULE}. "
            "specs/09 D7: every equity order reaching Alpaca passed the equity "
            "Gate. There is no bypass, no force flag and no debug path — if you "
            "need an approved order, call risk.equity_gate.evaluate_equity and "
            "get one honestly."
        )


@dataclass(frozen=True, slots=True)
class GatedEquityOrder:
    """A share order that passed every check. The only thing `submit_equity` accepts.

    `shares` is always positive and the direction lives in `side` — the same
    convention the planner uses, kept unchanged across the Gate so no sign is
    flipped anywhere between the plan and the wire.

    `reference_price` is not a limit price. These go as market orders (specs/09
    D7); the price is carried so the journal can record what the size was
    computed from and reconciliation can compare it against the fill.
    """

    symbol: Ticker
    side: EquitySide
    shares: Decimal
    reference_price: Decimal
    fractionable: bool
    fingerprint: str
    """The strategy this order serves. Travels onto the wire as part of the
    idempotency key, so two books cannot mint the same client order id."""
    book_as_of: str
    approved_at: datetime

    def __post_init__(self) -> None:
        _refuse_forgery("GatedEquityOrder")

    @property
    def notional(self) -> Decimal:
        return self.shares * self.reference_price


@dataclass(frozen=True, slots=True)
class ApprovedEquity:
    """The Gate allowed it. Carries the full check tape, not just the pass."""

    order: GatedEquityOrder
    checks: tuple[CheckResult, ...]
    waived: tuple[VetoReason, ...] = ()
    """Checks that failed and were waived because the order reduces risk.

    Recorded rather than silently dropped. "This sell was allowed past the
    turnover cap" is a fact somebody reading the journal at the end of a bad
    week needs to be able to find (specs/09 D5)."""

    def __post_init__(self) -> None:
        _refuse_forgery("ApprovedEquity")

    @property
    def is_approved(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class VetoedEquity:
    """The Gate refused it, and says how many ways it was wrong."""

    reasons: tuple[VetoReason, ...]
    checks: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise InvariantViolation(
                "a veto with no reason is an approval nobody admitted to"
            )

    @property
    def is_approved(self) -> bool:
        return False


type EquityVerdict = ApprovedEquity | VetoedEquity
"""Sealed union. There is no third answer, and no partial approval."""
