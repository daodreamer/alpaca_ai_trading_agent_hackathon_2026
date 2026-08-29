"""What the Gate returns — specs/03 D3.

PURE. Stdlib plus `alphagate.core` and `alphagate.options`.

The load-bearing type here is `GatedOrder`. specs/01 Rule 2 says every order
reaching Alpaca passed the Gate, and specs/04 D1 says `submit` accepts nothing
else. That claim is only worth stating if a `GatedOrder` cannot be conjured
elsewhere — so construction is refused unless the calling frame belongs to
`alphagate.risk.gate`.

Frame inspection is an unusual mechanism and it is chosen deliberately. The
alternatives are all weaker: a module-private token can be imported, a
convention can be forgotten in a hurry on day five, and a review cannot be run
by CI. This one fails loudly at the moment of misuse, in the caller's own
traceback, and it costs one stack walk per approved order.

Two consequences worth knowing before you are surprised by them at 2am:
`copy.deepcopy` and `pickle` of a `GatedOrder` raise, and so does
`dataclasses.replace`. That is correct — an order that can be cloned into a
second order is an order that can be submitted twice.

`Vetoed` has no such guard. Nothing dangerous happens downstream of a refusal,
and tests and the journal both benefit from being able to build one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import FrameType
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.options.structure import OptionStructure
from alphagate.risk.proposal import Intent

__all__ = [
    "Approved",
    "CheckResult",
    "GatedOrder",
    "Observation",
    "Verdict",
    "VetoReason",
    "Vetoed",
]

type Observation = Decimal | float | int | str | None
"""What a check saw, and what it was measured against. Rendered, never parsed."""

type Verdict = Approved | Vetoed
"""Sealed union. There is no third answer, and no partial approval."""

_GATE_MODULE: Final = "alphagate.risk.gate"
_MINTING_INTERNALS: Final = frozenset({__name__, "dataclasses"})


def _minted_by_the_gate() -> bool:
    """True when the nearest frame outside this module is the Gate itself."""
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
            "specs/01 Rule 2: every order reaching Alpaca passed the Gate. "
            "There is no bypass, no force flag and no debug path — if you need "
            "an approved order, call risk.gate.evaluate and get one honestly."
        )


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check's finding, whether it passed or not.

    `observed` and `limit` are carried on passing results too, so the dashboard
    can show the near-misses. A book sitting at 4.9% of a 5% heat limit is a
    different situation from one sitting at 0.4%, and only one of them is a
    reason to stop proposing.
    """

    name: str
    passed: bool
    detail: str
    observed: Observation = None
    limit: Observation = None


@dataclass(frozen=True, slots=True)
class VetoReason:
    """Why an order was refused. One per failing check, in check order."""

    check: str
    detail: str


@dataclass(frozen=True, slots=True)
class GatedOrder:
    """An order that passed every check. The only thing `execution` accepts.

    `limit_price` keeps the **domain** sign convention: a credit is positive.
    Alpaca inverts it, and that flip happens in exactly one named function in
    the execution adapter (specs/04 D2). Do not pre-flip it here.
    """

    structure: OptionStructure
    quantity: int
    intent: Intent
    limit_price: Decimal
    approved_at: datetime
    proposal_id: str

    def __post_init__(self) -> None:
        _refuse_forgery("GatedOrder")


@dataclass(frozen=True, slots=True)
class Approved:
    """The Gate allowed it. Carries the full check tape, not just the pass."""

    order: GatedOrder
    checks: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        _refuse_forgery("Approved")

    @property
    def is_approved(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Vetoed:
    """The Gate refused it, and says how many ways it was wrong.

    `reasons` is non-empty by construction, and `checks` is the complete tape —
    the Gate does not short-circuit. A veto with one reason and a veto with five
    are different situations, and the journal should be able to show which.
    """

    reasons: tuple[VetoReason, ...]
    checks: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise InvariantViolation("a veto with no reason is an approval nobody admitted to")

    @property
    def is_approved(self) -> bool:
        return False
