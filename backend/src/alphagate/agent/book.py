"""The open book — what the Gate is allowed to know about the account.

Between `execution/account.py` (which reads the broker) and `risk` (which reads
a `PortfolioSnapshot`) there is a gap nothing else fills, and it is a gap with a
real problem in it.

**Alpaca has no concept of a structure.** The positions endpoint returns option
*legs*: four lines with a sign and a quantity, not "one iron condor". The Gate,
meanwhile, budgets by structure — `max_open_structures`, exposure per underlying,
a net greeks budget — because those are the units risk is actually taken in.
Something has to turn one into the other.

Pairing legs by inspection is the obvious approach and it is wrong. Two put
credit spreads sharing a short strike, or a condor whose call side was closed
early, are indistinguishable from several other shapes if all you have is a list
of legs and some arithmetic. Guessing produces a book that looks tidy and is
false, and a false book means the Gate budgets against risk that is not there or
misses risk that is.

So: **the broker says what is open, the journal says what it was.** Every
structure we ever opened is in the journal with its `max_loss` and its greeks
(specs/06 D2). A journalled structure is still open if its legs are still at the
broker with the right signs. Nothing is inferred; two records are matched.

**Legs the journal cannot explain are reported, never modelled.** They mean
something happened outside this agent — a manual trade, a leg assigned, a fill
from a session whose journal is missing. `unexplained` carries them out to the
caller, and `PortfolioSnapshot` is built without them. An agent that quietly
absorbed an unknown short leg into its risk model would be reporting a defined
risk it had not defined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from alphagate.core.identifiers import ticker
from alphagate.execution import AccountRead, LegPosition
from alphagate.options import (
    Cover,
    Greeks,
    Leg,
    OptionContract,
    OptionStructure,
    Right,
    Side,
    StructureKind,
)
from alphagate.risk import OpenPosition, PortfolioSnapshot

__all__ = [
    "BookRead",
    "HeldPosition",
    "contract_from",
    "open_positions",
    "read_book",
    "structure_from",
]


@dataclass(frozen=True, slots=True)
class HeldPosition:
    """One open structure, plus the two facts an exit needs and the Gate does not.

    `OpenPosition` is a risk-layer type and stays that shape: the Gate budgets by
    maximum loss and greeks and has no business knowing what a position cost.
    Exits do — `evaluate_exit` compares what it would take to be flat now against
    the premium taken in — and neither the broker nor the position itself
    remembers that. The journal does, which is why this pairing happens here.
    """

    position: OpenPosition
    entry_premium: Decimal
    """The premium at open, in the domain sign convention (specs/02 D4): a credit
    received is positive. Read from the journalled fill, never re-derived from
    today's chain — that would compare the position against itself."""
    cycle_id: str
    """The decision that opened it. An exit amends the same journal line."""


@dataclass(frozen=True, slots=True)
class BookRead:
    """The snapshot, and everything the caller needs to distrust it."""

    snapshot: PortfolioSnapshot
    held: tuple[HeldPosition, ...] = ()
    """The same positions as `snapshot.positions`, in the same order, with their
    entry premium and originating cycle attached."""
    unexplained: tuple[LegPosition, ...] = ()
    """Option legs at the broker that no journalled structure accounts for.

    Not folded into the snapshot. See the module docstring — this is the one
    thing that must reach a human rather than a risk model."""
    closed: tuple[str, ...] = ()
    """Cycle ids whose structures are no longer at the broker: filled and since
    closed, expired, or assigned. Reported so a session that resumes mid-day can
    tell "closed" from "never opened"."""

    @property
    def is_clean(self) -> bool:
        return not self.unexplained

    def summary(self) -> str:
        parts = [
            f"equity {self.snapshot.equity}",
            f"{self.snapshot.open_structures} open",
            f"risk {self.snapshot.open_risk}",
        ]
        if self.unexplained:
            parts.append(f"{len(self.unexplained)} UNEXPLAINED legs")
        return ", ".join(parts)


def read_book(
    account: AccountRead,
    legs: Sequence[LegPosition],
    journal_records: Sequence[Mapping[str, Any]],
    *,
    peak_equity: Decimal | None = None,
    fills_today: int = 0,
    killswitch_tripped: bool = False,
) -> BookRead:
    """Assemble the Gate's view of the account. Pure.

    `peak_equity` is the high-water mark the caller carries across days. Passing
    `None` means "no history", and drawdown comes back zero — correct on the
    first run and dangerous to assume afterwards, which is why it has to be
    passed rather than defaulted quietly. specs/03 D4's kill switch watches this
    number; a drawdown that resets every morning is a kill switch that cannot
    latch across the days that matter.
    """
    held, unexplained, closed = open_positions(legs, journal_records)
    peak = peak_equity if peak_equity is not None else account.equity
    drawdown = _drawdown(peak, account.equity)
    return BookRead(
        snapshot=PortfolioSnapshot(
            equity=account.equity,
            positions=tuple(item.position for item in held),
            drawdown_pct=drawdown,
            fills_today=fills_today,
            killswitch_tripped=killswitch_tripped,
        ),
        held=held,
        unexplained=unexplained,
        closed=closed,
    )


def open_positions(
    legs: Sequence[LegPosition],
    journal_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[HeldPosition, ...], tuple[LegPosition, ...], tuple[str, ...]]:
    """Match broker legs against journalled structures.

    Walks the journal newest-first so that when two cycles opened the same
    structure, the most recent one claims the legs. Each leg is claimed at most
    once: a structure whose legs another structure already took is treated as
    closed, not as a second copy of the same risk.
    """
    available: dict[OptionContract, LegPosition] = {leg.contract: leg for leg in legs}
    positions: list[HeldPosition] = []
    closed: list[str] = []

    for record in reversed(list(journal_records)):
        if str(record.get("stage", "")) != "filled":
            continue
        proposal = record.get("proposal")
        if not isinstance(proposal, Mapping):
            continue
        structure = _structure_or_none(proposal.get("structure"))
        if structure is None:
            continue
        wanted = tuple(structure.legs)
        if not _all_present(wanted, available):
            closed.append(str(record.get("cycle_id", "")))
            continue
        for leg in wanted:
            available.pop(leg.contract, None)
        positions.append(
            HeldPosition(
                position=OpenPosition(
                    structure=structure,
                    quantity=_int(proposal.get("quantity"), default=1),
                    max_loss=_max_loss(proposal),
                    net_greeks=_greeks_or_none(proposal),
                    opened_at=_opened_at(record),
                ),
                entry_premium=_entry_premium(proposal),
                cycle_id=str(record.get("cycle_id", "")),
            )
        )

    positions.reverse()
    closed.reverse()
    return tuple(positions), tuple(available.values()), tuple(closed)


def structure_from(payload: Mapping[str, Any]) -> OptionStructure:
    """Rebuild an `OptionStructure` from its journalled form.

    The journal stores the domain type itself rather than a mirror of it
    (specs/06 D2), so this is a decode of the same fields — and it goes back
    through `OptionStructure.__post_init__`, which means a structure that would
    not be constructible today raises here rather than being reconstituted into
    a book. A journal line from before a rule tightened is not a licence to hold
    a position the rule now forbids.
    """
    legs = tuple(_leg_from(item) for item in payload.get("legs", ()))
    return OptionStructure(
        kind=StructureKind(str(payload["kind"])),
        legs=legs,
        cover=_cover_from(payload.get("cover")),
    )


def contract_from(payload: Mapping[str, Any]) -> OptionContract:
    return OptionContract(
        underlying=ticker(str(payload["underlying"])),
        expiry=_date(payload["expiry"]),
        strike=Decimal(str(payload["strike"])),
        right=Right(str(payload["right"])),
        multiplier=_int(payload.get("multiplier"), default=100),
    )


# ------------------------------------------------------------------ #
# Decoding helpers. Every one of them fails closed.
# ------------------------------------------------------------------ #


def _leg_from(payload: Mapping[str, Any]) -> Leg:
    return Leg(
        contract=contract_from(payload["contract"]),
        side=Side(str(payload["side"])),
        quantity=_int(payload.get("quantity"), default=1),
    )


def _cover_from(payload: Any) -> Cover | None:
    if not isinstance(payload, Mapping):
        return None
    return Cover(
        shares=_int(payload.get("shares"), default=0),
        basis=_optional_money(payload.get("basis")),
        cash=_optional_money(payload.get("cash")),
    )


def _structure_or_none(payload: Any) -> OptionStructure | None:
    """A structure we cannot decode is not a structure we can hold.

    Returning `None` rather than raising: one unreadable line from an older
    journal format must not stop the session from reading the rest of the book.
    The leg it would have claimed then shows up in `unexplained`, which is the
    loud path, so nothing is lost quietly.
    """
    if not isinstance(payload, Mapping):
        return None
    try:
        return structure_from(payload)
    except (KeyError, ValueError, ArithmeticError, TypeError):
        return None


def _all_present(legs: Sequence[Leg], available: Mapping[OptionContract, LegPosition]) -> bool:
    """Every leg still open at the broker, on the side we opened it.

    The side check is the point. A short put we believe we hold and the broker
    reports as long is not our position; treating it as one would net a risk
    figure against an exposure pointing the other way.
    """
    for leg in legs:
        position = available.get(leg.contract)
        if position is None:
            return False
        if (leg.side is Side.SELL) != position.is_short:
            return False
    return True


def _max_loss(proposal: Mapping[str, Any]) -> Decimal:
    """Remaining maximum loss for the whole position.

    `StructureRisk.max_loss` is per unit, so this multiplies by quantity — the
    same arithmetic `ProposalContext` does before the Gate sees it. Getting this
    wrong understates the book by a factor of the position size, which is the
    kind of error that only shows up once it matters.
    """
    risk = proposal.get("risk")
    per_unit = Decimal(0)
    if isinstance(risk, Mapping):
        per_unit = _optional_money(risk.get("max_loss")) or Decimal(0)
    total = per_unit * _int(proposal.get("quantity"), default=1)
    return total if total > 0 else Decimal("0.01")


def _entry_premium(proposal: Mapping[str, Any]) -> Decimal:
    """What the position took in — or paid — when it opened. **Per unit.**

    Deliberately *not* scaled by quantity, unlike `max_loss` two functions up,
    and the asymmetry is the whole subtlety of this file.

    Candidate structures are built at one contract a leg (`candidates.py`), so a
    journalled `StructureRisk` is per unit and the size lives separately in
    `TradeProposal.quantity`. Two consumers then read them differently:

    * the Gate multiplies — `risk.max_loss * proposal.quantity` — so
      `OpenPosition.max_loss`, which is documented as whole-position, has to be
      scaled here;
    * `evaluate_exit` multiplies too — `(entry - mark) * position.quantity` — so
      scaling here as well would double-count, and a ten-lot would report ten
      times its real unrealised P&L.

    Both are right; they just want different halves. Getting this backwards was
    caught by wiring exits up and watching a three-lot decline to close at a
    mark where a one-lot closed.
    """
    risk = proposal.get("risk")
    if not isinstance(risk, Mapping):
        return Decimal(0)
    return _optional_money(risk.get("net_premium")) or Decimal(0)


def _greeks_or_none(proposal: Mapping[str, Any]) -> Greeks | None:
    risk = proposal.get("risk")
    if not isinstance(risk, Mapping):
        return None
    raw = risk.get("net_greeks")
    if not isinstance(raw, Mapping):
        return None
    quantity = _int(proposal.get("quantity"), default=1)
    try:
        return Greeks(
            delta=float(raw["delta"]) * quantity,
            gamma=float(raw["gamma"]) * quantity,
            theta=float(raw["theta"]) * quantity,
            vega=float(raw["vega"]) * quantity,
            rho=float(raw["rho"]) * quantity,
            iv=float(raw.get("iv", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _opened_at(record: Mapping[str, Any]) -> datetime:
    stamp = record.get("as_of")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            pass
    raise ValueError(f"journalled fill {record.get('cycle_id')!r} has no usable as_of")


def _drawdown(peak: Decimal, equity: Decimal) -> Decimal:
    if peak <= 0 or equity >= peak:
        return Decimal(0)
    return (peak - equity) / peak


def _date(value: Any) -> Any:
    from datetime import date as _d

    if isinstance(value, _d):
        return value
    return _d.fromisoformat(str(value))


def _int(value: Any, *, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return default


def _optional_money(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None
