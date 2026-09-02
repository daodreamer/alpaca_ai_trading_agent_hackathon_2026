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
broker with the right signs, and in the quantity it was opened in. Nothing is
inferred; two records are matched.

**Legs the journal cannot explain are reported, never modelled.** They mean
something happened outside this agent — a manual trade, a leg assigned, a fill
from a session whose journal is missing, or simply more contracts at the broker
than the journalled fills add up to. `unexplained` carries them out to the
caller, and `PortfolioSnapshot` is built without them. An agent that quietly
absorbed an unknown short leg into its risk model would be reporting a defined
risk it had not defined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION
from alphagate.risk.sleeve import Sleeve

__all__ = [
    "BookRead",
    "HeldPosition",
    "contract_from",
    "open_positions",
    "read_book",
    "realised_pl",
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
    sleeve: Sleeve | None = None
    """The capital pool `snapshot.equity` came from — specs/03 D6.

    Carried so the status page and `preflight` can show the allocation, the
    realised and the mark-to-market separately. `snapshot.equity` has already
    summed them, and a dashboard that can only show the sum cannot answer "is
    this sleeve down because a trade lost, or because it never had the money"."""
    held: tuple[HeldPosition, ...] = ()
    """The same positions as `snapshot.positions`, in the same order, with their
    entry premium and originating cycle attached."""
    unexplained: tuple[LegPosition, ...] = ()
    """Option legs at the broker that no journalled structure accounts for — or
    the part of one that none does, carrying just the surplus quantity.

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
            f"sleeve equity {self.snapshot.equity}",
            f"{self.snapshot.open_structures} open",
            f"risk {self.snapshot.open_risk}",
        ]
        if self.unexplained:
            parts.append(f"{len(self.unexplained)} UNEXPLAINED legs")
        return ", ".join(parts)


def realised_pl(journal_records: Sequence[Mapping[str, Any]]) -> Decimal:
    """Closed round-trips, summed from the journal's outcome amendments. Pure.

    Reads `outcome.realised_pl`, the field `interface/read.py` already renders,
    so the number the dashboard shows and the number the kill switch measures
    come from one place. A record with no outcome is an open position and
    contributes nothing: specs/07 D7 keeps realised and mark-to-market apart,
    and a sum that quietly folded in unrealised marks would be the flattering
    number that spec exists to refuse.
    """
    total = Decimal(0)
    for record in journal_records:
        outcome = record.get("outcome")
        if not isinstance(outcome, Mapping):
            continue
        raw = outcome.get("realised_pl")
        if raw is None or raw == "":
            continue
        try:
            total += Decimal(str(raw))
        except (ArithmeticError, ValueError):
            # A malformed amendment is not a licence to guess at P&L. Skipping
            # it understates the sleeve, which fails towards a tighter budget.
            continue
    return total


def read_book(
    account: AccountRead,
    legs: Sequence[LegPosition],
    journal_records: Sequence[Mapping[str, Any]],
    *,
    sleeve_allocation: Decimal = OPTIONS_SLEEVE_ALLOCATION,
    peak_equity: Decimal | None = None,
    fills_today: int = 0,
    killswitch_tripped: bool = False,
) -> BookRead:
    """Assemble the Gate's view of the **options sleeve**. Pure.

    Not of the account. specs/03 D6: this system runs two strategies against one
    broker account, and every budget here is a fraction of what the options
    agent was allocated rather than of what the account happens to be worth. The
    equity book's mark-to-market does not appear in this arithmetic, so it can
    neither resize these budgets nor trip this kill switch.

    `account` is still read, because the Gate needs the broker's own view for
    things the sleeve cannot answer — options level, buying power, whether the
    account is blocked. It is no longer the base the limits scale off.

    `journal_records` should span **every day the sleeve has traded**, not just
    today. Two things depend on it: `realised_pl` is cumulative by definition,
    and `open_positions` matches broker legs against journalled fills — a spread
    opened on Monday and still held on Wednesday has no fill in Wednesday's file
    and would otherwise be reported as an unexplained leg.

    `peak_equity` is the high-water mark the caller carries across days. Passing
    `None` means "no history", and drawdown comes back zero — correct on the
    first run and dangerous to assume afterwards, which is why it has to be
    passed rather than defaulted quietly. specs/03 D4's kill switch watches this
    number; a drawdown that resets every morning is a kill switch that cannot
    latch across the days that matter.
    """
    held, unexplained, closed = open_positions(legs, journal_records)
    sleeve = Sleeve(
        name="options",
        allocation=sleeve_allocation,
        realised=realised_pl(journal_records),
        # Every option leg at the broker belongs to this sleeve, including the
        # unexplained ones. A leg we cannot account for is still capital at
        # risk, and leaving it out of the sleeve's equity would overstate what
        # is left to trade with by exactly the amount nobody can explain.
        unrealised=sum((leg.unrealised for leg in legs), Decimal(0)),
    )
    return BookRead(
        snapshot=PortfolioSnapshot(
            equity=sleeve.equity,
            positions=tuple(item.position for item in held),
            drawdown_pct=sleeve.drawdown(peak=peak_equity),
            fills_today=fills_today,
            killswitch_tripped=killswitch_tripped,
        ),
        sleeve=sleeve,
        held=held,
        unexplained=unexplained,
        closed=closed,
    )


def open_positions(
    legs: Sequence[LegPosition],
    journal_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[HeldPosition, ...], tuple[LegPosition, ...], tuple[str, ...]]:
    """Match broker legs against journalled structures.

    Walks the journal newest-first, and **the broker's quantity decides how many
    fills fit**. Each structure claims its share of each contract; the next one
    back gets what is left. So two cycles that opened the same spread are two
    positions when the broker holds two of each leg, and one position plus one
    closed cycle when it holds one — a contract cannot be two positions, and
    counting it twice would double the book's risk against one real one.

    A cycle whose stage reads `filled` is one whose order the broker confirmed,
    which for anything that did not fill instantly means `Journal.read` settled
    that from the outcome amendment (see `journal.writer._with_final_stage`).
    Matching on the decision line's own first guess would drop every
    slower-than-instant fill out of this book.

    The two ways the counts can disagree are treated differently, on purpose:

    * **The journal wants more than the broker holds** — a 3-lot against one
      contract. Claimed anyway, and the position stays open at its journalled
      size. Over-modelling is the safe direction: the position keeps its place
      in the Gate's budget and under the exit policy, where dropping it would
      leave a live short leg nobody was managing.
    * **The broker holds more than the journal explains.** The surplus comes
      back as an unexplained leg carrying only the quantity nobody accounted
      for. That is the direction where the extra is risk this agent never
      decided on, and the module docstring's rule applies: reported, never
      modelled.
    """
    holdings: dict[OptionContract, LegPosition] = {leg.contract: leg for leg in legs}
    remaining: dict[OptionContract, int] = {leg.contract: leg.quantity for leg in legs}
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
        wanted = _contracts_wanted(structure, _int(proposal.get("quantity"), default=1))
        if not _all_present(wanted, remaining):
            closed.append(str(record.get("cycle_id", "")))
            continue
        for contract, count in wanted.items():
            remaining[contract] = _after_claiming(remaining[contract], count)
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
    unexplained = tuple(
        _residual(holdings[contract], left)
        for contract, left in remaining.items()
        if left != 0
    )
    return tuple(positions), unexplained, tuple(closed)


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


def _contracts_wanted(structure: OptionStructure, quantity: int) -> dict[OptionContract, int]:
    """How many of each contract this position is, signed — long positive.

    `Leg.quantity` is the ratio inside the structure and `quantity` is how many
    structures were opened; the broker reports one signed line per contract and
    knows about neither. Multiplying here is what lets the two be compared at
    all. Legs are summed rather than assigned so a structure naming the same
    contract twice nets out instead of silently keeping the last one.
    """
    wanted: dict[OptionContract, int] = {}
    for leg in structure.legs:
        signed = leg.quantity * quantity * (-1 if leg.side is Side.SELL else 1)
        wanted[leg.contract] = wanted.get(leg.contract, 0) + signed
    return wanted


def _all_present(
    wanted: Mapping[OptionContract, int], remaining: Mapping[OptionContract, int]
) -> bool:
    """Every leg still open at the broker, on the side we opened it.

    The side check is the point. A short put we believe we hold and the broker
    reports as long is not our position; treating it as one would net a risk
    figure against an exposure pointing the other way. Comparing signs does that
    check and the "is it still there at all" check at once — nothing is left of
    a contract earlier structures used up, and zero has no side.

    A net-zero leg counts as absent. It cannot be matched against anything the
    broker reports, and treating it as trivially present would let a structure
    nobody can hold claim to be open.
    """
    for contract, count in wanted.items():
        have = remaining.get(contract, 0)
        if count == 0 or have == 0:
            return False
        if (have > 0) != (count > 0):
            return False
    return True


def _after_claiming(have: int, wanted: int) -> int:
    """What is left of a broker leg once a structure has taken its share.

    Never crosses zero. A journalled 3-lot matched against a single contract
    takes the one contract and leaves nothing; a negative remainder would come
    back out of `_residual` as a short leg nobody ever opened.
    """
    return 0 if abs(wanted) >= abs(have) else have - wanted


def _residual(leg: LegPosition, left: int) -> LegPosition:
    """The part of a broker leg no journalled structure accounts for.

    Returned untouched in the common case — nothing was claimed, so there is
    nothing to apportion and no rounding to explain. Otherwise the money is
    scaled to the share that is left: reporting the whole line's market value as
    unexplained would overstate the unaccounted-for exposure by exactly the part
    we *can* account for. `average_price` is per contract and does not scale.
    """
    if left == leg.quantity:
        return leg
    share = Decimal(left) / Decimal(leg.quantity)
    return replace(
        leg,
        quantity=left,
        market_value=_apportion(leg.market_value, share),
        unrealised=_apportion(leg.unrealised, share),
    )


def _apportion(amount: Decimal, share: Decimal) -> Decimal:
    """A share of a money amount, to the cent.

    Quantised because the division is exact only by luck — two thirds of a
    market value is not a number of cents — and this figure is reported to a
    human rather than added to anything the Gate measures. The sleeve's
    mark-to-market is summed from the broker's own lines in `read_book`, so
    nothing budgeted depends on this rounding.
    """
    return (amount * share).quantize(Decimal("0.01"))


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
