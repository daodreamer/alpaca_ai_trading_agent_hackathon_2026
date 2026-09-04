"""Two sleeves, one account, shown apart — specs/03 D6.

`risk/sleeve.py` says of itself that two strategies sharing one broker account
"budgeted against the same number" until `Sleeve` existed, and that the fix is
"an allocation, and the P&L of the positions that allocation paid for. The
account does not appear in the arithmetic." Neither `status.json` nor
`equity-status.json` currently *renders* that number — both still publish the
broker's whole-account equity under the key `equity` (see `live/status.py` and
`live/equity_status.py`), because a sleeve-scoped figure was wired into the
Gate's own budgets before it was wired into the two files a dashboard reads.

This module closes that gap **for display only**, by calling the same pure
functions the live agents already call to size their own budgets:
`alphagate.agent.book.realised_pl` and `alphagate.risk.sleeve.Sleeve` /
`residual_sleeve`. Nothing here is invented arithmetic — it is the identical
formula `agent/book.py`'s `read_book` and `live/equity.py`'s
`EquityContext.sleeve` use, run again against the two snapshots a browser is
actually allowed to see (a JSON file, never a broker session).

**Why the options sleeve is bottom-up and the equity sleeve is a residual.**
Exactly `risk/sleeve.residual_sleeve`'s own reasoning: the options sleeve's
realised and unrealised P&L are separately identifiable from its own contracts
and its own journalled round-trips, so it is measured directly. The equity book
is everything the account holds that is not that, so it is computed as
`account_equity - options.equity`. Computing the options sleeve does not need
the equity snapshot at all; computing the equity sleeve *does* need the options
one, which is why `equity_sleeve_view` takes the other sleeve's summary as an
argument instead of reaching for its own copy of the same arithmetic.

**Nothing here is what either kill switch actually enforces.** Each live agent
answers "is my sleeve down too far" from its own process state, at the moment
it decides whether to trade. This module answers the same question from
whatever the two agents last *published*, which is a snapshot a fifteen-minute
or thirty-second window old. The two agree at the moment either file is
written and can drift apart between writes — which is a property of reading
two independent heartbeats, not a bug in the arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from alphagate.agent.book import realised_pl
from alphagate.equity import EQUITY_SLEEVE_ALLOCATION
from alphagate.journal import Journal
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION
from alphagate.risk.sleeve import Sleeve, residual_sleeve

__all__ = [
    "SleeveSummary",
    "build_sleeve_overview",
    "equity_sleeve_view",
    "options_sleeve_view",
    "sleeve_to_json",
]


@dataclass(frozen=True, slots=True)
class SleeveSummary:
    """One sleeve, as far as it can be read from a published snapshot.

    `running=False` is not an error state: the agent may simply not have
    written a snapshot yet, in which case every other field but `allocation`
    is `None` rather than a guessed zero — CLAUDE.md's `None` means unmeasured
    rule applies to a sleeve exactly as it applies to one position's mark.
    """

    name: str
    allocation: Decimal
    running: bool
    equity: Decimal | None = None
    realised: Decimal | None = None
    unrealised: Decimal | None = None
    drawdown_pct: Decimal | None = None
    max_drawdown_pct: Decimal | None = None
    killswitch_tripped: bool | None = None
    open_positions: int | None = None
    activity_today: int | None = None
    activity_label: str = ""
    note: str = ""


def options_sleeve_view(
    status: Mapping[str, Any] | None, *, journal: Journal
) -> SleeveSummary:
    """The options sleeve, bottom-up from its own contracts and round-trips.

    Reads `journal.read_through` bounded at the snapshot's own `session_day` —
    never at whatever day it happens to be when this runs — so a replay of an
    earlier day cannot see a later one, the same look-ahead discipline
    `agent.book.read_book` observes when it computes the identical number.
    """
    if status is None:
        return SleeveSummary(
            name="options",
            allocation=OPTIONS_SLEEVE_ALLOCATION,
            running=False,
            note="the options agent has not published a snapshot yet",
        )

    day = _session_day(status)
    records = journal.read_through(day) if day is not None else ()
    positions = status.get("positions")
    unrealised = _sum_unrealised(positions)

    sleeve = Sleeve(
        name="options",
        allocation=OPTIONS_SLEEVE_ALLOCATION,
        realised=realised_pl(records),
        unrealised=unrealised,
    )
    return SleeveSummary(
        name="options",
        allocation=OPTIONS_SLEEVE_ALLOCATION,
        running=True,
        equity=sleeve.equity,
        realised=sleeve.realised,
        unrealised=sleeve.unrealised,
        drawdown_pct=_decimal(status.get("drawdown_pct")),
        max_drawdown_pct=_decimal(status.get("max_drawdown_pct")),
        killswitch_tripped=bool(status.get("killswitch_tripped", False)),
        open_positions=len(positions) if isinstance(positions, Sequence) else None,
        activity_today=_int(status.get("fills_today")),
        activity_label="trades today",
    )


def equity_sleeve_view(
    status: Mapping[str, Any] | None, *, options: SleeveSummary
) -> SleeveSummary:
    """The equity sleeve, as the residual of the account against the options
    sleeve — `risk.sleeve.residual_sleeve`'s own identity.

    Needs the options sleeve's `equity`, not merely whether it is running: a
    residual computed against `Decimal(0)` for an options sleeve that is
    actually holding a loss would hand that loss to the equity book, which is
    exactly the coupling specs/03 D6 exists to remove. So this sleeve reads
    `unavailable` whenever the options sleeve could not be measured, rather
    than guess.
    """
    if status is None:
        return SleeveSummary(
            name="equity",
            allocation=EQUITY_SLEEVE_ALLOCATION,
            running=False,
            note="the equity agent has not published a snapshot yet",
        )

    account_equity = _decimal(status.get("equity"))
    equity_value: Decimal | None = None
    realised_value: Decimal | None = None
    unrealised_value: Decimal | None = None
    note = ""
    if options.equity is not None and account_equity is not None:
        residual = residual_sleeve(
            "equity",
            allocation=EQUITY_SLEEVE_ALLOCATION,
            account_equity=account_equity,
            others=(
                Sleeve(
                    name="options",
                    allocation=options.allocation,
                    realised=options.realised or Decimal(0),
                    unrealised=options.unrealised or Decimal(0),
                ),
            ),
        )
        equity_value = residual.equity
        # `residual_sleeve` carries the whole of its P&L as `unrealised`,
        # deliberately: it is a mark against the allocation, not a proven
        # closed round-trip, and `realised` is always zero for a residual —
        # see the domain function's own docstring for why that split is
        # honest rather than an omission.
        realised_value = residual.realised
        unrealised_value = residual.unrealised
    else:
        note = "the options sleeve is not reporting, so the residual cannot be split from it"

    return SleeveSummary(
        name="equity",
        allocation=EQUITY_SLEEVE_ALLOCATION,
        running=True,
        equity=equity_value,
        realised=realised_value,
        unrealised=unrealised_value,
        drawdown_pct=_decimal(status.get("drawdown_pct")),
        max_drawdown_pct=_decimal(status.get("max_drawdown_pct")),
        killswitch_tripped=bool(status.get("killswitch_tripped", False)),
        open_positions=_int(status.get("positions_held")),
        activity_today=_int(status.get("orders_today")),
        activity_label="orders placed today",
        note=note,
    )


def build_sleeve_overview(
    options_status: Mapping[str, Any] | None,
    equity_status: Mapping[str, Any] | None,
    *,
    journal: Journal,
) -> dict[str, Any]:
    """Both sleeves, for `/api/sleeves`. The one place that composes the two."""
    options = options_sleeve_view(options_status, journal=journal)
    equity = equity_sleeve_view(equity_status, options=options)
    return {"options": sleeve_to_json(options), "equity": sleeve_to_json(equity)}


def sleeve_to_json(summary: SleeveSummary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "allocation": _str(summary.allocation),
        "running": summary.running,
        "equity": _str(summary.equity),
        "realised": _str(summary.realised),
        "unrealised": _str(summary.unrealised),
        "drawdown_pct": _str(summary.drawdown_pct),
        "max_drawdown_pct": _str(summary.max_drawdown_pct),
        "killswitch_tripped": summary.killswitch_tripped,
        "open_positions": summary.open_positions,
        "activity_today": summary.activity_today,
        "activity_label": summary.activity_label,
        "note": summary.note,
    }


# ------------------------------------------------------------------ #


def _session_day(status: Mapping[str, Any]) -> date | None:
    raw = status.get("session_day")
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _sum_unrealised(positions: Any) -> Decimal:
    if not isinstance(positions, Sequence):
        return Decimal(0)
    total = Decimal(0)
    for item in positions:
        if isinstance(item, Mapping):
            value = _decimal(item.get("unrealised"))
            if value is not None:
                total += value
    return total


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")
