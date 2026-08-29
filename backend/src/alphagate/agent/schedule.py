"""Cadence — specs/05 D8. Pure: a session in, a list of instants out.

> "Cycle every 15 minutes during RTH, plus one pass 20 minutes after the open.
> Not on the open itself — the first minutes' quotes are the widest of the day
> and `worst_spread_pct` would veto most candidates anyway."

The schedule is computed rather than slept into, which is the whole reason this
module is separate from `runner.py`. A loop that decides when to run *while*
running can only be tested by waiting; a function that returns the times can be
tested against a session boundary, a half day, and a holiday, in microseconds.

Two rules the spec states and one it implies:

* **not on the open.** The first minutes are the widest markets of the day. An
  agent that proposes into them spends its whole menu on candidates the
  liquidity check will refuse, which is a busy way to do nothing.
* **not into the close.** Not in the spec, and it belongs: an order placed at
  15:58 that does not fill is a `day` order that expires (specs/04 D3 — options
  support no other time in force), so the last slot is far enough from the bell
  for a limit to work.
* **exits are evaluated on every slot**, entries only on slots inside the entry
  window. Closing a position at 15:55 is exactly when you want to be able to.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Final
from zoneinfo import ZoneInfo

from alphagate.core.errors import InvariantViolation

__all__ = [
    "CYCLE_INTERVAL",
    "FIRST_CYCLE_DELAY",
    "LAST_ENTRY_BEFORE_CLOSE",
    "MARKET_TZ",
    "CycleKind",
    "Slot",
    "session_slots",
]

MARKET_TZ: Final = ZoneInfo("America/New_York")
CYCLE_INTERVAL: Final = timedelta(minutes=15)
FIRST_CYCLE_DELAY: Final = timedelta(minutes=20)
"""After the open. specs/05 D8 — the first minutes' quotes are the widest."""
LAST_ENTRY_BEFORE_CLOSE: Final = timedelta(minutes=20)
"""No new positions inside the last twenty minutes: a `day` limit that does not
fill before the bell expires, and options support no other time in force."""


class CycleKind(Enum):
    """What a slot is allowed to do."""

    FULL = "full"
    """Exits and entries."""
    EXITS_ONLY = "exits_only"
    """Too close to the bell to open something new — but never too close to
    close something. specs/03 D4: the Gate never blocks an exit, and neither
    does the schedule."""

    @property
    def may_open(self) -> bool:
        return self is CycleKind.FULL


@dataclass(frozen=True, slots=True)
class Slot:
    """One scheduled cycle."""

    at: datetime
    kind: CycleKind
    sequence: int
    """Position within the session, from zero. Feeds `cycle_id_for`, so the id
    of the third cycle of a day is the same on a replay as it was live."""

    @property
    def local(self) -> datetime:
        return self.at.astimezone(MARKET_TZ)

    def __str__(self) -> str:
        return f"{self.local:%H:%M} {self.kind.value} #{self.sequence:03d}"


def session_slots(
    open_at: datetime,
    close_at: datetime,
    *,
    interval: timedelta = CYCLE_INTERVAL,
    first_delay: timedelta = FIRST_CYCLE_DELAY,
    entry_cutoff: timedelta = LAST_ENTRY_BEFORE_CLOSE,
) -> tuple[Slot, ...]:
    """Every cycle for one session, in order.

    Both bounds come from the exchange clock (`get_clock`), never from a local
    one — a half day closing at 13:00 is a real thing that happens three times a
    year, and a hardcoded 16:00 would have the agent proposing into a closed
    market and reconciling into an empty one.
    """
    if open_at.tzinfo is None or close_at.tzinfo is None:
        raise InvariantViolation("session bounds must be tz-aware")
    if close_at <= open_at:
        raise InvariantViolation(
            f"session closes ({close_at.isoformat()}) before it opens ({open_at.isoformat()})"
        )
    if interval <= timedelta(0):
        raise InvariantViolation(f"interval must be positive, got {interval}")

    cutoff = close_at - entry_cutoff
    return tuple(
        Slot(
            at=moment,
            kind=CycleKind.FULL if moment <= cutoff else CycleKind.EXITS_ONLY,
            sequence=index,
        )
        for index, moment in enumerate(_ticks(open_at + first_delay, close_at, interval))
    )


def _ticks(start: datetime, end: datetime, interval: timedelta) -> Iterator[datetime]:
    """Half-open: the first tick is included, the close is not.

    Excluding the close matters — a cycle at exactly 16:00:00 would read a
    market that has already stopped and place an order that cannot fill.
    """
    moment = start
    while moment < end:
        yield moment
        moment += interval


def next_slot(slots: tuple[Slot, ...], after: datetime) -> Slot | None:
    """The first slot strictly after `after`. `None` once the session is done.

    Strictly after, so a runner that wakes a millisecond early does not run the
    same slot twice — which would produce two cycles sharing an id, and two
    journal lines claiming to be the same decision.
    """
    for slot in slots:
        if slot.at > after:
            return slot
    return None
