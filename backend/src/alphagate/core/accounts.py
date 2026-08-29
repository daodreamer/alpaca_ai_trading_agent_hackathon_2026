"""Users and watchlists — specs/04-domain-model.md (extended by ADR 0012).

`specs/12-storage.md` lists `users`, `user_watchlists` and `watchlist_items` as
tables while `specs/04-domain-model.md` never gives them a model. These are the
minimum shapes those tables need and nothing more: no roles, no preferences, no
sharing. Adding a field here is adding product scope (CLAUDE.md §6).

Ownership is carried even though the MVP is single-user, because
`specs/16-security.md` requires that user-owned resources carry an owner identity
so a later multi-user evolution is not a rewrite.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, UserId, WatchlistId
from alphagate.core.time_model import ensure_utc

__all__ = ["User", "Watchlist", "WatchlistItem"]


@dataclasses.dataclass(frozen=True, slots=True)
class User:
    id: UserId
    display_name: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.id:
            raise InvariantViolation("a user needs an id")
        if not self.display_name.strip():
            raise InvariantViolation(f"{self.id}: display_name is required")
        object.__setattr__(self, "created_at", ensure_utc(self.created_at, field="created_at"))


@dataclasses.dataclass(frozen=True, slots=True)
class WatchlistItem:
    """One symbol on a watchlist, at a position the user chose.

    `position` is explicit rather than implied by list order, because the order
    has to survive a round trip through a table whose rows have no order.
    """

    symbol: Ticker
    position: int

    def __post_init__(self) -> None:
        if self.position < 0:
            raise InvariantViolation(f"{self.symbol}: position must not be negative")


@dataclasses.dataclass(frozen=True, slots=True)
class Watchlist:
    id: WatchlistId
    user_id: UserId
    name: str
    items: tuple[WatchlistItem, ...] = ()
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation(f"{self.id}: name is required")
        if self.created_at is not None:
            object.__setattr__(self, "created_at", ensure_utc(self.created_at, field="created_at"))

        symbols = [item.symbol for item in self.items]
        if len(set(symbols)) != len(symbols):
            raise InvariantViolation(
                f"{self.id}: a symbol appears twice; a watchlist is a set of symbols "
                "in an order, not a bag"
            )
        positions = [item.position for item in self.items]
        if len(set(positions)) != len(positions):
            raise InvariantViolation(
                f"{self.id}: two items share a position, so the order is ambiguous"
            )
        object.__setattr__(self, "items", tuple(sorted(self.items, key=lambda item: item.position)))

    @property
    def symbols(self) -> tuple[Ticker, ...]:
        """The symbols, in the user's order."""
        return tuple(item.symbol for item in self.items)
