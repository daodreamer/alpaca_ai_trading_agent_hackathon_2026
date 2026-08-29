"""The user-level persistence port — specs/04-domain-model.md, specs/10-chart-ui.md.

A line the user drew is a domain entity with its own lifecycle, not a chart
artifact (`alphagate.core.levels.UserLevel`). `specs/10-chart-ui.md` puts it plainly:
anything the user expects to survive a session must be a persisted domain object
rather than an opaque chart object. This port is what that survival runs through.

Declared here for the same reason `AlertEventStore` is: a use case that reads or
writes user levels should depend on the shape of the store, not on PostgreSQL.
`UserLevelRepository` satisfies it structurally, and so does the in-memory store
the API uses when no database is configured.

`upsert` rather than separate create and update: a `UserLevel` carries its own
id, so "store this level" is one operation and the caller never has to know
whether the row already existed.
"""

from __future__ import annotations

from typing import Protocol

from alphagate.core.identifiers import Ticker, UserId, UserLevelId
from alphagate.core.levels import UserLevel

__all__ = ["UserLevelStore"]


class UserLevelStore(Protocol):
    def upsert(self, level: UserLevel) -> None:
        """Store `level`, replacing any existing level under the same id."""
        ...

    def delete(self, level_id: UserLevelId) -> None:
        """Remove a level. Deleting one that is not there is not an error —
        the caller's intent is that it be gone, and it is."""
        ...

    def get(self, level_id: UserLevelId) -> UserLevel | None: ...

    def for_user(self, user_id: UserId) -> tuple[UserLevel, ...]:
        """Every level this user owns, in a stable order."""
        ...

    def for_symbol(self, user_id: UserId, symbol: Ticker) -> tuple[UserLevel, ...]:
        """This user's levels on one symbol, in a stable order.

        Separate from filtering `for_user` in the caller because a database can
        answer it with an index, and a chart request asks exactly this question.
        """
        ...
