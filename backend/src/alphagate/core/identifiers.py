"""Identity types.

These are `NewType` aliases over `str` rather than value classes: they cost
nothing at runtime, they are hashable and orderable for free, and mypy still
refuses to let a `LevelId` be passed where an `AlertRuleId` is expected.

Surrogate database identifiers are *not* modelled here. Persistence assigns
them; the Domain treats an id as an opaque string it was handed.
"""

from __future__ import annotations

import re
from typing import Final, NewType

from alphagate.core.errors import InvariantViolation

__all__ = [
    "AlertEventId",
    "AlertRuleId",
    "LevelId",
    "Ticker",
    "UserId",
    "UserLevelId",
    "WatchlistId",
    "ticker",
]

Ticker = NewType("Ticker", str)
UserId = NewType("UserId", str)
LevelId = NewType("LevelId", str)
UserLevelId = NewType("UserLevelId", str)
WatchlistId = NewType("WatchlistId", str)
AlertRuleId = NewType("AlertRuleId", str)
AlertEventId = NewType("AlertEventId", str)

# US listings: letters, digits, dot (BRK.B) and hyphen (some preferred/warrant classes).
_TICKER_PATTERN: Final = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,15}$")


def ticker(raw: str) -> Ticker:
    """Normalize and validate a ticker.

    Normalization is uppercase plus surrounding-whitespace removal, so `"aapl"`
    and `" AAPL "` are the same symbol. Internal whitespace is a malformed
    ticker, not something to strip.
    """
    candidate = raw.strip().upper()
    if not _TICKER_PATTERN.match(candidate):
        raise InvariantViolation(f"malformed ticker: {raw!r}")
    return Ticker(candidate)
