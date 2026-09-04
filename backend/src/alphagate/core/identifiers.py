"""Identity types.

These are `NewType` aliases over `str` rather than value classes: they cost
nothing at runtime, they are hashable and orderable for free, and mypy still
refuses to let a `LevelId` be passed where a `Ticker` is expected.

Upstream carried five more — user, watchlist and alert ids — for a product with
accounts in it. AlphaGate has one account and no users, so they went with the
packages that named them (adr/0001 D5).
"""

from __future__ import annotations

import re
from typing import Final, NewType

from alphagate.core.errors import InvariantViolation

__all__ = [
    "LevelId",
    "Ticker",
    "ticker",
]

Ticker = NewType("Ticker", str)
LevelId = NewType("LevelId", str)

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
