"""The clock port.

Domain code never calls `datetime.now()`. It asks a `Clock`, so the same logic
runs identically under a live feed, a replay and a test — which is what
`CLAUDE.md` §11 requires and what Phase 2's replay depends on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

__all__ = ["Clock"]


class Clock(Protocol):
    def now(self) -> datetime:
        """The current instant, timezone-aware and in UTC."""
        ...
