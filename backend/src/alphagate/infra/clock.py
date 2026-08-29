"""Clock implementations for the `alphagate.core.clock.Clock` port."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alphagate.core.errors import InvariantViolation
from alphagate.core.time_model import ensure_utc

__all__ = ["ReplayClock", "SystemClock"]


class SystemClock:
    """Wall-clock time. The only place in the system that reads the real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class ReplayClock:
    """A clock the caller advances by hand.

    Drives replay and pins "now" in tests. Time only moves forward: a replay that
    can rewind is a replay that can produce a bar the engine has already seen,
    and that is indistinguishable from a look-ahead bug.
    """

    __slots__ = ("_now",)

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start, field="start")

    def now(self) -> datetime:
        return self._now

    def advance_to(self, instant: datetime) -> None:
        moment = ensure_utc(instant, field="instant")
        if moment < self._now:
            raise InvariantViolation(
                f"replay clock cannot move backwards: {self._now.isoformat()} -> "
                f"{moment.isoformat()}"
            )
        self._now = moment

    def advance_by(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise InvariantViolation(f"replay clock cannot move backwards by {delta}")
        self._now = self._now + delta
