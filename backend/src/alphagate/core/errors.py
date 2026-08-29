"""Domain error hierarchy.

Domain errors describe a violated rule, not a failed operation. They carry no
HTTP status, no error code registry and no i18n — those are presentation
concerns and belong to the layers above.
"""

from __future__ import annotations

__all__ = [
    "CalendarError",
    "CalendarHorizonExceeded",
    "DomainError",
    "InvalidTransition",
    "InvariantViolation",
    "UnknownExchange",
]


class DomainError(Exception):
    """Base class for every rule the Domain enforces."""


class InvariantViolation(DomainError):
    """A value or entity would violate a stated invariant.

    Raised at construction time. A constructed domain object is always valid.
    """


class InvalidTransition(DomainError):
    """A state machine was asked to make a transition it does not permit."""


class CalendarError(DomainError):
    """The `TradingCalendar` port cannot answer.

    Declared in the Domain because it is part of the port's contract: callers
    must be able to handle it without importing an infrastructure module.
    """


class CalendarHorizonExceeded(CalendarError):
    """A session was requested beyond the calendar's forward horizon.

    Raised rather than answered, because a rule-extrapolated future session that
    turns out to be a holiday corrupts every bar bucketed against it
    (ADR 0006 D4).
    """


class UnknownExchange(CalendarError):
    """No calendar is configured for the requested exchange."""
