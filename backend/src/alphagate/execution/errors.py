"""Execution failures — specs/04.

These are *operational* errors, not domain ones, so they descend from
`ExecutionError` rather than from `alphagate.core.errors.DomainError`. The
distinction is the one the agent loop keys off: a `DomainError` means a value was
wrong and the cycle should be journalled and abandoned; an `ExecutionError` means
the world did not cooperate and the order may or may not exist.

The most important type here is `ToolTimeout`, and the most important sentence in
the file is that it is **not** a failure. specs/04 D4: a timeout is an unknown
outcome. Resolving it by resubmitting is how one intended position becomes two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphagate.execution.lifecycle import Submission

__all__ = [
    "BrokerRefused",
    "ExecutionError",
    "MalformedToolOutput",
    "PartialFillBreach",
    "ToolTimeout",
    "TransportFailure",
    "UnsubmittableOrder",
]


class ExecutionError(Exception):
    """Something went wrong on the way to, or back from, the broker."""


class TransportFailure(ExecutionError):
    """The call did not reach the server, or the server answered 5xx.

    The only class of error `submit` retries. A 4xx is not this: a malformed
    request retried three times is a malformed request three times.
    """


class ToolTimeout(ExecutionError):
    """The call was sent and no answer came back.

    **Not a failure — an unknown outcome.** The order may be live. It is resolved
    by reading back `get_order_by_client_id`, never by resubmitting.
    """


class MalformedToolOutput(ExecutionError):
    """The MCP server answered with something we cannot read.

    Raised rather than defaulted. A missing `status` field silently treated as
    "probably fine" is the kind of shortcut that reconciles at 16:05 into a
    position nobody knew about.
    """


class BrokerRefused(MalformedToolOutput):
    """The broker answered, legibly, that it would not place the order.

    A subclass rather than a sibling, deliberately. Every handler that treats an
    unreadable answer as "no order exists" must treat a refusal the same way —
    that is the same conclusion reached for a better reason — so widening the
    hierarchy here would silently change what those handlers catch.

    What it buys is the journal line. A refused order used to be recorded as
    `MalformedToolOutput: place_option_order returned an error: {...}`, which
    reads as "our parser broke" and sends the reader into this codebase. The
    fact is `422 position intent mismatch` — the broker read the order fine and
    said no, and the answer is in the order, not in the adapter.
    """


class UnsubmittableOrder(ExecutionError):
    """The order cannot be expressed as an Alpaca order.

    Every case is a bug rather than a market condition — more than four legs, an
    unmappable intent. specs/02 D3 makes most of them unconstructible; this is
    what happens if a new `StructureKind` is added without reading specs/04 D3.
    """


class PartialFillBreach(ExecutionError):
    """A multi-leg order filled on some legs and not others.

    This is the state specs/03 exists to prevent: a spread half filled is a naked
    leg, with the unbounded loss that the whole type system was arranged to make
    unrepresentable. Alpaca fills `mleg` atomically, so this should not happen —
    which is exactly why it is an exception and not a branch.

    We do **not** attempt to leg out automatically. Trading into a broken
    position with an automated system that has just been surprised is how a bad
    afternoon becomes a bad week. The handler latches the kill switch
    (`PortfolioSnapshot.killswitch_tripped`, specs/03 D4), which blocks opens and
    leaves closes permitted, and waits for a human.
    """

    def __init__(self, submission: Submission) -> None:
        self.submission = submission
        filled = sum(1 for leg in submission.legs if leg.is_filled)
        super().__init__(
            f"order {submission.client_order_id} filled {filled} of "
            f"{len(submission.legs)} legs: this is a naked leg, not a position. "
            "Latch the kill switch and reconcile by hand — specs/04 D5."
        )
