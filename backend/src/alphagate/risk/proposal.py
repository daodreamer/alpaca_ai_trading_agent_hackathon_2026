"""What the agent proposes — specs/03 D2.

PURE. Stdlib plus `alphagate.core` and `alphagate.options`.

A proposal is a *request*, not a decision. It carries the model's reasoning
because the journal (specs/06) needs to show why a trade was proposed next to
whether it was allowed — but `rationale` is evidence, never input. Nothing in
`alphagate.risk` parses it. A Gate that reads the argument for a trade is a Gate
that can be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from alphagate.core.errors import InvariantViolation
from alphagate.options.risk import StructureRisk
from alphagate.options.structure import OptionStructure

__all__ = ["Intent", "TradeProposal"]


class Intent(Enum):
    """Why the order exists. The Gate treats these three very differently."""

    OPEN = "open"
    CLOSE = "close"
    ROLL = "roll"

    @property
    def opens_risk(self) -> bool:
        """True when the fill can leave the account holding more risk.

        A roll counts as an open: the near leg goes away, but a new position
        with its own maximum loss arrives, and budget arithmetic that ignored
        it would understate the book.
        """
        return self is not Intent.CLOSE


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """One structure, one quantity, one intent, and the reasoning behind it."""

    structure: OptionStructure
    risk: StructureRisk
    quantity: int
    intent: Intent
    rationale: str
    """The model's reasoning. Carried for the journal; never parsed by the Gate."""
    proposed_by: str
    """Model id, e.g. ``"claude-opus-5"``."""
    proposal_id: str
    """Stable identity. Feeds the derived idempotency key in specs/04 D4."""
    risk_as_of: datetime
    """The instant `risk` was computed against.

    Carried so the Gate can age the quotes forward to *its* `as_of` rather than
    trusting a number that was true when the agent looked. An LLM call sits
    between perception and this Gate, and it can easily take longer than the
    freshness limit — a proposal whose quotes were 5s old when the model started
    thinking is not 5s old when the verdict is written. Without this field
    `fresh_quotes` would be a check on how recently somebody did arithmetic.
    """

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InvariantViolation(
                f"proposal quantity must be positive, got {self.quantity}; "
                "direction lives in `intent`, not in the sign of a count"
            )
        if not self.proposal_id.strip():
            raise InvariantViolation("proposal_id must be a non-empty identifier")
        if self.risk_as_of.tzinfo is None:
            raise InvariantViolation(f"risk_as_of must be tz-aware, got {self.risk_as_of!r}")

    @property
    def total_max_loss(self) -> Decimal:
        """Maximum loss across the whole order, in account currency."""
        return self.risk.max_loss * self.quantity
