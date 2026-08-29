"""What the cycle passes around — specs/05 D1, D2, D3.

Pure value types. No I/O, no clock, no model call; those live in `proposer.py`
and `cycle.py`.

Two of these carry the whole safety argument of the layer.

`Candidate` is the unit the model chooses between. It is **fully formed before
the model sees it**: a validated `OptionStructure`, its `StructureRisk`, and a
quantity already decided by the risk budget. The model's entire channel of
expression is an integer index into a list of these.

`Choice` is what comes back: an index or `None`. There is no field in which a
model could write a symbol, a strike, a quantity or a price, so a hallucinated
contract cannot reach the broker — not because it is validated away, but because
there is nowhere to put it. Same move as specs/02 D3 making naked shorts
unconstructible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.options import OptionStructure, StructureRisk

__all__ = [
    "Candidate",
    "Choice",
    "MarketRead",
    "ModelCall",
    "Setup",
    "Stage",
]


class Stage(Enum):
    """How far a cycle got. Every cycle records one — specs/05 D1.

    `NO_SETUP` and `DECLINED` are the majority and they are the point. A journal
    that only contains trades cannot answer "why didn't it trade at 14:30?",
    which is the question a judge asks.
    """

    NO_SETUP = "no_setup"
    NO_CANDIDATES = "no_candidates"
    DECLINED = "declined"
    VETOED = "vetoed"
    SUBMITTED = "submitted"
    REJECTED = "rejected"
    FILLED = "filled"
    DRY_RUN = "dry_run"
    """Approved by the Gate, deliberately not sent — the pre-open check runs
    this way. A first-class outcome rather than a flag threaded through the
    trading path, so the journal never has to be read as "submitted, but not
    really"."""
    BREACHED = "breached"
    """A partial fill on a multi-leg order — specs/04 D5. Not in the spec's
    original list because it is not a stage the cycle passes through; it is a
    stage the cycle stops at, loudly, with the kill switch latched."""

    @property
    def traded(self) -> bool:
        return self in {Stage.SUBMITTED, Stage.FILLED}


@dataclass(frozen=True, slots=True)
class MarketRead:
    """What the deterministic engines saw — specs/05 D2.

    Facts from the core engines, never raw prices for the model to interpret and
    never an image. This is the reuse dividend from adr/0001: the median entrant
    pastes OHLC into a prompt; we hand the model a trend state machine's output.

    `trend`, `confluence` and `levels` are optional because perception can be
    **incomplete**, and an incomplete read must be able to say so. A `None` trend
    is not a neutral trend — it is a trend nobody measured, and `screen` refuses
    to produce a `Setup` without one. That is specs/05 D6's fail-closed rule
    applied to perception itself, and it is the same discipline as specs/02 D2's
    missing greeks: absence is representable, and never reads as zero.
    """

    underlying: Ticker
    as_of: datetime
    spot: Decimal
    """The only field that is never optional. A read with no price is not a read."""
    atr_pct: Decimal | None = None
    iv_rank: Decimal | None = None
    """Where current IV sits inside its own trailing range, 0–100. The headline
    input: it is what separates "sell premium" from "buy premium".

    **This is a rank, not a level.** SPY at 15% IV can be at the 90th percentile
    of its own year or the 10th, and those are opposite trades. `None` means
    nobody had enough history to rank it — never a middling default, because a
    default here is a claim about how rich premium is."""
    iv_percentile: Decimal | None = None
    """The robust sibling of `iv_rank`: the share of the window below current.
    One panic spike three months ago moves the rank a long way and this barely
    at all, so both are carried and the strategy says which it uses."""
    iv_vs_hv: Decimal | None = None
    """Implied over realised, exactly computable from today's chain and bars.
    Carries the "is premium rich" question while `iv_rank` is unavailable."""
    hv_rank: Decimal | None = None
    """Where realised volatility sits in its own trailing range, 0-100.

    Not a substitute for `iv_rank` and never presented as one — it ranks what
    the underlying *did*, not what options are *charging*. It earns its place
    because it needs no options-data entitlement: stock bars are enough, so this
    is available on day one where `iv_rank` is not."""
    earnings_within_dte: bool | None = None
    """`True` yes, `False` no, **`None` nobody checked** — see `agent/earnings.py`.

    Alpaca has no earnings calendar, so for a single name with no hand-maintained
    entry this is `None` and stays `None`. Collapsing it to `False` would render
    "we did not look" as "there is none", and an earnings print inside the
    holding period is the most common way a defined-risk premium sale becomes a
    maximum loss."""
    trend: Any | None = None
    """`core.trend_engine.TrendState`. `None` is *unmeasured*, not neutral."""
    confluence: Any | None = None
    levels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise InvariantViolation(f"as_of must be tz-aware, got {self.as_of!r}")
        if not isinstance(self.spot, Decimal):
            raise InvariantViolation(f"spot must be Decimal, got {type(self.spot).__name__}")
        if not self.spot.is_finite() or self.spot <= 0:
            raise InvariantViolation(f"spot must be finite and positive, got {self.spot}")
        for name in ("atr_pct", "iv_rank", "iv_percentile", "iv_vs_hv", "hv_rank"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")
        for name in ("iv_rank", "iv_percentile", "hv_rank"):
            rank = getattr(self, name)
            if rank is not None and not 0 <= rank <= 100:
                raise InvariantViolation(
                    f"{name} must be a rank in 0-100, got {rank}; "
                    "a level of volatility is not a rank of one"
                )

    @property
    def is_complete(self) -> bool:
        """Whether enough was measured to act on. The screen requires it.

        `earnings_within_dte` must be an explicit `False`. Neither `True` (an
        event inside the window) nor `None` (nobody checked) is a book to sell
        premium into.

        Deliberately strict, and deliberately satisfiable without `iv_rank`:
        that field needs an options-data entitlement this account does not have
        (see `agent/iv_store.py`), and gating every trade on it would mean not
        trading at all. `iv_vs_hv` answers the same question — is premium rich
        relative to what the underlying is actually doing — from data we can
        read today, so it is the one that is required.
        """
        return (
            self.atr_pct is not None
            and self.iv_vs_hv is not None
            and self.earnings_within_dte is False
        )


@dataclass(frozen=True, slots=True)
class Setup:
    """A `MarketRead` shape the strategy recognises — specs/05 D1 step 2.

    The concrete rules are specs/07's, deliberately: tuning a threshold should
    not mean editing the orchestration. This type is the interface between them.
    """

    underlying: Ticker
    name: str
    """e.g. `"high_iv_rank_bullish"`. Recorded, so the journal can group by it."""
    bias: str
    """`bullish` | `bearish` | `neutral`. Which structure family applies."""
    reason: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """One fully-formed, already-sized structure the model may choose.

    Everything dangerous is decided before this exists. By the time the model
    sees it, the structure is validated (specs/02 D3), the risk is computed, and
    the quantity is a pure function of the risk budget (specs/05 D4).
    """

    index: int
    structure: OptionStructure
    risk: StructureRisk
    quantity: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise InvariantViolation(f"candidate index must not be negative, got {self.index}")
        if self.quantity <= 0:
            raise InvariantViolation(
                f"candidate quantity must be positive, got {self.quantity}; "
                "a candidate that cannot be sized is dropped, never shown"
            )

    def _net_ror(self) -> Decimal:
        from alphagate.agent.candidates import net_return_on_risk

        return net_return_on_risk(self.risk)

    def summarise(self) -> dict[str, Any]:
        """The model-facing view. Numbers as strings; no OCC symbols.

        Symbols are withheld deliberately. The model does not need them to
        choose, and a prompt that contains well-formed OCC symbols is a prompt
        that teaches a model to emit one.
        """
        greeks = self.risk.net_greeks
        return {
            "index": self.index,
            "structure": self.structure.kind.value,
            "expiry": self.structure.expiry.isoformat(),
            "days_to_expiry": self.risk.days_to_expiry,
            "width": str(self.structure.width),
            "strikes": [str(leg.contract.strike) for leg in self.structure.legs],
            "net_premium": str(self.risk.net_premium),
            "max_loss": str(self.risk.max_loss),
            "max_profit": str(self.risk.max_profit),
            "return_on_risk": f"{self.risk.return_on_risk:.3f}",
            "return_on_risk_after_spread": f"{self._net_ror():.3f}",
            "breakevens": [str(b) for b in self.risk.breakevens],
            "quantity": self.quantity,
            "net_delta": None if greeks is None else round(greeks.delta, 3),
            "net_vega": None if greeks is None else round(greeks.vega, 3),
            "worst_spread_pct": f"{self.risk.worst_spread_pct:.4f}",
        }


@dataclass(frozen=True, slots=True)
class Choice:
    """What the model returned — specs/05 D3.

    `candidate_index is None` means decline, and **declining is a valid
    answer**, not an error. specs/05 D6: doing nothing is always available and
    correct.

    `confidence` is recorded and never acted on (specs/05 D5). Self-reported
    confidence is not calibrated, and scaling position size by it is the most
    common way an agent turns a good structure into a bad bet. It is kept so the
    demo can show whether it *would* have correlated — an honest observation is
    worth more than a fake edge.
    """

    candidate_index: int | None
    rationale: str
    self_reported_confidence: float = 0.0
    """Named at length on purpose.

    `TrendState.confidence` and `Level.confidence` are *measured* quantities —
    how much of the requested evidence an engine could actually read. This one
    is a model's opinion of its own opinion. Two different things called
    `confidence` in one codebase is the ambiguity that produced the `iv_rank`
    incident, and the fix there and here is the same: put the meaning in the
    name.

    The length is also the point. Nobody reaches for
    `choice.self_reported_confidence` while tuning position size without
    noticing what they are about to do."""

    @property
    def declined(self) -> bool:
        return self.candidate_index is None

    def resolve(self, candidates: tuple[Candidate, ...]) -> Candidate | None:
        """Look the choice up, treating anything unusable as a decline.

        An index outside the range is **not** an error to retry around: a model
        that names a candidate that does not exist has not chosen, and asking it
        again is asking it to guess harder. specs/05 D3.
        """
        if self.candidate_index is None:
            return None
        if not 0 <= self.candidate_index < len(candidates):
            return None
        return candidates[self.candidate_index]


@dataclass(frozen=True, slots=True)
class ModelCall:
    """The record of one proposal — specs/06 D2.

    Everything needed to explain, and to replay, a nondeterministic step:
    which model, which prompt version, what came back, and how long it took.
    A prompt edit is a version bump (specs/05 D7).
    """

    model: str
    prompt_version: str
    temperature: float
    latency_ms: int
    raw_response: str
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
