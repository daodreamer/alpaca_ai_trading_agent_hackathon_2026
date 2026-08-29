"""Fixtures for the agent suite. Offline, always.

Nothing here opens a socket. The proposer is a stub or a recorded one, the
session is `RecordedSession` replaying the payloads captured on 2026-08-26, and
`as_of` is a constant — which together are what make specs/05 D7's determinism
claim testable rather than asserted.

The `StubProposer` is worth a word. It returns whatever it is told to, including
shapes a real model should never produce: an index past the end of the menu, a
`None` with a rationale, an error with no choice. Those are the inputs specs/05
D3 and D6 are about, and a suite that could only produce well-behaved model
output would test none of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Final

import pytest

from alphagate.agent import Candidate, Choice, MarketRead, ModelCall, Setup
from alphagate.agent.candidates import build_candidates, vertical_credit_spreads
from alphagate.agent.proposer import Proposal
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import Timeframe
from alphagate.options import (
    Greeks,
    OptionContract,
    OptionQuote,
    Right,
    compute_risk,
)
from alphagate.risk import DEFAULT_LIMITS, PortfolioSnapshot, RiskLimits

SPY: Ticker = ticker("SPY")
NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)
EXPIRY = date(2026, 9, 4)  # 9 DTE — inside the (3, 21) window
EQUITY = Decimal(100_000)
SPOT = Decimal("766.00")

GREEKS = Greeks(delta=-0.20, gamma=0.02, theta=-0.08, vega=0.15, rho=-0.03, iv=0.19)

STRIKE_DELTAS = {
    "757": -0.40,
    "752": -0.35,
    "747": -0.28,
    "742": -0.22,
    "737": -0.17,
    "732": -0.13,
}
"""Per-strike deltas, falling as the strikes go further out of the money.

The first version of this fixture used one `Greeks` for every strike, which made
every spread's net delta exactly zero — the two legs cancelled — and quietly
made every delta-budget test vacuous. A chain where the skew is flat is not a
chain."""


def contract(strike: str, right: Right = Right.PUT) -> OptionContract:
    return OptionContract(SPY, EXPIRY, Decimal(strike), right)


def quote(
    c: OptionContract,
    bid: str,
    ask: str,
    *,
    age: int = 0,
    greeks: Greeks | None = GREEKS,
) -> OptionQuote:
    return OptionQuote(c, NOW - timedelta(seconds=age), Decimal(bid), Decimal(ask), greeks)


def strike_greeks(strike: str, base: Greeks | None = GREEKS) -> Greeks | None:
    """`base` with this strike's own delta. `None` stays `None`."""
    if base is None:
        return None
    from dataclasses import replace

    return replace(base, delta=STRIKE_DELTAS.get(strike, base.delta))


def put_chain(*, age: int = 0, greeks: Greeks | None = GREEKS) -> dict[OptionContract, OptionQuote]:
    """A 5-wide ladder of SPY puts below spot, with plausible prices and skew.

    Prices decline as strikes fall, which is what makes every 5-wide pair a
    genuine credit spread rather than an accident of made-up numbers; deltas
    decline with them, which is what makes the net delta of a spread a real
    number rather than zero.
    """
    ladder = {
        "757": ("4.10", "4.16"),
        "752": ("3.20", "3.25"),
        "747": ("2.60", "2.65"),
        "742": ("2.05", "2.10"),
        "737": ("1.60", "1.64"),
        "732": ("1.25", "1.29"),
    }
    return {
        (c := contract(strike)): quote(
            c, bid, ask, age=age, greeks=strike_greeks(strike, greeks)
        )
        for strike, (bid, ask) in ladder.items()
    }


@dataclass(frozen=True)
class FakeTrend:
    """The shape the screen and the prompt actually read off a `TrendState`.

    The first version of this fixture used the bare string `"UPTREND"`, which
    has no `.state` and no `.strength` — so `DefaultScreen` read the strength as
    zero and refused every setup, and the runner tests failed for a reason that
    had nothing to do with the runner. A stand-in that does not have the shape
    of the thing it stands in for tests the code's error handling, not its
    behaviour.
    """

    phase: str = "BULLISH"
    strength: float = 66.7
    confidence: float = 1.0
    reason_codes: tuple[str, ...] = ("EMA_ALIGNMENT_BULL", "STRUCTURE_BULL", "RSI_BULL")
    timeframe: object = Timeframe.D1

    @property
    def state(self) -> object:
        return SimpleNamespace(value=self.phase, name=self.phase)


UNSET: Final = object()
"""Distinguishes "the caller said nothing" from "the caller said None".

`read(trend=None)` has to mean *unmeasured* — half this suite is about that
distinction — so the default cannot also be `None`."""


def read(
    *,
    iv_rank: str | None = "62",
    iv_vs_hv: str | None = "1.15",
    trend: object = UNSET,
    earnings: bool | None = False,
    as_of: datetime = NOW,
) -> MarketRead:
    return MarketRead(
        underlying=SPY,
        as_of=as_of,
        atr_pct=Decimal("0.9"),
        iv_rank=None if iv_rank is None else Decimal(iv_rank),
        iv_vs_hv=None if iv_vs_hv is None else Decimal(iv_vs_hv),
        earnings_within_dte=earnings,
        spot=SPOT,
        trend=FakeTrend() if trend is UNSET else trend,
    )


def setup(bias: str = "bullish") -> Setup:
    return Setup(
        underlying=SPY,
        name=f"high_iv_rank_{bias}",
        bias=bias,
        reason="iv_rank 62 with an intact uptrend; sell put premium below support",
    )


def menu(
    *,
    limits: RiskLimits = DEFAULT_LIMITS,
    equity: Decimal = EQUITY,
    age: int = 0,
    greeks: Greeks | None = GREEKS,
    as_of: datetime = NOW,
) -> tuple[Candidate, ...]:
    quotes = put_chain(age=age, greeks=greeks)
    structures = vertical_credit_spreads(
        quotes, right=Right.PUT, width=Decimal(5), as_of=as_of
    )
    return build_candidates(
        structures, limits=limits, equity=equity, as_of=as_of, limit=12
    )


def book(
    *,
    equity: Decimal = EQUITY,
    drawdown: str = "0.00",
    fills_today: int = 0,
    killswitch_tripped: bool = False,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity,
        positions=(),
        drawdown_pct=Decimal(drawdown),
        fills_today=fills_today,
        killswitch_tripped=killswitch_tripped,
    )


def risk_for(strikes: tuple[str, str], *, as_of: datetime = NOW) -> Any:
    quotes = put_chain()
    short, long = contract(strikes[0]), contract(strikes[1])
    from alphagate.options import Leg, OptionStructure, Side, StructureKind

    structure = OptionStructure(
        StructureKind.VERTICAL_CREDIT, (Leg(short, Side.SELL), Leg(long, Side.BUY))
    )
    return compute_risk(structure, {short: quotes[short], long: quotes[long]}, as_of)


@dataclass
class StubProposer:
    """Returns a scripted `Choice`, including ones a real model should not send.

    `seen` records what it was shown, which is how the suite asserts that a
    candidate was dropped *before* proposal rather than after — the difference
    specs/05 D6 turns on.
    """

    choice: Choice = field(default_factory=lambda: Choice(0, "taking the top of the menu", 0.7))
    error: str | None = None
    model: str = "stub-model"
    seen: list[tuple[str, int]] = field(default_factory=list)
    reads: list[MarketRead] = field(default_factory=list)

    def propose(
        self, read: MarketRead, candidates: Any, *, cycle_id: str
    ) -> Proposal:
        self.seen.append((cycle_id, len(candidates)))
        self.reads.append(read)
        return Proposal(
            choice=self.choice,
            call=ModelCall(
                model=self.model,
                prompt_version="test",
                temperature=0.0,
                latency_ms=7,
                raw_response=self.choice.rationale,
                error=self.error,
            ),
        )


@dataclass
class ExplodingProposer:
    """Raises. Nothing in the cycle should let this escape without a record."""

    def propose(self, read: MarketRead, candidates: Any, *, cycle_id: str) -> Proposal:
        raise RuntimeError("the model layer blew up")


@pytest.fixture
def limits() -> RiskLimits:
    return DEFAULT_LIMITS
