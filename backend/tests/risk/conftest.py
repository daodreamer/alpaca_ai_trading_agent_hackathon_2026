"""Shared fixtures for the Gate suite.

The baseline is a proposal that passes every check with room to spare, on a book
that is empty and healthy. Each test then breaks exactly one thing. That shape
matters: a test that fails for two reasons cannot tell you which check it is
actually exercising, and this suite exists to pin thirteen checks individually.

Numbers are hand-computed, exact, and small enough to check in your head:
equity $100,000, and a 150/155 call credit spread taking $1.50 for one contract,
so `max_loss = 5.00 * 100 - 150 = 350`.

Two ways to build a `StructureRisk` are offered on purpose. `spread_risk` runs
the real `compute_risk` over real quotes, and the happy-path tests use it so the
suite proves `options` and `risk` compose. `make_risk` builds the dataclass
directly, and the single-check tests use it so a test about the delta band can
set a delta without reverse-engineering a quote set that produces one.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.core.identifiers import Ticker, ticker
from alphagate.options import (
    Greeks,
    Leg,
    OptionContract,
    OptionQuote,
    OptionStructure,
    Right,
    Side,
    StructureKind,
    StructureRisk,
    compute_risk,
)
from alphagate.risk import (
    DEFAULT_LIMITS,
    Intent,
    OpenPosition,
    PortfolioSnapshot,
    RiskLimits,
    TradeProposal,
)

AAPL: Ticker = ticker("AAPL")
MSFT: Ticker = ticker("MSFT")
NOW = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
EXPIRY = date(2026, 9, 11)  # 10 DTE from NOW — comfortably inside (3, 21)
EQUITY = Decimal(100_000)

FLAT = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=0.25)


def contract(
    strike: str, right: Right = Right.CALL, *, underlying: Ticker = AAPL, expiry: date = EXPIRY
) -> OptionContract:
    return OptionContract(underlying, expiry, Decimal(strike), right)


def credit_spread(
    *, underlying: Ticker = AAPL, expiry: date = EXPIRY, qty: int = 1
) -> OptionStructure:
    """Short the 150 call, long the 155. Five dollars wide, $1.50 credit."""
    return OptionStructure(
        StructureKind.VERTICAL_CREDIT,
        (
            Leg(contract("150", underlying=underlying, expiry=expiry), Side.SELL, qty),
            Leg(contract("155", underlying=underlying, expiry=expiry), Side.BUY, qty),
        ),
    )


def spread_risk(
    structure: OptionStructure | None = None,
    *,
    greeks: Greeks | None = FLAT,
    age: int = 0,
    spread: str = "0.02",
    as_of: datetime = NOW,
) -> StructureRisk:
    """Real risk, from real quotes, through the real `compute_risk`.

    `spread` is the fractional bid/ask width applied to both legs. Sub-penny
    half-widths are deliberate: quantising them would make the resulting
    `worst_spread_pct` merely near the requested value, and a liquidity test
    wants it exactly on the number.
    """
    built = credit_spread() if structure is None else structure
    width = Decimal(spread)
    quotes = {}
    for leg, mid in zip(built.legs, (Decimal("2.00"), Decimal("0.50")), strict=True):
        half = mid * width / 2
        quotes[leg.contract] = OptionQuote(
            leg.contract,
            as_of - timedelta(seconds=age),
            bid=mid - half,
            ask=mid + half,
            greeks=greeks,
        )
    return compute_risk(built, quotes, as_of)


def make_risk(**overrides: object) -> StructureRisk:
    """The baseline risk with one field replaced. For single-check tests."""
    return replace(spread_risk(), **overrides)  # type: ignore[arg-type]


def forge_risk(**overrides: object) -> StructureRisk:
    """Build a `StructureRisk` that its own invariants would refuse.

    Used by exactly one test: `defined_risk` guards against a maximum loss that
    is not finite and positive, but specs/02 D4 already makes such a value
    unconstructible, so the only way to exercise the veto branch is to smuggle
    one past `__post_init__`. That the smuggling is this awkward *is* the point
    — it is a measure of how well the domain holds.
    """
    base = spread_risk()
    forged = object.__new__(StructureRisk)
    for field_name in StructureRisk.__slots__:
        object.__setattr__(forged, field_name, getattr(base, field_name))
    for field_name, value in overrides.items():
        object.__setattr__(forged, field_name, value)
    return forged


def proposal(
    *,
    structure: OptionStructure | None = None,
    risk: StructureRisk | None = None,
    quantity: int = 1,
    intent: Intent = Intent.OPEN,
    proposal_id: str = "p-0001",
    risk_as_of: datetime = NOW,
) -> TradeProposal:
    built = credit_spread() if structure is None else structure
    return TradeProposal(
        structure=built,
        risk=spread_risk(built) if risk is None else risk,
        quantity=quantity,
        intent=intent,
        rationale="IV rank 62, price rejected the 150 supply zone twice.",
        proposed_by="claude-opus-5",
        proposal_id=proposal_id,
        risk_as_of=risk_as_of,
    )


def position(
    *,
    max_loss: str = "350",
    underlying: Ticker = AAPL,
    greeks: Greeks | None = FLAT,
) -> OpenPosition:
    return OpenPosition(
        structure=credit_spread(underlying=underlying),
        quantity=1,
        max_loss=Decimal(max_loss),
        net_greeks=greeks,
        opened_at=NOW - timedelta(days=1),
    )


def snapshot(
    *,
    equity: Decimal = EQUITY,
    positions: tuple[OpenPosition, ...] = (),
    drawdown: str = "0.00",
    fills_today: int = 0,
    killswitch_tripped: bool = False,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=equity,
        positions=positions,
        drawdown_pct=Decimal(drawdown),
        fills_today=fills_today,
        killswitch_tripped=killswitch_tripped,
    )


@pytest.fixture
def limits() -> RiskLimits:
    return DEFAULT_LIMITS
