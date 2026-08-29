"""Fixtures for the execution suite.

Two things here are worth knowing before reading the tests.

**The `GatedOrder`s are real.** There is no way to fake one — `risk.gate` is the
only module that can mint one (specs/03 D3) — so every order below is built by
running an actual proposal through an actual `evaluate`. That makes the fixtures
slightly heavier than a hand-rolled stub and it is the right trade: a test that
constructs its subject through a back door is testing the back door.

**The recorded payloads are real too.** Every file in `tests/fixtures/mcp/` was
captured from alpaca-mcp-server 2.3.0 against the paper account on 2026-08-26,
not hand-written from the documentation. Account identity is redacted (specs/06
D4); nothing else is touched, including the `_alpaca_mcp_security` envelope,
whose exact shape several tests assert on.

The suite never opens a socket or a subprocess. `RecordedSession` replays those
files, which is the whole point of the seam in adr/0002 D4.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.core.identifiers import Ticker, ticker
from alphagate.options import (
    Cover,
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
    Approved,
    GatedOrder,
    Intent,
    PortfolioSnapshot,
    TradeProposal,
    evaluate,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mcp"

SPY: Ticker = ticker("SPY")
CHEAP: Ticker = ticker("SOFI")
"""A low-priced underlying, used only for the covered call and the cash-secured
put.

Not an arbitrary choice. Both of those structures have a maximum loss close to
their full notional — the stock going to zero, or assignment at the strike — so
on a $700 underlying one contract risks ~$70,000. Against the 1%-of-equity
per-trade limit in specs/03 D5 that needs a $7M account, which means those two
kinds are *structurally unapprovable* on any book this competition will run.
That is a real property of the limits and not a testing inconvenience, so the
fixtures move to an underlying where the sizes are honest rather than moving the
limits to where the fixtures are convenient."""
NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)
EXPIRY = date(2026, 9, 4)  # 9 DTE from NOW — inside the (3, 21) window
EQUITY = Decimal(100_000)
FLAT = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=0.20)


def payload(name: str) -> str:
    """One recorded MCP response, verbatim."""
    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8")


def payload_json(name: str) -> dict:
    return json.loads(payload(name))


def contract(
    strike: str, right: Right = Right.PUT, *, underlying: Ticker = SPY
) -> OptionContract:
    return OptionContract(underlying, EXPIRY, Decimal(strike), right)


def quote(
    c: OptionContract, bid: str, ask: str, *, greeks: Greeks | None = FLAT
) -> OptionQuote:
    return OptionQuote(c, NOW, Decimal(bid), Decimal(ask), greeks)


# --------------------------------------------------------------------- #
# One structure of each kind, priced. specs/04 test plan item 2.
# --------------------------------------------------------------------- #


def put_credit_spread(qty: int = 1) -> tuple[OptionStructure, dict[OptionContract, OptionQuote]]:
    """Short the 752 put, long the 747. Five wide, 0.60 credit per share."""
    short, long = contract("752"), contract("747")
    structure = OptionStructure(
        StructureKind.VERTICAL_CREDIT, (Leg(short, Side.SELL, qty), Leg(long, Side.BUY, qty))
    )
    return structure, {
        short: quote(short, "3.20", "3.25"),
        long: quote(long, "2.60", "2.65"),
    }


def call_debit_spread() -> tuple[OptionStructure, dict[OptionContract, OptionQuote]]:
    """Long the 770 call, short the 775. Five wide, 2.36 debit per share."""
    long, short = contract("770", Right.CALL), contract("775", Right.CALL)
    structure = OptionStructure(
        StructureKind.VERTICAL_DEBIT, (Leg(long, Side.BUY), Leg(short, Side.SELL))
    )
    return structure, {
        long: quote(long, "8.15", "8.21"),
        short: quote(short, "5.80", "5.84"),
    }


def iron_condor() -> tuple[OptionStructure, dict[OptionContract, OptionQuote]]:
    put_long, put_short = contract("740"), contract("745")
    call_short, call_long = contract("790", Right.CALL), contract("795", Right.CALL)
    structure = OptionStructure(
        StructureKind.IRON_CONDOR,
        (
            Leg(put_long, Side.BUY),
            Leg(put_short, Side.SELL),
            Leg(call_short, Side.SELL),
            Leg(call_long, Side.BUY),
        ),
    )
    return structure, {
        put_long: quote(put_long, "1.90", "1.95"),
        put_short: quote(put_short, "2.30", "2.35"),
        call_short: quote(call_short, "2.10", "2.15"),
        call_long: quote(call_long, "1.70", "1.75"),
    }


def covered_call() -> tuple[OptionStructure, dict[OptionContract, OptionQuote]]:
    """Short the 22 call against 100 shares held at 20.00. Max loss 1,976."""
    c = contract("22", Right.CALL, underlying=CHEAP)
    structure = OptionStructure(
        StructureKind.COVERED_CALL,
        (Leg(c, Side.SELL),),
        cover=Cover(shares=100, basis=Decimal("20.00")),
    )
    return structure, {c: quote(c, "0.24", "0.25")}


def cash_secured_put() -> tuple[OptionStructure, dict[OptionContract, OptionQuote]]:
    """Short the 19 put, cash-secured. Max loss 1,876 — assignment to zero."""
    c = contract("19", underlying=CHEAP)
    structure = OptionStructure(
        StructureKind.CASH_SECURED_PUT,
        (Leg(c, Side.SELL),),
        cover=Cover(cash=Decimal(1_900)),
    )
    return structure, {c: quote(c, "0.24", "0.25")}


STRUCTURES = {
    StructureKind.VERTICAL_CREDIT: put_credit_spread,
    StructureKind.VERTICAL_DEBIT: call_debit_spread,
    StructureKind.IRON_CONDOR: iron_condor,
    StructureKind.COVERED_CALL: covered_call,
    StructureKind.CASH_SECURED_PUT: cash_secured_put,
}


# --------------------------------------------------------------------- #
# Gated orders — minted through the real Gate, never faked.
# --------------------------------------------------------------------- #


def risk_of(
    structure: OptionStructure, quotes: dict[OptionContract, OptionQuote]
) -> StructureRisk:
    return compute_risk(structure, quotes, NOW)


def gated(
    *,
    structure: OptionStructure | None = None,
    quotes: dict[OptionContract, OptionQuote] | None = None,
    risk: StructureRisk | None = None,
    quantity: int = 1,
    intent: Intent = Intent.OPEN,
    proposal_id: str = "spy-2026-08-26-752",
    equity: Decimal = EQUITY,
) -> GatedOrder:
    """Run a proposal through the Gate and return the approved order.

    Asserts approval rather than handling a veto: a test that silently received
    a `Vetoed` here would go on to test nothing at all.
    """
    if structure is None or quotes is None:
        structure, quotes = put_credit_spread(qty=1)
    proposal = TradeProposal(
        structure=structure,
        risk=risk if risk is not None else risk_of(structure, quotes),
        quantity=quantity,
        intent=intent,
        rationale="Short strike ~1 sigma below spot; defined risk; no event in the window.",
        proposed_by="claude-opus-5",
        proposal_id=proposal_id,
        risk_as_of=NOW,
    )
    book = PortfolioSnapshot(
        equity=equity, positions=(), drawdown_pct=Decimal(0), fills_today=0
    )
    verdict = evaluate(proposal, book, DEFAULT_LIMITS, NOW)
    assert isinstance(verdict, Approved), (
        "fixture proposal was vetoed: "
        f"{[c.name for c in verdict.checks if not c.passed]}"
    )
    return verdict.order


@pytest.fixture
def order() -> GatedOrder:
    """The baseline: a one-contract SPY put credit spread taking 0.60."""
    return gated()
