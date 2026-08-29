"""Shared fixtures for the equity suite — specs/09.

The baseline is a book that loads cleanly and a plan that trades nothing: three
symbols, weights that already match what is held, and prices that make the
arithmetic checkable in your head. Each test then breaks or moves exactly one
thing.

Numbers are chosen so nothing needs a calculator. Equity is $100,000, so a 0.10
weight is a $10,000 target; AAA is $100 a share, so that is 100 shares. The
default no-trade band is 0.25% of equity, which is $250 — two and a half shares
of AAA, and every band test below is written against that.

The weights are small — 0.10 / 0.06 / 0.04 rather than 0.5 / 0.3 / 0.2 — because
the real book's largest position is 8.2% and the Gate's concentration cap is
15%. A fixture whose largest name was half the account would have been vetoed by
`position_cap` on every single test, which is a fixture testing the fixture.

The book payload is a *dict*, deliberately, and it is shaped like the real
artefact `ai_quant_researcher` writes rather than like whatever the loader
happens to need. `sample_book_payload` is checked against the committed target
book by `test_book.py::test_the_fixture_matches_the_real_artefact_shape`, so a
schema change upstream fails this suite rather than production.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from alphagate.core.identifiers import Ticker, ticker
from alphagate.equity import (
    DEFAULT_EQUITY_POLICY,
    EquityPolicy,
    Holding,
    Mark,
    TargetBook,
    load_target_book,
)
from alphagate.risk import EquityPortfolio

AAA: Ticker = ticker("AAA")
BBB: Ticker = ticker("BBB")
CCC: Ticker = ticker("CCC")

FINGERPRINT = "3f6e2c8a9309068b"
BOOK_AS_OF = date(2026, 8, 27)
NOW = datetime(2026, 8, 28, 13, 45, tzinfo=UTC)
EQUITY = Decimal(100_000)


@pytest.fixture
def policy() -> EquityPolicy:
    return DEFAULT_EQUITY_POLICY


@pytest.fixture
def book_payload() -> dict[str, Any]:
    """A valid artefact, in the shape `aqr target-book` writes."""
    return {
        "schema_version": 1,
        "generated_at": "2026-08-28T07:17:06.502329+00:00",
        "spec_fingerprint": FINGERPRINT,
        "spec_name": "rs_volatility_consistency_neutral_v1",
        "spec_version": 1,
        "as_of": BOOK_AS_OF.isoformat(),
        "as_of_event_time": 1787803200,
        "dataset_version": "csv:1D:2023-05-17:2026-08-29",
        "universe": "sp500_pit",
        "timeframe": "1D",
        "symbols_loaded": 604,
        "symbols_declared": 680,
        "gross": 0.2,
        "positions": 3,
        "weights": {"AAA": 0.1, "BBB": 0.06, "CCC": 0.04},
        "core_weights": {"AAA": 0.08, "BBB": 0.04},
        "sleeve_weights": {"AAA": 0.02, "BBB": 0.02, "CCC": 0.04},
        "seal": {"phase": "research", "tainted": True, "loads": 0},
        "provenance": {
            "status": "PAPER",
            "score": 83.56,
            "hypothesis": "persistent relative strength, low volatility",
            "distinct_hypotheses": 324,
            "sealed_look": 1,
            "sealed_looks_total": 1,
            "preregistration": {
                "declared_at": "2026-08-28T22:27:02.062757+00:00",
                "seal_digest": "e74496c02cdc8200",
                "selection_rule": "Sole ACCEPT of campaign 07",
            },
            "sealed_measurement": {
                "refuted": False,
                "can_confirm": False,
                "strategy_return": 0.5638939447062332,
                "strategy_sharpe": 1.859964456248445,
                "benchmark_sharpe": 0.9760919924116748,
                "max_drawdown": -0.10395816342147657,
                "trades": 561,
                "observations": 498,
                "first_session": "2024-09-03T04:00:00+00:00",
                "last_session": "2026-08-27T04:00:00+00:00",
                "residual": {
                    "alpha": 0.1671764341583906,
                    "beta": 0.4346632991909543,
                    "t_alpha": 2.2202313842814925,
                    "information_ratio": 1.5823586266143104,
                    "is_significant": True,
                },
            },
        },
        "fill_convention": "decided at the prior close, filled at the open of as_of",
        "consumer_must_supply": ["account equity", "a kill switch"],
    }


@pytest.fixture
def book(book_payload: Mapping[str, Any]) -> TargetBook:
    return load_target_book(
        book_payload, pinned_fingerprint=FINGERPRINT, digest="deadbeef"
    )


@pytest.fixture
def marks() -> dict[Ticker, Mark]:
    """$100, $50 and $25 a share, all fresh and all fractionable."""
    return {
        AAA: Mark(AAA, Decimal(100), 1.0, tradeable=True, fractionable=True),
        BBB: Mark(BBB, Decimal(50), 1.0, tradeable=True, fractionable=True),
        CCC: Mark(CCC, Decimal(25), 1.0, tradeable=True, fractionable=True),
    }


@pytest.fixture
def holdings() -> list[Holding]:
    """Exactly the book: 100 × $100, 120 × $50, 160 × $25 on $100k of equity.

    $20,000 of the account, which is the book's gross of 0.2. The other $80,000
    is cash, so the buying-power tests have room to move and the gross-exposure
    test has to build its own fully-invested portfolio to have anything to
    refuse.
    """
    return [
        Holding(AAA, Decimal(100), Decimal(90), Decimal(10_000)),
        Holding(BBB, Decimal(120), Decimal(45), Decimal(6_000)),
        Holding(CCC, Decimal(160), Decimal(20), Decimal(4_000)),
    ]


def portfolio_for(
    holdings: list[Holding],
    marks: Mapping[Ticker, Mark],
    **overrides: Any,
) -> EquityPortfolio:
    """An `EquityPortfolio` over the fixture book, with one field moved at a time."""
    defaults: dict[str, Any] = {
        "equity": EQUITY,
        "cash": Decimal(0),
        "buying_power": Decimal(200_000),
        "holdings": tuple(holdings),
        "marks": {symbol: mark.price for symbol, mark in marks.items()},
        "drawdown_pct": Decimal(0),
        "orders_today": 0,
        "turnover_today": Decimal(0),
        "killswitch_tripped": False,
    }
    return EquityPortfolio(**{**defaults, **overrides})


@pytest.fixture
def portfolio(
    holdings: list[Holding], marks: dict[Ticker, Mark]
) -> EquityPortfolio:
    return portfolio_for(holdings, marks)
