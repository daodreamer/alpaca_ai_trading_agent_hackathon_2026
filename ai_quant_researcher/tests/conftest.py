"""Shared fixtures.

All tests run on the synthetic provider. That is a deliberate constraint, not a
convenience: a test suite that reaches the network is a test suite that fails for
reasons unrelated to the code, and one that depends on live market data cannot be
re-run in a year and give the same answer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.providers import SyntheticProvider
from aqr.dsl.loader import loads
from aqr.dsl.schema import StrategySpec

START = datetime(2010, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOLS = ("SPY", "QQQ", "IWM")


@pytest.fixture(scope="session")
def provider() -> SyntheticProvider:
    return SyntheticProvider()


@pytest.fixture(scope="session")
def spy(provider: SyntheticProvider) -> Bars:
    return provider.load("SPY", START, END)


@pytest.fixture(scope="session")
def universe(provider: SyntheticProvider) -> dict[str, Bars]:
    return {s: provider.load(s, START, END) for s in SYMBOLS}


@pytest.fixture(scope="session")
def regimes(provider: SyntheticProvider) -> dict[str, list[str]]:
    return {s: provider.regimes(s, START, END) for s in SYMBOLS}


@pytest.fixture
def spec() -> StrategySpec:
    return loads(
        """
        strategy:
          name: test_pullback
          hypothesis: Pullbacks in an uptrend resolve upward.
          universe: {symbols: [SPY], timeframe: 1D}
          regime: close > ema(200)
          entry: close <= ema(20) * 1.01 and rsi(14) > 40
          exit:
            stop_loss: {type: atr, multiplier: 2.0, period: 14}
            take_profit: {type: risk_reward, ratio: 2.0}
            max_holding_bars: 20
          sizing: {risk_per_trade: 0.01, max_position_pct: 0.25}
        """
    )


def make_bars(
    closes: list[float],
    *,
    symbol: str = "TEST",
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    volume: float = 1_000_000.0,
    start: datetime = datetime(2020, 1, 1, tzinfo=UTC),
) -> Bars:
    """Hand-built bars for tests that need an exactly known price path."""
    n = len(closes)
    close = np.array(closes, dtype=np.float64)
    open_ = np.array(opens, dtype=np.float64) if opens else close.copy()
    high = np.array(highs, dtype=np.float64) if highs else np.maximum(open_, close)
    low = np.array(lows, dtype=np.float64) if lows else np.minimum(open_, close)
    stamps = np.array(
        [int(start.timestamp()) + i * 86_400 for i in range(n)], dtype=np.int64
    )
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=stamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=np.full(n, volume, dtype=np.float64),
    )
