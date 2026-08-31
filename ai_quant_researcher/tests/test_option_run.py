"""``run_option_strategy`` / ``OptionMarket`` — the seam specs/10-options-research.md
D9's diagram draws between "option features / OptionSpec" and "options engine".

Reuses ``test_option_engine.py``'s hand-built world so the only thing under
test is the bundling, never the arithmetic: if ``run_option_strategy`` and
``run_option_backtest`` ever disagreed on the same inputs, every other options
test in this suite would already be lying about which one it exercises.
"""

from __future__ import annotations

from datetime import date

import pytest

from aqr.options.engine import run_option_backtest
from aqr.options.features import VolatilityHistory
from aqr.options.run import OptionMarket, run_option_strategy
from tests.test_option_engine import build_world, credit_spread_spec, trading_days


def test_run_option_strategy_matches_run_option_backtest_exactly() -> None:
    days = trading_days(200)
    chain, bars = build_world(sessions=days[5:30], days=days)
    spec = credit_spread_spec()
    market = OptionMarket(underlying=bars, chain=chain)

    direct = run_option_backtest(spec, chain, bars)
    via_market = run_option_strategy(spec, market)

    assert [t.as_dict() for t in direct.option_trades] == [
        t.as_dict() for t in via_market.option_trades
    ]
    assert direct.equity.tolist() == via_market.equity.tolist()


def test_run_option_strategy_passes_volatility_through() -> None:
    """A spec that reads ``iv_rank()`` needs ``volatility``, which only reaches
    the engine if the bundle actually carries it. Without it, the market path
    must refuse in exactly the way the direct call refuses -- the same
    ``ValueError`` ``options/features.py`` raises when a rule names a feature
    this run was not given the data for -- rather than swallowing it or
    silently producing no trades, which would hide a spec that could never
    have worked."""
    days = trading_days(120)
    chain, bars = build_world(sessions=days[5:6], days=days)
    spec = credit_spread_spec(entry="iv_rank() > 0")

    without_vol = OptionMarket(underlying=bars, chain=chain)
    with pytest.raises(ValueError, match="volatility_history"):
        run_option_strategy(spec, without_vol)

    with_vol = OptionMarket(
        underlying=bars,
        chain=chain,
        volatility=VolatilityHistory.from_rows(
            [
                {
                    "date": day.isoformat(),
                    "act_symbol": "SPY",
                    "iv_current": "60",
                    "iv_year_high": "80",
                    "iv_year_low": "10",
                }
                for day in days[:6]
            ]
        ),
    )
    # iv_rank() = (60-10)/(80-10)*100 = 71.4, so `iv_rank() > 0` fires and the
    # bundle now behaves exactly like the equivalent direct
    # ``run_option_backtest(..., volatility=...)`` call.
    result = run_option_strategy(spec, with_vol)
    assert len(result.option_trades) == 1


def test_with_chain_swaps_only_the_chain() -> None:
    days = trading_days(200)
    chain, bars = build_world(sessions=days[5:30], days=days)
    market = OptionMarket(underlying=bars, chain=chain)
    narrowed = market.with_chain(chain.slice_dates(days[5], days[10]))

    assert narrowed.underlying is market.underlying
    assert narrowed.volatility is market.volatility
    assert narrowed.chain.sessions != market.chain.sessions
    assert set(narrowed.chain.sessions) <= set(market.chain.sessions)


def test_a_market_with_an_empty_chain_produces_no_trades() -> None:
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    empty = chain.slice_dates(date(1999, 1, 1), date(1999, 1, 2))
    market = OptionMarket(underlying=bars, chain=empty)
    result = run_option_strategy(credit_spread_spec(), market)
    assert result.option_trades == []
