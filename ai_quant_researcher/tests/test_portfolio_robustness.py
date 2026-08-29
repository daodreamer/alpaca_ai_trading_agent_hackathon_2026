"""Asset robustness for a strategy whose unit of decision is the universe.

The question section 14.3 asks is "is this an edge, or a property of one
ticker?", and for a trigger rule the way to ask it is to trade one name at a
time. The README already records what happened when that machinery met
peer-relative features: ranked against itself every cross-sectional feature is
undefined, the rule fires on nothing, and the score comes out 0.0 -- an active
penalty for using the feature, on a tenth of the total.

Portfolio mode brings the same failure back in a new shape, and the first real
run showed it: ``asset_robustness 0.0`` on a strategy with a +5.5% alpha. The
peer fix is not enough here, because a book that holds ten names cannot be run
on one. Weight per name is ``core_budget / hold``, so a single-symbol run puts
8% in the only name available, leaves the sleeve with nowhere to go, and
measures a portfolio that is 72% cash. Whatever that number is, it is not an
answer to "does this edge depend on one ticker".

So for portfolio mode the question is asked the way it makes sense: **leave one
out**. Re-run the strategy on the universe minus a name, once per name. If the
edge survives every deletion it does not live in any single ticker; if removing
one name destroys it, the finding was that name.
"""

from __future__ import annotations

import numpy as np

from aqr.backtest.engine import BacktestConfig
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict
from aqr.validation.robustness import asset_robustness

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N = 500
T0 = 1_400_000_000


def _data(n: int = N) -> dict[str, Bars]:
    rng = np.random.default_rng(11)
    t = np.arange(T0, T0 + n * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, symbol in enumerate(SYMBOLS):
        steps = rng.normal(0.0004, 0.012, n) + np.sin(
            np.arange(n) * 2 * np.pi / 70.0 + i
        ) * 0.0015
        close = 100.0 * np.exp(np.cumsum(steps))
        out[symbol] = Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.011,
            low=close * 0.989,
            close=close,
            volume=np.full(n, 2e6),
        )
    return out


def _spec(**over: object):
    body: dict[str, object] = {
        "name": "xs_probe",
        "mode": "portfolio",
        "rank_by": "roc(40) - roc(5)",
        "hold": 3,
        "rebalance_every": 10,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def test_a_portfolio_strategy_is_not_scored_on_single_symbol_runs() -> None:
    """The bug: a book that holds ten names cannot be run on one, and the number
    that comes back is not an answer to the question that was asked."""
    report = asset_robustness(_spec(), _data(), config=CONFIG)
    assert set(report.per_symbol) == set(SYMBOLS)
    assert report.score > 0.0, "leave-one-out must produce a real measurement"


def test_each_entry_is_the_universe_without_that_name() -> None:
    """Keyed by the symbol that was *removed*, so a low score points at the name
    the edge depended on rather than at the name that was tested."""
    report = asset_robustness(_spec(), _data(), config=CONFIG)
    assert report.per_symbol["AAA"].num_trades > 0
    # Six names, hold three: dropping one still leaves a real book.
    assert all(m.num_trades > 0 for m in report.per_symbol.values())


def test_dropping_a_name_the_edge_lives_in_shows_up() -> None:
    """One name carries the whole result; removing it must move the score.

    Constructed rather than hoped for: five flat series and one that trends, so
    the ranking has exactly one thing to find.
    """
    data = _data()
    t = np.arange(T0, T0 + N * 86_400, 86_400, dtype=np.int64)
    flat = 100.0 * np.ones(N)
    for symbol in SYMBOLS[1:]:
        data[symbol] = Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=t,
            open=flat,
            high=flat * 1.001,
            low=flat * 0.999,
            close=flat,
            volume=np.full(N, 2e6),
        )
    full = asset_robustness(_spec(hold=1), data, config=CONFIG)
    assert full.per_symbol["AAA"].total_return != full.per_symbol["BBB"].total_return


def test_signal_mode_still_runs_one_symbol_at_a_time() -> None:
    """The old question is right for a trigger rule and is left alone."""
    spec = spec_from_dict(
        {
            "strategy": {
                "name": "trigger",
                "entry": "close > ema(20)",
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
            }
        }
    )
    report = asset_robustness(spec, _data(), config=CONFIG)
    assert set(report.per_symbol) <= set(SYMBOLS)


def test_a_two_name_universe_cannot_be_measured_this_way() -> None:
    """Leave-one-out on two names leaves one, and a one-name 'portfolio' is the
    degenerate case this whole file exists to avoid. Report nothing rather than
    a number that means nothing."""
    data = {s: b for s, b in _data().items() if s in ("AAA", "BBB")}
    spec = _spec(universe={"symbols": ["AAA", "BBB"], "timeframe": "1D"}, hold=1)
    report = asset_robustness(spec, data, config=CONFIG)
    assert report.per_symbol == {}
    assert report.score == 0.0
