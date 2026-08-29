"""Portfolio mode reaching the rest of the machinery.

A second engine is only useful if walk-forward, robustness, the overfitting
detector and the evaluator all run against it. Two things had to be true for
that, and neither was free.

**One dispatch point.** Eight call sites ran ``run_backtest`` directly. Each one
would have needed its own ``if spec.mode ==`` and one of them would eventually
have been missed -- silently, because running a portfolio spec through the
event-driven engine does not crash, it just simulates a strategy nobody wrote.

**Parameters the perturbation test can find.** ``slots()`` looked at ``entry``,
``regime``, ``signal_exit`` and the exit/sizing knobs. A portfolio spec has none
of those: its parameters live in ``rank_by``, ``screen``, ``hold`` and
``rebalance_every``. Left alone, every perturbation would have returned the
identical strategy, every result would have been identical, and
``parameter_stability`` -- 15% of the score -- would have been a free 100 for
being unperturbable. That is worse than not measuring it.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig, BacktestResult
from aqr.backtest.portfolio import PortfolioResult
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict
from aqr.validation.params import get_param, set_param, slots
from aqr.validation.walkforward import run_walk_forward

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
N = 900
T0 = 1_400_000_000


def _data(n: int = N) -> dict[str, Bars]:
    rng = np.random.default_rng(7)
    t = np.arange(T0, T0 + n * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, symbol in enumerate(SYMBOLS):
        steps = rng.normal(0.0004, 0.013, n) + 0.15 * np.sin(
            np.arange(n) * 2 * np.pi / 90.0 + i
        ) / 90.0
        close = 100.0 * np.exp(np.cumsum(steps))
        out[symbol] = Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.012,
            low=close * 0.988,
            close=close,
            volume=np.full(n, 2e6),
        )
    return out


def _portfolio_spec(**over: object):
    body: dict[str, object] = {
        "name": "xs_probe",
        "mode": "portfolio",
        "rank_by": "roc(60) - roc(5)",
        "hold": 3,
        "rebalance_every": 10,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "sleeve": {"budget": 0.20, "idle": "benchmark"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


def _signal_spec():
    return spec_from_dict(
        {
            "strategy": {
                "name": "trigger_probe",
                "entry": "close > ema(20)",
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
            }
        }
    )


# --------------------------------------------------------------------------
# Dispatch


def test_run_strategy_picks_the_engine_from_the_spec() -> None:
    data = _data()
    assert isinstance(run_strategy(_portfolio_spec(), data), PortfolioResult)
    assert not isinstance(run_strategy(_signal_spec(), data), PortfolioResult)


def test_a_portfolio_result_is_usable_wherever_a_backtest_result_is() -> None:
    """Downstream code -- metrics, walk-forward stitching, the overfitting
    detector -- reads ``equity``, ``timeline``, ``trades`` and ``warmup_bars``.
    Making the portfolio result a kind of backtest result means none of it
    needs to know there are two engines."""
    result = run_strategy(_portfolio_spec(), _data())
    assert isinstance(result, BacktestResult)
    from aqr.backtest.metrics import compute_metrics

    metrics = compute_metrics(result)
    assert metrics.num_trades == len(result.trades)


def test_the_dispatcher_passes_the_peer_universe_through() -> None:
    """Asset robustness shrinks the traded set to one name while holding the
    market definition fixed. A dispatcher that dropped ``peers`` would make
    every cross-sectional feature undefined and score the rule at zero."""
    data = _data()
    one = _portfolio_spec(universe={"symbols": ["AAA"], "timeframe": "1D"}, hold=1)
    result = run_strategy(one, {"AAA": data["AAA"]}, peers=data)
    assert result.equity.size > 0


# --------------------------------------------------------------------------
# Parameters


def test_a_portfolio_spec_exposes_its_parameters() -> None:
    found = {slot.path for slot in slots(_portfolio_spec())}
    assert "hold" in found
    assert "rebalance_every" in found
    assert any(path.startswith("rank_by#") for path in found), found


def test_a_signal_spec_exposes_no_portfolio_parameters() -> None:
    found = {slot.path for slot in slots(_signal_spec())}
    assert "hold" not in found
    assert "rebalance_every" not in found


def test_the_portfolio_parameters_round_trip() -> None:
    spec = _portfolio_spec()
    assert get_param(spec, "hold") == 3
    changed = set_param(spec, "hold", 5)
    assert changed.hold == 5
    assert get_param(changed, "rebalance_every") == 10


def test_a_ranking_literal_can_be_perturbed() -> None:
    spec = _portfolio_spec()
    changed = set_param(spec, "rank_by#0", 90)
    assert "90" in str(changed.rank_by)
    assert changed.rank_by != spec.rank_by


def test_perturbing_hold_keeps_it_a_whole_number() -> None:
    """A book cannot hold 4.7 names, and a float here would be silently
    truncated somewhere further down."""
    changed = set_param(_portfolio_spec(), "hold", 4.7)
    assert isinstance(changed.hold, int)
    assert changed.hold == 5


def test_a_perturbation_that_would_be_invalid_is_refused_here() -> None:
    with pytest.raises(ValueError):
        set_param(_portfolio_spec(), "hold", 0)


# --------------------------------------------------------------------------
# Walk-forward


def test_walk_forward_runs_a_portfolio_spec() -> None:
    report = run_walk_forward(
        _portfolio_spec(),
        _data(),
        train_bars=400,
        test_bars=150,
        config=BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True),
    )
    assert report.folds, "no folds were produced"
    assert report.stitched is not None
    assert report.stitched_equity is not None
    assert report.stitched_equity.size > 1
