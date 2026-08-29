"""The portfolio engine: always invested, ranked cross-sectionally, 80/20.

Every strategy this project has ever promoted lost to buy and hold, and the
reason was structural rather than a bad rule: an event-driven book that holds at
most a few names is out of the market most of the time, so it forfeits a share of
the drift it is being measured against. Winning on Sharpe then requires cutting
volatility by more than it cut return, which almost nothing does.

This engine takes the other form. It stays invested, ranks the universe, and
holds the top of it -- so beta is roughly one and the excess return is the
cross-sectional spread rather than a bet on being out at the right moments.

The capital is split by construction:

* **80% core.** Equal weight across the top ``hold`` names by ``rank_by``.
* **20% sleeve.** Reserved for event-driven deviations. Until that exists the
  sleeve holds the benchmark, which is what keeps the split from being a 20%
  cash drag -- at the benchmark's historical CAGR, idle cash would cost more per
  year than the entire realistic alpha budget, and the strategy would be
  structurally behind before the first rebalance.

The tests below pin the properties that make the result mean anything: fills on
the next bar, deterministic tie-breaking, NaN as absence rather than as a weak
signal, and prefix stability under truncation.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.portfolio import run_portfolio
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N = 260
STEP = 86_400
T0 = 1_500_000_000


def _bars(symbol: str, drift: float, n: int = N) -> Bars:
    """A deterministic ramp. Different drift per symbol, so the cross-sectional
    ranking has something unambiguous to sort on."""
    t = np.arange(T0, T0 + n * STEP, STEP, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1e6),
    )


def _rotating(n: int = N) -> dict[str, Bars]:
    """A universe whose leadership actually changes hands.

    ``_universe`` is monotone by construction so that the expected top-k is
    knowable without re-implementing the engine -- but nothing ever leaves the
    book there, so it cannot exercise a round-trip. Phase-shifted cycles give
    every name a turn at the top.
    """
    out: dict[str, Bars] = {}
    t = np.arange(T0, T0 + n * STEP, STEP, dtype=np.int64)
    for i, symbol in enumerate(SYMBOLS):
        phase = 2.0 * np.pi * i / len(SYMBOLS)
        close = 100.0 * np.exp(0.15 * np.sin(np.arange(n) * 2.0 * np.pi / 60.0 + phase))
        out[symbol] = Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=np.full(n, 1e6),
        )
    return out


def _universe(n: int = N) -> dict[str, Bars]:
    # Monotone in symbol order: AAA weakest, FFF strongest. Nothing crosses, so
    # the expected top-k is knowable without re-implementing the engine.
    return {s: _bars(s, 0.0002 * (i + 1), n) for i, s in enumerate(SYMBOLS)}


def _spec(**over: object) -> object:
    body = {
        "name": "xs_momentum",
        "mode": "portfolio",
        "rank_by": "roc(20)",
        "hold": 2,
        "rebalance_every": 20,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "sleeve": {"budget": 0.20, "idle": "benchmark"},
    }
    body.update(over)  # type: ignore[arg-type]
    return spec_from_dict({"strategy": body})


def _run(data: dict[str, Bars] | None = None, **over: object):
    return run_portfolio(
        _spec(**over),  # type: ignore[arg-type]
        data if data is not None else _universe(),
        BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True),
    )


# --------------------------------------------------------------------------
# The spec


def test_portfolio_mode_requires_a_ranking_expression() -> None:
    with pytest.raises(ValueError, match="rank_by"):
        _spec(rank_by=None)


def test_signal_mode_refuses_portfolio_fields() -> None:
    """The two engines are different machines. A spec that names fields from
    both has not chosen one, and guessing which it meant would silently run the
    wrong simulation."""
    with pytest.raises(ValueError, match="rank_by"):
        spec_from_dict(
            {
                "strategy": {
                    "name": "confused",
                    "entry": "close > ema(20)",
                    "rank_by": "roc(20)",
                    "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
                }
            }
        )


def test_the_ranking_expression_must_be_a_number_not_a_condition() -> None:
    """``rank_by: close > ema(20)`` sorts a book by True and False.

    It parses, it runs, and it produces an arbitrary two-tier ordering that looks
    like a ranking -- which is exactly the kind of mistake that has to fail loudly.
    """
    with pytest.raises(ValueError, match="number|condition"):
        _spec(rank_by="close > ema(20)")


def test_hold_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="hold"):
        _spec(hold=0)


def test_the_sleeve_budget_must_leave_a_core() -> None:
    with pytest.raises(ValueError, match="sleeve"):
        _spec(sleeve={"budget": 1.0, "idle": "benchmark"})


def test_the_spec_reports_the_features_its_ranking_needs() -> None:
    spec = _spec(rank_by="roc(60)", screen="rvol(20) > 1.0")
    names = {key.name for key in spec.features()}  # type: ignore[attr-defined]
    assert "roc" in names
    assert "rvol" in names


# --------------------------------------------------------------------------
# Always invested, and split 80/20


def test_the_book_is_fully_invested() -> None:
    """The whole point of the form. Cash is the drag that beat every previous
    strategy this project produced."""
    result = _run()
    invested = result.exposure[result.warmup_bars + 1 :]
    assert invested.size > 0
    assert np.allclose(invested, 1.0, atol=0.02)


def test_the_core_takes_eighty_percent_and_the_sleeve_twenty() -> None:
    result = _run()
    step = result.warmup_bars + 2
    assert result.core_weight[step] == pytest.approx(0.80, abs=0.02)
    assert result.sleeve_weight[step] == pytest.approx(0.20, abs=0.02)


def test_an_idle_sleeve_holds_the_benchmark_not_cash() -> None:
    """Idle cash at the benchmark's CAGR costs more per year than the entire
    realistic alpha budget. The sleeve is a deviation budget, not a cash bucket."""
    result = _run()
    step = result.warmup_bars + 2
    held = result.weights_at(step)
    sleeve_names = [s for s in SYMBOLS if held.get(s, 0.0) > 0]
    assert len(sleeve_names) == len(SYMBOLS), "every eligible name carries sleeve weight"


def test_the_core_holds_the_top_k_by_rank() -> None:
    result = _run()
    step = result.warmup_bars + 2
    core = result.core_weights_at(step)
    assert set(core) == {"EEE", "FFF"}, f"expected the two strongest, got {sorted(core)}"


def test_the_core_is_equally_weighted() -> None:
    result = _run()
    core = result.core_weights_at(result.warmup_bars + 2)
    assert len(set(round(w, 6) for w in core.values())) == 1


# --------------------------------------------------------------------------
# Selection rules


def test_a_nan_rank_is_absence_not_a_weak_signal() -> None:
    """A name whose ranking feature has not warmed up is not a bad name.

    Treating NaN as a low score would fill the book with whatever happened to
    have the longest history, which is a survivorship rule wearing a momentum
    label.
    """
    data = _universe()
    data["FFF"] = _bars("FFF", 0.001, n=15)  # never warms up: roc(20) needs 20 bars
    result = _run(data, hold=2, rebalance_every=20)
    core = result.core_weights_at(result.warmup_bars + 2)
    assert "FFF" not in core


def test_ties_are_broken_by_symbol_so_the_run_is_deterministic() -> None:
    flat = {s: _bars(s, 0.0) for s in SYMBOLS}
    core = _run(flat, hold=2).core_weights_at(60)
    assert sorted(core) == ["AAA", "BBB"]


def test_a_screen_removes_names_before_ranking() -> None:
    result = _run(screen="close < 100.5", hold=2)
    core = result.core_weights_at(result.warmup_bars + 2)
    assert "FFF" not in core, "the strongest name fails the screen and must not be held"


def test_fewer_eligible_names_than_hold_does_not_lever_up() -> None:
    """Holding two names' worth of capital in one name is a different strategy
    with twice the idiosyncratic risk, arrived at by accident."""
    result = _run(screen="close < 100.5", hold=5)
    core = result.core_weights_at(result.warmup_bars + 2)
    assert sum(core.values()) <= 0.80 + 1e-9
    assert all(w <= 0.80 / 5 + 1e-9 for w in core.values())


# --------------------------------------------------------------------------
# Timing


def test_the_book_only_changes_on_a_rebalance() -> None:
    result = _run(rebalance_every=20)
    first = result.warmup_bars + 2
    a = result.core_weights_at(first)
    b = result.core_weights_at(first + 5)
    assert set(a) == set(b)


def test_a_decision_at_bar_t_is_filled_at_the_next_open() -> None:
    """The invariant the whole project rests on. Filling at bar ``t``'s close on
    a decision taken from bar ``t`` is look-ahead with a rationale attached."""
    result = _run()
    assert result.first_fill_step is not None
    assert result.first_decision_step is not None
    assert result.first_fill_step == result.first_decision_step + 1


# --------------------------------------------------------------------------
# The properties that make a number trustworthy


def test_the_run_is_deterministic() -> None:
    a, b = _run(), _run()
    assert np.array_equal(a.equity, b.equity)
    assert [t.symbol for t in a.trades] == [t.symbol for t in b.trades]


def test_truncating_the_data_cannot_change_earlier_weights() -> None:
    """Prefix stability, end to end. If a future bar influenced a past holding,
    the two runs diverge before the cut."""
    full = _run(_universe(N))
    short = _run(_universe(N - 60))
    overlap = short.equity.size
    assert np.allclose(full.equity[:overlap], short.equity[:overlap], rtol=1e-9)


def test_leaving_the_book_produces_a_completed_trade() -> None:
    """The evaluator gates on out-of-sample trade count and profit factor. A
    rebalanced book still makes round-trips and has to report them, or every
    portfolio strategy is rejected for having no trades."""
    result = _run(_rotating(), rebalance_every=20, hold=2)
    assert result.trades, "a rebalancing book must emit round-trips"
    assert all(t.exit_time > t.entry_time for t in result.trades)


def test_turnover_is_charged() -> None:
    """A book that rebalances for free is the most flattering bug available."""
    from aqr.backtest.costs import ZERO_COST, CostModel

    free = run_portfolio(
        _spec(),  # type: ignore[arg-type]
        _universe(),
        BacktestConfig(
            initial_equity=1_000_000.0,
            allow_fractional_shares=True,
            costs=ZERO_COST,
        ),
    )
    costly = run_portfolio(
        _spec(),  # type: ignore[arg-type]
        _universe(),
        BacktestConfig(
            initial_equity=1_000_000.0,
            allow_fractional_shares=True,
            costs=CostModel(spread_bps=20.0, slippage_bps=20.0, commission_per_share=0.01),
        ),
    )
    assert costly.final_equity < free.final_equity


def test_the_benchmark_is_the_equal_weight_universe() -> None:
    """Beta has to cancel out of the comparison, so the benchmark is the same
    names the strategy chose from -- not an index that happens to be handy."""
    result = _run()
    assert result.benchmark_equity is not None
    assert result.benchmark_equity.shape == result.equity.shape
    assert result.benchmark_equity[0] == pytest.approx(result.equity[0])
