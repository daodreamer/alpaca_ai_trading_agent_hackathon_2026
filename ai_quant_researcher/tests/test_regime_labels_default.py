"""Where regime labels come from when nobody supplies them.

`regime_robustness` is 10% of the score and needs a label per bar. The README
records what happened the first time nobody supplied them: only the simulator
could, so on real data the pipeline passed none, the report was never built, and
the evaluator substituted 0.5. A tenth of every real-data score was a constant
standing in for a measurement, and the experiment records show it plainly --
`"regime_robustness": 50.0`, on every single one.

That was fixed by writing a classifier. The hole it left is narrower and has the
same shape: `evaluate_candidate` still takes `regime_labels` as an optional
argument and still falls back to 0.5 when it is absent. The CLI passes them; a
caller that does not is silently scored on a placeholder. It happened again in
the first real portfolio run, from a script that simply did not know to pass
them -- `regime_robustness 50.0`, once more, with nothing in the output saying
the number was invented.

The labels are derivable from the bars alone. There is no reason for a caller to
have to supply what the pipeline can compute, and making it optional is exactly
what allowed the placeholder. So the pipeline computes them; the argument stays
as an override for the simulator, which knows the regime it generated and can
therefore supply ground truth rather than an estimate.
"""

from __future__ import annotations

import numpy as np

from aqr.backtest.engine import BacktestConfig
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict
from aqr.pipeline import evaluate_candidate

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N = 1200
T0 = 1_300_000_000


def _data() -> dict[str, Bars]:
    rng = np.random.default_rng(3)
    t = np.arange(T0, T0 + N * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, symbol in enumerate(SYMBOLS):
        steps = rng.normal(0.0004, 0.013, N) + np.sin(
            np.arange(N) * 2 * np.pi / 120.0 + i
        ) * 0.0012
        close = 100.0 * np.exp(np.cumsum(steps))
        out[symbol] = Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.012,
            low=close * 0.988,
            close=close,
            volume=np.full(N, 2e6),
        )
    return out


def _spec():
    return spec_from_dict(
        {
            "strategy": {
                "name": "xs_probe",
                "mode": "portfolio",
                "rank_by": "roc(40) - roc(5)",
                "hold": 2,
                "rebalance_every": 10,
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
            }
        }
    )


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def _run(**over: object):
    return evaluate_candidate(
        _spec(), _data(), config=CONFIG, train_bars=500, test_bars=200, **over
    )


def test_labels_are_computed_when_the_caller_supplies_none() -> None:
    """The bug: a caller that does not know to pass labels was scored on 0.5."""
    outcome = _run()
    assert outcome.regimes is not None, "no regime report was built"


def test_the_score_is_a_measurement_not_the_placeholder() -> None:
    outcome = _run()
    assert outcome.evaluation is not None
    component = outcome.evaluation.components["regime_robustness"]
    assert outcome.regimes is not None
    assert component == 50.0 * 2 * outcome.regimes.score or component != 50.0, (
        "regime_robustness is exactly the 0.5 placeholder again"
    )


def test_supplied_labels_still_win() -> None:
    """The argument stays, for the simulator: it knows the regime it generated,
    so it can supply truth where the classifier can only estimate."""
    data = _data()
    forced = {symbol: ["TREND_BULL"] * len(bars) for symbol, bars in data.items()}
    outcome = evaluate_candidate(
        _spec(),
        data,
        config=CONFIG,
        train_bars=500,
        test_bars=200,
        regime_labels=forced,
    )
    assert outcome.regimes is not None
    assert set(outcome.regimes.per_regime) <= {"TREND_BULL"}


def test_the_labels_are_the_classifier_the_project_already_has() -> None:
    """Not a second implementation. A regime classifier that disagreed with
    `aqr regimes` would make the score unexplainable from the CLI."""
    from aqr.features.regime import regime_series

    data = _data()
    outcome = evaluate_candidate(
        _spec(), data, config=CONFIG, train_bars=500, test_bars=200
    )
    assert outcome.regimes is not None
    expected = set(regime_series(data["AAA"]))
    assert set(outcome.regimes.per_regime) <= expected | {"UNKNOWN"}
