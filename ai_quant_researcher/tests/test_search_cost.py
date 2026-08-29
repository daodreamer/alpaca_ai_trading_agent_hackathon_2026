"""What counts as a trial in the multiple-comparisons denominator.

The deflated Sharpe subtracts what a best-of-N search would produce by luck
alone. N is meant to be the number of *selection events*: distinct hypotheses
you looked at and could have chosen between.

It was ``SUM(backtests_run)``, which is a different and much larger number. One
hypothesis costs about 57 backtests here -- an in-sample run, a frictionless
run, ten walk-forward folds, forty parameter perturbations, one per symbol for
asset robustness. None of those forty perturbations is a hypothesis anyone
chose; they are diagnostics on a hypothesis already chosen. Counting them
inflated N by a factor of 57, and ``sqrt(2 ln N)`` by about a third.

Erring toward scepticism is the intended direction of this module's error, and
it stays that way. But an error of a *stated* size in a *stated* direction is a
choice; an error that arrives because a variable named ``backtests_run`` was
convenient is an accident, and accidents do not stay in the direction you like.
"""

from __future__ import annotations

import math

import numpy as np

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.dsl.loader import loads
from aqr.dsl.schema import spec_from_dict
from aqr.pipeline import evaluate_candidate
from aqr.registry.db import ExperimentRecord, Registry
from aqr.validation.overfitting import deflated_sharpe
from aqr.validation.params import neighbours, slots
from aqr.validation.robustness import _DEFAULT_FACTORS

SPEC = """
strategy:
  name: {name}
  universe: {{symbols: [SPY], timeframe: 1D}}
  entry: rsi(14) < {threshold}
  exit: {{stop_loss: {{type: atr, multiplier: 2.0, period: 14}}, max_holding_bars: 20}}
  sizing: {{risk_per_trade: 0.005}}
"""


def _record(fingerprint: str, backtests: int) -> ExperimentRecord:
    return ExperimentRecord(
        fingerprint=fingerprint,
        strategy_name="x",
        symbols=("SPY",),
        timeframe="1D",
        data_start="",
        data_end="",
        dataset_version="test",
        backtests_run=backtests,
    )


class TestDistinctHypotheses:
    def test_it_counts_rules_not_backtests(self, tmp_path) -> None:
        with Registry(tmp_path / "r.sqlite") as registry:
            for spec_name, threshold in (("a", 30), ("b", 35), ("c", 40)):
                spec = loads(SPEC.format(name=spec_name, threshold=threshold))
                registry.upsert_strategy(spec)
                registry.record_experiment(_record(spec.fingerprint(), backtests=57))

            assert registry.total_backtests() == 171
            assert registry.distinct_hypotheses() == 3

    def test_re_evaluating_one_rule_is_one_hypothesis(self, tmp_path) -> None:
        """A rule the loop rediscovers is not a new place to have got lucky."""
        with Registry(tmp_path / "r.sqlite") as registry:
            spec = loads(SPEC.format(name="a", threshold=30))
            registry.upsert_strategy(spec)
            for _ in range(4):
                registry.record_experiment(_record(spec.fingerprint(), backtests=57))

            assert registry.distinct_hypotheses() == 1
            assert registry.total_backtests() == 228

    def test_a_failed_hypothesis_still_counts(self, tmp_path) -> None:
        """Especially a failed one. Forgetting the attempts that went nowhere is
        the whole mechanism by which a search looks luckier than it was."""
        with Registry(tmp_path / "r.sqlite") as registry:
            spec = loads(SPEC.format(name="dead", threshold=30))
            registry.upsert_strategy(spec)
            record = _record(spec.fingerprint(), backtests=0)
            record.error = "never fires"
            registry.record_experiment(record)

            assert registry.distinct_hypotheses() == 1

    def test_an_empty_registry_has_no_trials(self, tmp_path) -> None:
        with Registry(tmp_path / "r.sqlite") as registry:
            assert registry.distinct_hypotheses() == 0
            assert registry.total_backtests() == 0

    def test_it_only_ever_grows(self, tmp_path) -> None:
        with Registry(tmp_path / "r.sqlite") as registry:
            spec = loads(SPEC.format(name="a", threshold=30))
            registry.upsert_strategy(spec)
            registry.record_experiment(_record(spec.fingerprint(), backtests=1))
            before = registry.distinct_hypotheses()

            other = loads(SPEC.format(name="b", threshold=31))
            registry.upsert_strategy(other)
            registry.record_experiment(_record(other.fingerprint(), backtests=1))
            assert registry.distinct_hypotheses() > before


class TestTheSizeOfTheCorrection:
    def test_counting_diagnostics_as_trials_over_deflates(self) -> None:
        """The measured ratio: 8,705 backtests bought 153 hypotheses, and the
        difference is a third of the deflation term."""
        periods = 2500
        as_backtests = 1.0 - deflated_sharpe(1.0, 8705, periods)
        as_hypotheses = 1.0 - deflated_sharpe(1.0, 153, periods)

        assert as_backtests > as_hypotheses
        assert as_backtests / as_hypotheses == round(
            math.sqrt(math.log(8705) / math.log(153)), 6
        ) or as_backtests / as_hypotheses > 1.3

    def test_the_correction_does_not_flip_the_sign_of_the_scepticism(self) -> None:
        # Still harsh, deliberately: 153 independent tries over ten years of
        # daily bars produce a best-of Sharpe near 1.0 out of nothing at all.
        assert 1.0 - deflated_sharpe(1.0, 153, 2500) > 0.9


# --------------------------------------------------------------------------
# The same accounting, for a portfolio spec
# --------------------------------------------------------------------------
#
# Everything above was written when every strategy was signal mode. A portfolio
# spec has different knobs -- `rank_by`, `hold`, `rebalance_every` -- and the
# question has to be asked again of them, because the answer is not inherited:
# if a perturbation of `hold` were counted as a hypothesis, forty diagnostics
# on one rule would inflate the multiple-comparisons denominator forty-fold.

PORTFOLIO_SYMBOLS = [f"P{i:02d}" for i in range(8)]
_T0 = 1_451_952_000  # 2016-01-05, in seconds


def _portfolio_bars() -> dict[str, Bars]:
    rng = np.random.default_rng(31)
    t = np.arange(_T0, _T0 + 400 * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, sym in enumerate(PORTFOLIO_SYMBOLS):
        steps = rng.normal(0.0005, 0.012, 400) + np.sin(
            np.arange(400) * 2 * np.pi / 60.0 + i
        ) * 0.0015
        close = 100.0 * np.exp(np.cumsum(steps))
        out[sym] = Bars(
            symbol=sym, timeframe="1D", event_time=t,
            open=close * 0.999, high=close * 1.01, low=close * 0.99,
            close=close, volume=np.full(400, 1e6),
        )
    return out


def _portfolio_spec(**over):
    body = {
        "name": "xs", "mode": "portfolio", "rank_by": "roc(40) - roc(5)",
        "hold": 3, "rebalance_every": 10,
        "universe": {"symbols": PORTFOLIO_SYMBOLS, "timeframe": "1D"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


class TestAPortfolioSpecIsIdentifiedByItsOwnKnobs:
    """If two different rules shared a fingerprint the denominator would
    *under*-count, and a search would look narrower than it was."""

    def test_the_ranking_expression_is_part_of_the_identity(self) -> None:
        base = _portfolio_spec()
        assert _portfolio_spec(rank_by="roc(40) - roc(6)").fingerprint() != base.fingerprint()

    def test_how_many_names_it_holds_is_part_of_the_identity(self) -> None:
        base = _portfolio_spec()
        assert _portfolio_spec(hold=4).fingerprint() != base.fingerprint()

    def test_how_often_it_rebalances_is_part_of_the_identity(self) -> None:
        base = _portfolio_spec()
        assert _portfolio_spec(rebalance_every=11).fingerprint() != base.fingerprint()


class TestPerturbationsAreDiagnosticsNotHypotheses:
    def test_the_perturbations_would_count_if_they_were_recorded(self) -> None:
        """Establishes that the next test is not passing trivially. Each
        neighbour really is a distinct rule by fingerprint; what keeps it out of
        the denominator is that nobody chose between them, not that they happen
        to collide."""
        base = _portfolio_spec()
        variants = [
            variant
            for slot in slots(base)
            for variant in neighbours(base, slot, _DEFAULT_FACTORS)
        ]
        distinct = {v.fingerprint() for v in variants} - {base.fingerprint()}
        assert len(distinct) >= 10

    def test_evaluating_one_portfolio_rule_adds_one_hypothesis(self, tmp_path) -> None:
        """The acceptance. Many backtests, one selection event."""
        with Registry(tmp_path / "r.sqlite") as registry:
            outcome = evaluate_candidate(
                _portfolio_spec(),
                _portfolio_bars(),
                registry=registry,
                train_bars=150,
                test_bars=60,
                config=BacktestConfig(
                    initial_equity=1_000_000.0, allow_fractional_shares=True
                ),
            )
            assert outcome.backtests_run > 20
            assert registry.distinct_hypotheses() == 1


class TestEverySlotIsOneTheEngineReads:
    """A knob the engine ignores perturbs to a bit-identical backtest, and
    `parameter_stability` reads identical as *stable*. Six of a portfolio
    spec's ten slots were `exit` and `sizing` fields, which `run_portfolio`
    never consults -- two thirds of the stability mark paid out for being
    unperturbable.
    """

    def test_a_portfolio_spec_exposes_only_its_own_knobs(self) -> None:
        paths = {slot.path for slot in slots(_portfolio_spec())}
        assert "hold" in paths
        assert "rebalance_every" in paths
        assert any(p.startswith("rank_by") for p in paths)
        assert not [p for p in paths if p.startswith(("exit.", "sizing.", "entry"))]

    def test_a_signal_spec_does_not_expose_the_portfolio_knobs(self) -> None:
        """The mirror image, which was wrong in the same way: the signal engine
        has no ranking and no rebalance clock."""
        paths = {slot.path for slot in slots(loads(SPEC.format(name="a", threshold=30)))}
        assert "hold" not in paths
        assert "rebalance_every" not in paths
        assert any(p.startswith("exit.") for p in paths)

    def test_every_perturbation_actually_moves_the_book(self) -> None:
        """The property the two tests above exist to protect."""
        base = _portfolio_spec()
        data = _portfolio_bars()
        config = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)
        baseline = run_strategy(base, data, config).equity

        inert = []
        for slot in slots(base):
            for variant in neighbours(base, slot, _DEFAULT_FACTORS):
                if variant.fingerprint() == base.fingerprint():
                    continue
                if np.array_equal(run_strategy(variant, data, config).equity, baseline):
                    inert.append(slot.path)
        assert not inert, f"perturbing these changed nothing: {sorted(set(inert))}"
