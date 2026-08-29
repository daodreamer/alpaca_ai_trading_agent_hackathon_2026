"""Reading the sealed window more than once, and paying for it honestly.

The one-shot rule is per *candidate*, not per window. That is deliberate and it
is what makes an ongoing research loop possible: the LLM proposes a genuinely new
hypothesis, that hypothesis earns its own sealed run, and a loop that could never
take a second look would have to stop after its first strategy.

What a second look costs is multiplicity. A window that has screened seven
candidates has done a seven-way selection, and the survivor of a seven-way screen
is a weaker claim than the survivor of a one-way screen -- the same
multiple-comparisons problem the search already accounts for, moved up a level.

So nothing here forbids the seventh look. Everything here counts it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aqr.backtest.alpha import SIGNIFICANCE_T, ResidualAlpha
from aqr.dsl.schema import StrategySpec, spec_from_dict
from aqr.registry.db import Registry
from aqr.validation.sealed import SealedMeasurement, multiplicity_bar


def _spec(name: str, roc: int) -> StrategySpec:
    return spec_from_dict(
        {
            "strategy": {
                "name": name,
                "hypothesis": "Leaders keep leading.",
                "mode": "portfolio",
                "rank_by": f"roc({roc})",
                "hold": 2,
                "rebalance_every": 20,
                "universe": {"symbols": ["SPY", "QQQ", "IWM"], "timeframe": "1D"},
            }
        }
    )


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Registry]:
    with Registry(tmp_path / "research.sqlite") as reg:
        yield reg


def _spend(reg: Registry, spec: StrategySpec, rule: str) -> str:
    fingerprint = reg.upsert_strategy(spec)
    reg.preregister(fingerprint, selection_rule=rule, seal_digest="a" * 64)
    reg.record_sealed_run(fingerprint, result={"measurement": {"refuted": False}})
    return fingerprint


def _measurement(t_alpha: float, looks: int) -> SealedMeasurement:
    return SealedMeasurement(
        strategy="probe",
        fingerprint="f" * 16,
        since=datetime(2024, 9, 1, tzinfo=UTC),
        first_session=datetime(2024, 9, 3, tzinfo=UTC),
        last_session=datetime(2026, 8, 27, tzinfo=UTC),
        observations=498,
        backtest_sessions=1200,
        strategy_return=0.5,
        strategy_sharpe=1.8,
        max_drawdown=-0.2,
        benchmark_return=0.4,
        benchmark_sharpe=1.5,
        trades=300,
        residual=ResidualAlpha(
            alpha=0.16,
            beta=0.43,
            t_alpha=t_alpha,
            information_ratio=1.58,
            residual_vol=0.10,
            excess_sharpe=0.3,
            r_squared=0.5,
            observations=498,
        ),
        looks=looks,
    )


# --------------------------------------------------------------------------
# The loop the project actually runs


def test_a_second_strategy_gets_its_own_sealed_run(registry: Registry) -> None:
    """The workflow this exists for: explore, validate, explore again.

    One-shot is per fingerprint. A new hypothesis is a new fingerprint, so it is
    entitled to its own reading -- otherwise the research loop would be a
    one-strategy loop.
    """
    first = _spend(registry, _spec("alpha_v1", 20), "best of campaign 08")
    second = _spend(registry, _spec("beta_v1", 40), "best of campaign 09")

    assert first != second
    assert registry.sealed_run(first) is not None
    assert registry.sealed_run(second) is not None
    assert registry.ancestry_taint(second).clean


def test_the_same_candidate_still_gets_only_one(registry: Registry) -> None:
    """The refusal that is not relaxed. Re-running one fingerprint until it
    passes is the thing the protocol exists to make impossible, and it is
    unaffected by the window being readable again for something else."""
    from aqr.registry.db import PreregistrationError

    fingerprint = _spend(registry, _spec("alpha_v1", 20), "best of campaign 08")
    with pytest.raises(PreregistrationError, match="no second one"):
        registry.record_sealed_run(fingerprint, result={"measurement": {}})


def test_the_looks_are_counted_and_ordered(registry: Registry) -> None:
    assert registry.sealed_looks() == 0

    first = _spend(registry, _spec("alpha_v1", 20), "campaign 08")
    assert registry.sealed_looks() == 1
    assert registry.sealed_look(first) == 1

    second = _spend(registry, _spec("beta_v1", 40), "campaign 09")
    third = _spend(registry, _spec("gamma_v1", 60), "campaign 10")
    assert registry.sealed_looks() == 3
    assert registry.sealed_look(second) == 2
    assert registry.sealed_look(third) == 3


def test_an_unspent_candidate_has_no_look(registry: Registry) -> None:
    fingerprint = registry.upsert_strategy(_spec("alpha_v1", 20))
    assert registry.sealed_look(fingerprint) is None
    assert registry.sealed_looks() == 0


# --------------------------------------------------------------------------
# What the count costs


def test_the_bar_rises_with_the_number_of_looks() -> None:
    bars = [multiplicity_bar(n) for n in (1, 2, 5, 10, 20)]
    assert bars == sorted(bars)
    assert bars[0] == pytest.approx(1.96, abs=0.01)
    assert bars[2] == pytest.approx(2.58, abs=0.01)
    assert bars[4] == pytest.approx(3.02, abs=0.01)


def test_one_look_is_the_threshold_the_project_already_used() -> None:
    """The adjustment has to reduce to the unadjusted case, or the first sealed
    run would be judged by a different rule than the one it was run under."""
    assert multiplicity_bar(1) == pytest.approx(SIGNIFICANCE_T, abs=0.05)


def test_a_nonsense_look_count_does_not_lower_the_bar() -> None:
    assert multiplicity_bar(0) == multiplicity_bar(1)
    assert multiplicity_bar(-3) == multiplicity_bar(1)


def test_an_alpha_that_passed_alone_can_fail_at_the_seventh_look() -> None:
    """The whole point of counting. t = +2.22 clears the bar as the only
    candidate ever screened, and does not clear it as the seventh."""
    alone = _measurement(t_alpha=2.22, looks=1)
    seventh = _measurement(t_alpha=2.22, looks=7)

    assert alone.alpha_clears_bar
    assert not seventh.alpha_clears_bar
    assert seventh.significance_bar > alone.significance_bar


def test_clearing_the_bar_is_still_not_confirmation() -> None:
    """``can_confirm`` is False by construction and the multiplicity adjustment
    does not touch it. Clearing the bar buys the right not to be dismissed on
    that ground, and nothing else."""
    measurement = _measurement(t_alpha=5.0, looks=1)
    assert measurement.alpha_clears_bar
    assert not measurement.can_confirm


def test_a_measurement_without_a_residual_clears_nothing() -> None:
    """No regression means no t. Reporting that as a pass would let a window too
    short to measure anything look like one that measured something."""
    measurement = SealedMeasurement(
        strategy="probe",
        fingerprint="f" * 16,
        since=datetime(2024, 9, 1, tzinfo=UTC),
        first_session=None,
        last_session=None,
        observations=0,
        backtest_sessions=0,
        strategy_return=0.0,
        strategy_sharpe=0.0,
        max_drawdown=0.0,
        benchmark_return=0.0,
        benchmark_sharpe=0.0,
        trades=0,
        residual=None,
        note="too short",
        looks=4,
    )
    assert not measurement.alpha_clears_bar
    assert not measurement.refuted


def test_the_summary_says_which_look_it_was() -> None:
    text = _measurement(t_alpha=2.22, looks=7).summary()
    assert "look 7" in text
    assert "2.69" in text  # the adjusted bar
    assert "does not clear that bar" in text


def test_a_first_look_summary_does_not_mention_multiplicity() -> None:
    """Noise on the common path. The first reading of the window is the
    unadjusted case and saying so every time would train the reader to skip it."""
    text = _measurement(t_alpha=2.22, looks=1).summary()
    assert "look 1" not in text
    assert "does not clear that bar" not in text


def test_the_count_reaches_the_stored_record() -> None:
    payload = _measurement(t_alpha=2.22, looks=7).as_dict()
    assert payload["looks"] == 7
    assert payload["significance_bar"] == pytest.approx(2.69, abs=0.01)
    assert payload["alpha_clears_bar"] is False
    assert payload["can_confirm"] is False
