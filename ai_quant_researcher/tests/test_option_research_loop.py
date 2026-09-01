"""The option research loop and its budget — specs/10-options-research.md D8.

Two claims are under test and both are about *denominators*.

**The search budget is a gate, not a default.** 71 non-overlapping 28-DTE cycles
in 5.55 years is the whole sample. The equity campaign spent 414 hypotheses; 400
trials against 71 cycles produces a winner indistinguishable from the luckiest
draw, so the cap is enforced against the registry's own count rather than
against one run's ``iterations``, which a second invocation would reset.

**The two searches are counted separately.** The multiplicity bar is Bonferroni
on a family-wise 5% rate, so a denominator mixing 414 equity hypotheses with 20
option ones makes both bars wrong: far too strict for the option side, far too
loose for the equity side. Nothing here sums them, and the test says so
directly.

The market is hand-built and small. What is being tested is wiring — that a
budget is read from the right place, that a family stamp survives a round trip,
that a failure is still recorded — none of which needs the real cache, and all
of which would take minutes against it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from aqr.agent import option_research
from aqr.agent.option_proposer import TemplateOptionProposer, build_option_spec
from aqr.agent.option_research import (
    OPTION_SEARCH_BUDGET,
    OptionResearchConfig,
    OptionResearchLoop,
)
from aqr.agent.proposer import Proposal
from aqr.dsl.schema import StrategySpec, Universe
from aqr.options.chain import ChainIndex
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket
from aqr.options.spec import Cadence, OptionSpec, StructureSpec
from aqr.registry.db import EQUITY, OPTION, ExperimentRecord, Registry
from tests.test_option_engine import chain_row, make_underlying, trading_days

# --------------------------------------------------------------------------- #
# A small world that spans enough calendar time for folds to exist
# --------------------------------------------------------------------------- #


def _market(first: date = date(2019, 2, 4), last: date = date(2024, 6, 28)) -> OptionMarket:
    """A session every fortnight, one 28-day expiry, a put and a call ladder.

    Prices are ``build_world``'s, so the arithmetic of any one trade is already
    pinned elsewhere and what varies here is only the calendar span.
    """
    days = trading_days(1500, start=first - timedelta(days=30))
    sessions = [d for i, d in enumerate(days) if i >= 10 and i % 10 == 0 and d <= last]
    rows: list[dict[str, str]] = []
    for session in sessions:
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            chain_row(session, expiry, 370.0, "put", bid=0.20, ask=0.25, delta=-0.04),
            chain_row(session, expiry, 410.0, "call", bid=1.40, ask=1.50, delta=0.16),
            chain_row(session, expiry, 420.0, "call", bid=0.40, ask=0.45, delta=0.09),
            chain_row(session, expiry, 430.0, "call", bid=0.15, ask=0.20, delta=0.04),
        ]
    return OptionMarket(underlying=make_underlying({}, days), chain=ChainIndex.from_rows(rows))


@pytest.fixture(scope="module")
def market() -> OptionMarket:
    return _market()


@pytest.fixture
def registry(tmp_path: Path) -> Any:
    with Registry(tmp_path / "research.sqlite") as reg:
        yield reg


def _equity_spec(i: int) -> StrategySpec:
    # The lookback varies, not just the name: a fingerprint is content-addressed
    # and drops the name on purpose, so twenty rules called different things and
    # trading identically are one hypothesis, which is the correct answer and an
    # unhelpful fixture.
    return StrategySpec(
        name=f"equity_rule_{i}",
        entry=f"close > ema({20 + i})",
        universe=Universe(symbols=("SPY",), timeframe="1D"),
        hypothesis="an equity rule, recorded to fill the other denominator",
    )


def _record_equity(reg: Registry, count: int) -> None:
    for i in range(count):
        spec = _equity_spec(i)
        reg.upsert_strategy(spec)
        reg.record_experiment(
            ExperimentRecord(
                fingerprint=spec.fingerprint(),
                strategy_name=spec.name,
                symbols=("SPY",),
                timeframe="1D",
                data_start="",
                data_end="",
                dataset_version="synthetic-v1",
                verdict="REJECT",
            )
        )


def _option_spec(i: int) -> OptionSpec:
    """A distinct rule per ``i``, and distinct from anything the template library
    proposes: the cadence starts at 100 sessions, which no template uses. A
    fixture that collided with the library would fill the budget and then have
    the loop skip every proposal as "already evaluated", which is correct
    behaviour and tests nothing about the budget."""
    return OptionSpec(
        name=f"prior_option_rule_{i}",
        underlying="SPY",
        entry="close > 0",
        structure=StructureSpec(type="put_credit_spread", width_delta=0.06),
        cadence=Cadence(min_sessions_between_entries=100 + i),
        hypothesis="a prior option hypothesis, already spent from the budget",
    )


def _record_option(reg: Registry, count: int) -> None:
    for i in range(count):
        spec = _option_spec(i)
        reg.upsert_option_strategy(spec)
        reg.record_experiment(
            ExperimentRecord(
                fingerprint=spec.fingerprint(),
                strategy_name=spec.name,
                symbols=("SPY",),
                timeframe="option_chain",
                data_start="",
                data_end="",
                dataset_version="options-cache",
                family=OPTION,
                verdict="REJECT",
            )
        )


# --------------------------------------------------------------------------- #
# The two denominators never merge
# --------------------------------------------------------------------------- #


def test_the_two_searches_are_counted_separately(registry: Registry) -> None:
    _record_equity(registry, 7)
    _record_option(registry, 3)

    assert registry.distinct_hypotheses(family=EQUITY) == 7
    assert registry.distinct_hypotheses(family=OPTION) == 3
    # The combined figure exists for display and is never what a bar is computed
    # against: 10 would over-penalise the option search by a factor of three and
    # under-penalise the equity one by a third.
    assert registry.distinct_hypotheses() == 10


def test_an_equity_hypothesis_never_raises_the_option_bar(registry: Registry) -> None:
    _record_option(registry, 2)
    before = registry.distinct_hypotheses(family=OPTION)
    _record_equity(registry, 50)
    assert registry.distinct_hypotheses(family=OPTION) == before


def test_experiments_and_memory_can_be_scoped_to_one_family(registry: Registry) -> None:
    _record_equity(registry, 4)
    _record_option(registry, 2)

    option_names = {row["name"] for row in registry.memory(50, family=OPTION)}
    equity_names = {row["name"] for row in registry.memory(50, family=EQUITY)}
    assert option_names and equity_names
    assert not option_names & equity_names
    assert len(registry.experiments(50, family=OPTION)) == 2
    assert len(registry.experiments(50, family=EQUITY)) == 4


def test_rows_written_before_the_split_read_back_as_equity(registry: Registry) -> None:
    """The 414 hypotheses already in the ledger predate the ``family`` column.
    They are equity experiments -- the option pipeline did not exist -- and
    resolving NULL any other way would either lose them from the equity
    denominator or add them to the option one."""
    _record_equity(registry, 3)
    registry._conn.execute("UPDATE experiments SET family = NULL")  # noqa: SLF001
    registry._conn.commit()  # noqa: SLF001

    assert registry.distinct_hypotheses(family=EQUITY) == 3
    assert registry.distinct_hypotheses(family=OPTION) == 0


def test_the_sealed_windows_are_counted_separately(registry: Registry) -> None:
    """Two different data sets: S&P 500 bars past the embargo, and SPY chains
    past it. A candidate screened against one has not consumed a look at the
    other, and charging it as though it had would raise a bar for a reading that
    never touched the data."""
    _record_equity(registry, 1)
    _record_option(registry, 1)
    for fingerprint in (
        registry.strategies(family=EQUITY)[0].fingerprint,
        registry.strategies(family=OPTION)[0].fingerprint,
    ):
        registry.preregister(fingerprint, selection_rule="the only one", seal_digest="d")
        registry.record_sealed_run(fingerprint, result={"measurement": {}})

    assert registry.sealed_looks(family=EQUITY) == 1
    assert registry.sealed_looks(family=OPTION) == 1
    assert registry.sealed_looks() == 2


def test_an_option_strategy_round_trips_through_the_registry(registry: Registry) -> None:
    """The two spec formats are not interchangeable, so the row has to say which
    one it holds -- reading an option rule with the equity loader produces a
    spec with no entry condition rather than an error."""
    from aqr.options.spec import loads_option_spec

    spec = build_option_spec(
        TemplateOptionProposer().propose(underlying="SPY", memory=[]), "SPY"
    )
    registry.upsert_option_strategy(spec)
    record = registry.get_strategy(spec.fingerprint())

    assert record is not None
    assert record.family == OPTION
    assert loads_option_spec(record.spec_yaml).fingerprint() == spec.fingerprint()


# --------------------------------------------------------------------------- #
# The budget is a gate
# --------------------------------------------------------------------------- #


def test_a_campaign_wider_than_the_ceiling_is_refused_outright() -> None:
    """The ceiling is a guardrail against a runaway loop -- a script with a typo
    in its iteration count -- not the statistical control. The control is the
    deflation term, which scales with the trial count whether or not anything is
    capped: on the first real campaign it took a 97/100 rule's Sharpe of 0.67
    down to -0.28 at twenty trials, and the cap was never involved."""
    with pytest.raises(ValueError, match="guardrail"):
        OptionResearchConfig(iterations=OPTION_SEARCH_BUDGET + 1)


def test_the_ceiling_is_wide_enough_for_the_campaigns_this_is_used_for() -> None:
    """Hundreds, not dozens. Kept as an assertion because the number is a
    decision -- specs/10 D8 argues for 20 and this project diverged from it
    deliberately -- and a decision nobody wrote down gets un-made by accident."""
    assert OPTION_SEARCH_BUDGET >= 1000


def test_the_budget_counts_the_registry_not_this_run(
    registry: Registry, market: OptionMarket, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ceiling enforced per run is not a ceiling: two invocations of ten would
    spend twenty and a third would spend thirty, and the multiple-comparisons
    problem does not reset when a process exits.

    The real ceiling is a thousand; patched down here because what is under test
    is where the count is read from, not how big it is."""
    monkeypatch.setattr(option_research, "OPTION_SEARCH_BUDGET", 6)
    _record_option(registry, 6)
    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=3, save_accepted_to=None),
    )
    steps = loop.run()

    assert len(steps) == 1
    assert steps[0].verdict == "ERROR"
    assert "budget is spent" in (steps[0].error or "")
    # Nothing was proposed, so nothing was recorded: a refused campaign must not
    # itself inflate the denominator it refused over.
    assert registry.distinct_hypotheses(family=OPTION) == 6


def test_a_campaign_stops_at_the_budget_mid_run(
    registry: Registry, market: OptionMarket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(option_research, "OPTION_SEARCH_BUDGET", 6)
    _record_option(registry, 5)
    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=4, save_accepted_to=None),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    )
    steps = loop.run()

    assert len(steps) == 2
    assert "budget is spent" in (steps[-1].error or "")


# --------------------------------------------------------------------------- #
# The loop itself
# --------------------------------------------------------------------------- #


def test_the_loop_records_every_attempt_including_the_failures(
    registry: Registry, market: OptionMarket
) -> None:
    """Forgetting the ones that went nowhere is the mechanism by which a search
    looks luckier than it was."""
    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=2, save_accepted_to=None),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    )
    steps = loop.run()

    assert len(steps) == 2
    assert registry.distinct_hypotheses(family=OPTION) == 2
    for row in registry.experiments(10, family=OPTION):
        assert row["family"] == OPTION
        # Not a bar size. The chain's grid is one snapshot per session and is not
        # the bar grid at all, so "1D" here would claim the walk-forward was cut
        # in daily bars.
        assert row["timeframe"] == "option_chain"


def test_a_proposer_that_raises_costs_one_iteration_not_the_run(
    registry: Registry, market: OptionMarket
) -> None:
    class Broken:
        def __init__(self) -> None:
            self.calls = 0

        def propose(self, **kwargs: Any) -> Proposal:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("the endpoint returned 502")
            return TemplateOptionProposer().propose(**kwargs)

    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=2, save_accepted_to=None),
        proposer=Broken(),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    )
    steps = loop.run()

    assert steps[0].verdict == "ERROR"
    assert "502" in (steps[0].error or "")
    assert steps[1].verdict != "ERROR"


def test_the_same_rule_is_not_evaluated_twice(
    registry: Registry, market: OptionMarket
) -> None:
    class Repetitive:
        def propose(self, **kwargs: Any) -> Proposal:
            return TemplateOptionProposer().propose(**kwargs)

    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=2, save_accepted_to=None),
        proposer=Repetitive(),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    )
    steps = loop.run()

    assert "already evaluated" in (steps[1].error or "")


def test_a_surviving_rule_is_written_beside_the_equity_ones_but_not_among_them(
    registry: Registry, market: OptionMarket, tmp_path: Path
) -> None:
    """``dsl/loader.py`` would read an option rule as a spec with no entry
    condition rather than refusing it, so one directory holding both would make
    ``aqr backtest strategies/*.yaml`` a loaded gun."""
    target = tmp_path / "strategies" / "options"
    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=1, save_accepted_to=str(target)),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    )
    step = loop.run()[0]

    if step.verdict in ("ACCEPT", "PAPER"):
        assert list(target.glob("*.yaml"))
    else:
        # The far more likely outcome on this window, and it is a result rather
        # than a failure -- specs/07 D0 says the premium's alpha has decayed.
        assert not target.exists() or not list(target.glob("*.yaml"))



# --------------------------------------------------------------------------- #
# The loop hands the proposer the ranges its own market actually holds
# --------------------------------------------------------------------------- #


def test_the_loop_measures_spans_from_its_own_market(
    registry: Registry, market: OptionMarket
) -> None:
    """A campaign without this lost seven of twenty slots to conditions no
    session could satisfy. The spans must come from the market the loop was
    handed -- measured, not documented -- because a re-pull moves them."""
    from aqr.features.engine import FeatureKey

    loop = OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=1, save_accepted_to=None),
    )
    span = loop.span(FeatureKey("close", ()))
    assert span is not None
    low, high = span
    assert low <= float(market.underlying.close.min())
    assert high >= float(market.underlying.close.max()) - 1e-9
    # A feature this fixture's market cannot answer is None, never a guess: the
    # chain here carries no volatility history.
    assert loop.span(FeatureKey("iv_rank", ())) is None


def test_the_proposer_is_handed_the_span_resolver(
    registry: Registry, market: OptionMarket
) -> None:
    seen: dict[str, Any] = {}

    class Watching:
        def propose(self, **kwargs: Any) -> Proposal:
            seen.update(kwargs)
            return TemplateOptionProposer().propose(**kwargs)

    OptionResearchLoop(
        market=market,
        registry=registry,
        config=OptionResearchConfig(iterations=1, save_accepted_to=None),
        proposer=Watching(),
        backtest_config=OptionBacktestConfig(initial_equity=1_000_000.0),
    ).run()

    assert callable(seen["span"])
    assert seen["spans"] is not None
