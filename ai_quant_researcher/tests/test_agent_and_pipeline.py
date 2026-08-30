"""The agent layer and the pipeline it feeds, end to end.

The containment properties matter most here. A proposer is, by construction,
untrusted input: these tests assert that a hostile or merely incompetent
proposal costs one iteration and is written down, rather than crashing the run
or — far worse — quietly executing.

No test in this file calls an LLM. ``AnthropicProposer`` is exercised only for
the parts that do not need the network; the loop runs on the offline proposer,
which is the same code path a real run takes after the proposal arrives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aqr.agent.prompts import PROPOSAL_SCHEMA, SYSTEM_PROMPT, build_user_prompt, prompt_hash
from aqr.agent.proposer import (
    HeuristicProposer,
    Proposal,
    build_spec,
    check_proposal,
    spec_to_proposal_fields,
)
from aqr.agent.research import ResearchConfig, ResearchLoop
from aqr.data.providers import SyntheticProvider
from aqr.features.registry import REGISTRY
from aqr.pipeline import evaluate_candidate
from aqr.registry.db import Registry
from aqr.validation.splits import window_bars


class _ScriptedProposer:
    """Returns whatever it was handed. Stands in for a model with an agenda."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def propose(self, **kwargs: Any) -> Proposal:
        self.calls.append(kwargs)
        payload = self._payloads[min(len(self.calls) - 1, len(self._payloads) - 1)]
        if isinstance(payload, Exception):  # pragma: no cover - defensive
            raise payload
        return Proposal(fields=dict(payload), source="scripted")


GOOD = {
    "name": "scripted_pullback_v1",
    "hypothesis": "Pullbacks in an uptrend resolve upward.",
    "direction": "long",
    "regime": "close > ema(200)",
    "entry": "close <= ema(20) * 1.01 and rsi(14) > 40",
    "signal_exit": "",
    "stop_loss_atr_multiple": 2.0,
    "take_profit_r_multiple": 2.0,
    "max_holding_bars": 20,
    "expected_trades_per_year": 20,
}


class TestPrompts:
    def test_the_catalogue_lists_every_registered_feature(self) -> None:
        catalogue = build_user_prompt(symbols=["SPY"], timeframe="1D", memory=[])
        for name in REGISTRY:
            assert name in catalogue, f"{name} is callable but not offered to the model"

    def test_the_schema_has_no_free_text_code_field(self) -> None:
        """Containment starts here: there is nowhere to put code."""
        assert PROPOSAL_SCHEMA["additionalProperties"] is False
        assert not {"code", "python", "script", "expression_code"} & set(
            PROPOSAL_SCHEMA["properties"]
        )

    def test_memory_renders_failures_not_just_successes(self) -> None:
        rendered = build_user_prompt(
            symbols=["SPY"],
            timeframe="1D",
            memory=[
                {"name": "a", "verdict": "REJECT", "hypothesis": "h", "error": "never fires"}
            ],
        )
        assert "REJECT" in rendered and "never fires" in rendered

    def test_prompt_hash_changes_with_the_prompt(self) -> None:
        a = prompt_hash(SYSTEM_PROMPT, "one")
        b = prompt_hash(SYSTEM_PROMPT, "two")
        assert a != b and len(a) == 16


class TestBuildSpec:
    def test_a_valid_proposal_compiles(self) -> None:
        spec = build_spec(Proposal(GOOD, "test"), ["SPY", "QQQ"])
        assert spec.name == "scripted_pullback_v1"
        assert spec.universe.symbols == ("SPY", "QQQ")

    def test_the_proposal_cannot_choose_its_own_universe(self) -> None:
        """A model that picks its own symbols picks the ones its idea works on."""
        greedy = dict(GOOD, universe={"symbols": ["NVDA"]}, symbols=["NVDA"])
        spec = build_spec(Proposal(greedy, "test"), ["SPY"])
        assert spec.universe.symbols == ("SPY",)

    def test_the_proposal_cannot_choose_its_own_risk_budget(self) -> None:
        greedy = dict(GOOD, risk_per_trade=0.5, sizing={"risk_per_trade": 0.5})
        spec = build_spec(Proposal(greedy, "test"), ["SPY"], risk_per_trade=0.005)
        assert spec.sizing.risk_per_trade == 0.005

    @pytest.mark.parametrize(
        "entry",
        [
            "__import__('os')",
            "ema(20)",  # not a condition
            "unknown_feature(3) > 1",
            "",
        ],
    )
    def test_a_bad_entry_expression_is_refused(self, entry: str) -> None:
        with pytest.raises((ValueError, KeyError)):
            build_spec(Proposal(dict(GOOD, entry=entry), "test"), ["SPY"])

    def test_zero_take_profit_means_no_target(self) -> None:
        spec = build_spec(Proposal(dict(GOOD, take_profit_r_multiple=0), "test"), ["SPY"])
        assert spec.exit.take_profit.type == "none"

    def test_round_trips_back_into_proposal_fields(self) -> None:
        spec = build_spec(Proposal(GOOD, "test"), ["SPY"])
        fields = spec_to_proposal_fields(spec)
        assert fields["entry"] == GOOD["entry"]
        assert fields["stop_loss_atr_multiple"] == GOOD["stop_loss_atr_multiple"]


class TestHeuristicProposer:
    def test_walks_the_template_library_without_repeating(self) -> None:
        proposer = HeuristicProposer()
        names = {
            proposer.propose(symbols=["SPY"], timeframe="1D", memory=[]).fields["name"]
            for _ in range(8)
        }
        assert len(names) == 8, "the offline proposer repeated a template"

    def test_every_template_compiles(self) -> None:
        proposer = HeuristicProposer()
        for _ in range(8):
            proposal = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
            build_spec(proposal, ["SPY"])  # must not raise

    def test_mutation_changes_exactly_one_parameter(self) -> None:
        proposer = HeuristicProposer()
        parent = spec_to_proposal_fields(
            build_spec(proposer.propose(symbols=["SPY"], timeframe="1D", memory=[]), ["SPY"])
        )
        child = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[], parent=parent).fields
        knobs = ["stop_loss_atr_multiple", "take_profit_r_multiple", "max_holding_bars"]
        differing = [k for k in knobs if parent.get(k) != child.get(k)]
        assert len(differing) == 1, f"mutation changed {differing}; attribution is lost"

    def test_is_deterministic(self) -> None:
        first = HeuristicProposer().propose(symbols=["SPY"], timeframe="1D", memory=[])
        second = HeuristicProposer().propose(symbols=["SPY"], timeframe="1D", memory=[])
        assert first.fields == second.fields


class TestPipeline:
    def test_a_strategy_that_never_fires_is_rejected_before_any_backtest(
        self, universe, tmp_path
    ) -> None:
        spec = build_spec(Proposal(dict(GOOD, entry="rsi(14) > 101"), "t"), ["SPY"])
        with Registry(tmp_path / "r.sqlite") as reg:
            outcome = evaluate_candidate(spec, universe, registry=reg)
        assert outcome.verdict == "REJECT"
        assert outcome.backtests_run == 0, "compute was spent on an impossible strategy"

    def test_a_rejection_is_still_recorded(self, universe, tmp_path) -> None:
        spec = build_spec(Proposal(dict(GOOD, entry="rsi(14) > 101"), "t"), ["SPY"])
        with Registry(tmp_path / "r.sqlite") as reg:
            evaluate_candidate(spec, universe, registry=reg)
            assert reg.memory()[0]["verdict"] == "REJECT"

    def test_the_full_gauntlet_populates_every_report(self, universe, regimes, tmp_path) -> None:
        spec = build_spec(Proposal(GOOD, "t"), ["SPY", "QQQ", "IWM"])
        with Registry(tmp_path / "r.sqlite") as reg:
            outcome = evaluate_candidate(spec, universe, regime_labels=regimes, registry=reg)
        assert outcome.in_sample is not None
        assert outcome.walk_forward is not None and outcome.walk_forward.folds
        assert outcome.parameters is not None
        assert outcome.assets is not None
        assert outcome.regimes is not None
        assert outcome.monte_carlo is not None
        assert outcome.overfitting is not None
        assert outcome.evaluation is not None
        assert outcome.verdict in ("ACCEPT", "PAPER", "REVIEW", "REJECT")

    def test_the_search_cost_is_carried_into_the_next_evaluation(
        self, universe, tmp_path
    ) -> None:
        """The second identical run must face a higher multiple-comparisons bar."""
        spec = build_spec(Proposal(GOOD, "t"), ["SPY"])
        with Registry(tmp_path / "r.sqlite") as reg:
            first = evaluate_candidate(spec, universe, registry=reg)
            second = evaluate_candidate(spec, universe, registry=reg)
        assert first.overfitting is not None and second.overfitting is not None
        first_cost = next(s for s in first.overfitting.signals if s.name == "search_cost")
        second_cost = next(s for s in second.overfitting.signals if s.name == "search_cost")
        assert second_cost.penalty > first_cost.penalty

    def test_evaluation_is_deterministic(self, universe, tmp_path) -> None:
        spec = build_spec(Proposal(GOOD, "t"), ["SPY"])
        a = evaluate_candidate(spec, universe)
        b = evaluate_candidate(spec, universe)
        assert a.score == b.score
        assert a.verdict == b.verdict


class TestResearchLoop:
    def test_a_malformed_proposal_costs_one_iteration_not_the_run(
        self, universe, tmp_path
    ) -> None:
        proposer = _ScriptedProposer(dict(GOOD, entry="__import__('os').system('x')"), GOOD)
        with Registry(tmp_path / "r.sqlite") as reg:
            loop = ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(symbols=["SPY"], iterations=2, save_accepted_to=None),
                proposer=proposer,  # type: ignore[arg-type]
            )
            steps = loop.run()
        assert steps[0].verdict == "ERROR"
        assert "did not compile" in (steps[0].error or "")
        assert steps[1].verdict != "ERROR", "the run did not recover from a bad proposal"

    def test_a_proposer_that_raises_is_survived(self, universe, tmp_path) -> None:
        class Exploding:
            def propose(self, **_: Any) -> Proposal:
                raise RuntimeError("the model is down")

        with Registry(tmp_path / "r.sqlite") as reg:
            loop = ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(symbols=["SPY"], iterations=2, save_accepted_to=None),
                proposer=Exploding(),  # type: ignore[arg-type]
            )
            steps = loop.run()
        assert all(s.verdict == "ERROR" for s in steps)
        assert all("the model is down" in (s.error or "") for s in steps)

    def test_a_repeated_proposal_does_not_spend_a_backtest(self, universe, tmp_path) -> None:
        proposer = _ScriptedProposer(GOOD, GOOD)
        with Registry(tmp_path / "r.sqlite") as reg:
            loop = ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(symbols=["SPY"], iterations=2, save_accepted_to=None),
                proposer=proposer,  # type: ignore[arg-type]
            )
            steps = loop.run()
        assert steps[1].error is not None
        assert "already evaluated" in steps[1].error

    def test_memory_is_passed_to_the_proposer(self, universe, tmp_path) -> None:
        proposer = _ScriptedProposer(GOOD, dict(GOOD, name="second_v1"))
        with Registry(tmp_path / "r.sqlite") as reg:
            ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(symbols=["SPY"], iterations=2, save_accepted_to=None),
                proposer=proposer,  # type: ignore[arg-type]
            ).run()
        assert proposer.calls[0]["memory"] == []
        assert proposer.calls[1]["memory"], "the second proposal was made with no memory"

    def test_the_offline_loop_completes_and_ranks(self, universe, regimes, tmp_path) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            loop = ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(
                    symbols=["SPY", "QQQ", "IWM"], iterations=4, save_accepted_to=None
                ),
                regime_labels=regimes,
            )
            steps = loop.run()
            assert len(steps) == 4
            assert reg.total_backtests() > 0
        assert loop.summary()

    def test_nothing_is_promoted_past_paper(self, universe, tmp_path) -> None:
        """Live promotion is a human decision the loop cannot make."""
        with Registry(tmp_path / "r.sqlite") as reg:
            ResearchLoop(
                data=universe,
                registry=reg,
                config=ResearchConfig(symbols=["SPY"], iterations=3, save_accepted_to=None),
            ).run()
            assert all(s.status != "LIVE" for s in reg.strategies())


class TestTimeframes:
    """The proposer may choose a granularity, inside the run's allowed set."""

    def test_a_proposal_may_choose_a_timeframe_inside_the_allowed_set(self) -> None:
        proposal = Proposal(dict(GOOD, timeframe="1h"), "test")
        spec = build_spec(proposal, ["SPY"], allowed_timeframes=("1D", "1h"))
        assert spec.universe.timeframe == "1h"

    def test_a_timeframe_outside_the_allowed_set_is_refused(self) -> None:
        proposal = Proposal(dict(GOOD, timeframe="4h"), "test")
        with pytest.raises(ValueError, match="timeframe"):
            build_spec(proposal, ["SPY"], allowed_timeframes=("1D", "1h"))

    def test_a_proposal_without_a_timeframe_gets_the_run_default(self) -> None:
        spec = build_spec(Proposal(dict(GOOD), "test"), ["SPY"], timeframe="1h")
        assert spec.universe.timeframe == "1h"

    def test_check_proposal_names_an_unknown_timeframe(self) -> None:
        problems = check_proposal(dict(GOOD, timeframe="5m"))
        assert any("timeframe" in p and "5m" in p for p in problems)

    def test_a_multi_timeframe_prompt_covers_each_granularity_and_the_choice(self) -> None:
        prompt = build_user_prompt(
            symbols=["SPY"], timeframe="1D", memory=[], timeframes=("1D", "1h", "4h")
        )
        assert "On 1h bars" in prompt and "On 4h bars" in prompt
        assert "six bars per regular session" in prompt
        assert "two bars per regular session" in prompt
        assert 'write it into the "timeframe" field' in prompt

    def test_a_single_timeframe_prompt_fixes_the_field(self) -> None:
        prompt = build_user_prompt(symbols=["SPY"], timeframe="1D", memory=[])
        assert 'fixed to "1D"' in prompt

    def test_memory_renders_each_experiments_timeframe(self) -> None:
        rendered = build_user_prompt(
            symbols=["SPY"],
            timeframe="1D",
            memory=[{"name": "a", "verdict": "REJECT", "timeframe": "1h"}],
        )
        assert "1h bars" in rendered

    def test_the_heuristic_proposer_stamps_the_run_default(self) -> None:
        fields = HeuristicProposer().propose(symbols=["SPY"], timeframe="1h", memory=[]).fields
        assert fields["timeframe"] == "1h"

    def test_proposal_fields_round_trip_the_timeframe(self) -> None:
        proposal = Proposal(dict(GOOD, timeframe="1h"), "test")
        spec = build_spec(proposal, ["SPY"], allowed_timeframes=("1D", "1h"))
        assert spec_to_proposal_fields(spec)["timeframe"] == "1h"

    def test_a_timeframe_mismatch_is_rejected_before_any_backtest(self, universe) -> None:
        """An hourly spec on daily bars is a data error, not a bad strategy."""
        proposal = Proposal(dict(GOOD, timeframe="1h"), "t")
        spec = build_spec(proposal, ["SPY"], allowed_timeframes=("1D", "1h"))
        outcome = evaluate_candidate(spec, universe)
        assert outcome.verdict == "REJECT"
        assert "1h" in (outcome.rejected_early or "")
        assert outcome.backtests_run == 0

    def test_the_loop_evaluates_each_candidate_on_its_own_bar_size(self, tmp_path) -> None:
        provider = SyntheticProvider()
        start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC)
        daily = {"SPY": provider.load("SPY", start, end)}
        hourly = {"SPY": provider.load("SPY", start, end, "1h")}
        proposer = _ScriptedProposer(dict(GOOD, timeframe="1h"))
        with Registry(tmp_path / "r.sqlite") as reg:
            loop = ResearchLoop(
                data=daily,
                registry=reg,
                config=ResearchConfig(
                    symbols=["SPY"],
                    iterations=1,
                    timeframes=("1D", "1h"),
                    save_accepted_to=None,
                    mutate_best_every=0,
                ),
                proposer=proposer,  # type: ignore[arg-type]
                data_by_timeframe={"1D": daily, "1h": hourly},
            )
            steps = loop.run()
            memory_timeframe = reg.memory()[0]["timeframe"]
        step = steps[0]
        assert step.error is None
        assert step.spec is not None and step.spec.universe.timeframe == "1h"
        assert memory_timeframe == "1h"
        assert step.outcome is not None and step.outcome.walk_forward is not None
        train, _ = window_bars("1h")
        assert len(step.outcome.walk_forward.folds[0].fold.train) == train
