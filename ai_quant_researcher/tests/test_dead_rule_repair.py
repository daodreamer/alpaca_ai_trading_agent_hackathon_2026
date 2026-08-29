"""Recovering the iterations spent on rules that can never fire.

Measured, not guessed. A 40-hypothesis DeepSeek campaign over 19 real symbols
produced 33 experiments in its first four minutes, of which ten were rejected
with "never fires": the entry condition was syntactically perfect, referenced
only real features, and was unsatisfiable on daily bars. Thirty percent of the
research budget bought nothing.

Two defences, tested here.

**Tell the model what a bar contains.** Most of the dead rules asked daily bars
about intraday structure -- an opening drive, a midday VWAP reversion, a funding
rate. The feature catalogue lists what parses; it never said what is
*observable*, and a model with no other information will reach for the concepts
it knows.

**Give a dead rule one repair turn.** A rule that compiles and never fires is
exactly the recoverable failure the parse-error repair path already handles, and
it arrives with a reason worth sending back. One turn, not five: a model that
cannot fix an unsatisfiable condition after being shown it is not going to
produce a good hypothesis on the third try either.
"""

from __future__ import annotations

from typing import Any

from aqr.agent.prompts import build_user_prompt
from aqr.agent.proposer import Proposal
from aqr.agent.research import ResearchConfig, ResearchLoop
from aqr.data.bars import Bars
from aqr.registry.db import Registry

# Fires on nothing: RSI is bounded to [0, 100].
DEAD = {
    "name": "impossible_v1",
    "hypothesis": "An unsatisfiable rule.",
    "direction": "long",
    "regime": "",
    "entry": "rsi(14) > 150",
    "signal_exit": "",
    "stop_loss_atr_multiple": 2.0,
    "take_profit_r_multiple": 2.0,
    "max_holding_bars": 20,
    "expected_trades_per_year": 40,
}

ALIVE = {
    **DEAD,
    "name": "revived_v1",
    "entry": "rsi(14) < 35 and close < ema(20)",
}


class _StubProposer:
    """Proposes a dead rule, and repairs it if asked."""

    def __init__(self, *, can_repair: bool = True) -> None:
        self.can_repair = can_repair
        self.proposals = 0
        self.repairs: list[list[str]] = []

    def propose(self, **_: Any) -> Proposal:
        self.proposals += 1
        return Proposal(fields=dict(DEAD), source="stub", model="stub-1")

    def repair(self, *, proposal: Proposal, problems: list[str]) -> Proposal:
        if not self.can_repair:
            raise NotImplementedError
        self.repairs.append(list(problems))
        return Proposal(fields=dict(ALIVE), source="stub", model="stub-1")


class _StubberbornProposer(_StubProposer):
    """Asked to repair, returns the same dead rule."""

    def repair(self, *, proposal: Proposal, problems: list[str]) -> Proposal:
        self.repairs.append(list(problems))
        return Proposal(fields=dict(DEAD), source="stub", model="stub-1")


def _loop(
    proposer: Any, universe: dict[str, Bars], registry: Registry, iterations: int = 1
) -> ResearchLoop:
    return ResearchLoop(
        data=universe,
        registry=registry,
        config=ResearchConfig(
            symbols=sorted(universe),
            iterations=iterations,
            save_accepted_to=None,
            mutate_best_every=0,
        ),
        proposer=proposer,
    )


class TestPromptTellsTheModelWhatABarContains:
    def test_a_daily_prompt_names_what_daily_bars_cannot_see(self) -> None:
        prompt = build_user_prompt(symbols=["AAPL"], timeframe="1D", memory=[])
        lowered = prompt.lower()
        assert "intraday" in lowered
        assert "open, high, low, close" in lowered or "ohlcv" in lowered

    def test_an_intraday_prompt_does_not_claim_daily_limits(self) -> None:
        prompt = build_user_prompt(symbols=["AAPL"], timeframe="5m", memory=[])
        assert "one bar per session" not in prompt.lower()

    def test_the_guidance_changes_the_prompt_hash_across_timeframes(self) -> None:
        # Two runs on different timeframes must not share a prompt fingerprint,
        # or a recorded experiment cannot be reproduced from it.
        daily = build_user_prompt(symbols=["AAPL"], timeframe="1D", memory=[])
        intraday = build_user_prompt(symbols=["AAPL"], timeframe="5m", memory=[])
        assert daily != intraday


class TestRepairingADeadRule:
    def test_an_unsatisfiable_rule_is_sent_back_once_with_the_reason(
        self, universe: dict[str, Bars], tmp_path: Any
    ) -> None:
        proposer = _StubProposer()
        with Registry(tmp_path / "r.sqlite") as registry:
            _loop(proposer, universe, registry).run()

        assert len(proposer.repairs) == 1
        assert any("never fires" in p for p in proposer.repairs[0])

    def test_a_repaired_rule_is_evaluated_normally(
        self, universe: dict[str, Bars], tmp_path: Any
    ) -> None:
        proposer = _StubProposer()
        with Registry(tmp_path / "r.sqlite") as registry:
            steps = _loop(proposer, universe, registry).run()

        step = steps[0]
        assert step.spec is not None
        assert step.spec.name == "revived_v1", "the repaired rule should be the one evaluated"
        assert step.outcome is not None

    def test_repair_is_attempted_once_and_not_again(
        self, universe: dict[str, Bars], tmp_path: Any
    ) -> None:
        """One turn, not five. A model that cannot fix an unsatisfiable
        condition after being shown it will not fix it on the third try."""
        proposer = _StubberbornProposer()
        with Registry(tmp_path / "r.sqlite") as registry:
            steps = _loop(proposer, universe, registry).run()

        assert len(proposer.repairs) == 1
        assert steps[0].outcome is not None
        assert steps[0].outcome.verdict == "REJECT"

    def test_a_proposer_that_cannot_repair_still_works(
        self, universe: dict[str, Bars], tmp_path: Any
    ) -> None:
        # The offline heuristic proposer has no model to ask. Its dead rules are
        # rejected exactly as before rather than crashing the run.
        proposer = _StubProposer(can_repair=False)
        with Registry(tmp_path / "r.sqlite") as registry:
            steps = _loop(proposer, universe, registry).run()

        assert steps[0].outcome is not None
        assert steps[0].outcome.verdict == "REJECT"

    def test_the_failed_first_attempt_still_counts_against_the_search_budget(
        self, universe: dict[str, Bars], tmp_path: Any
    ) -> None:
        """A repaired proposal is the second thing tried, not the first. Hiding
        the attempt would understate the multiple-comparisons denominator, which
        is the one number the overfitting detector cannot do without."""
        proposer = _StubProposer()
        with Registry(tmp_path / "r.sqlite") as registry:
            _loop(proposer, universe, registry).run()
            names = [e["strategy_name"] for e in registry.experiments(limit=10)]

        assert "impossible_v1" in names, "the dead attempt must be on the record"
        assert "revived_v1" in names


class TestSavedStrategiesDoNotCollide:
    """The model reuses names, and two different rules are two different rules.

    Seen in a live campaign: `breadth_extreme_reversal_long_v1` appeared twice
    with different fingerprints and different verdicts (REVIEW 47.4, then PAPER
    60.1). The registry handled it correctly -- it is keyed by content -- but
    both were written to `breadth_extreme_reversal_long_v1.yaml`, so the file on
    disk was whichever finished last and the other was gone.
    """

    def test_two_rules_sharing_a_name_get_different_filenames(self) -> None:
        from aqr.agent.research import saved_filename
        from aqr.dsl.loader import loads

        one = loads(_NAMESAKE.format(threshold=30))
        two = loads(_NAMESAKE.format(threshold=45))

        assert one.name == two.name
        assert one.fingerprint() != two.fingerprint()
        assert saved_filename(one) != saved_filename(two)

    def test_the_filename_carries_the_name_and_the_fingerprint(self) -> None:
        """The name so a human can find it, the fingerprint so it can be matched
        back to its experiment record without parsing the file."""
        from aqr.agent.research import saved_filename
        from aqr.dsl.loader import loads

        spec = loads(_NAMESAKE.format(threshold=30))
        name = saved_filename(spec)
        assert spec.name in name
        assert spec.fingerprint() in name
        assert name.endswith(".yaml")

    def test_the_same_rule_always_lands_on_the_same_file(self) -> None:
        # Re-evaluating a rule must overwrite its own file, not accumulate copies.
        from aqr.agent.research import saved_filename
        from aqr.dsl.loader import loads

        spec = loads(_NAMESAKE.format(threshold=30))
        again = loads(_NAMESAKE.format(threshold=30))
        assert saved_filename(spec) == saved_filename(again)

    def test_the_research_loop_uses_it(self) -> None:
        # Discovered by duck-typing nothing: this is a direct call, so a rename
        # would break the import rather than silently restore the collision.
        import inspect

        from aqr.agent import research

        assert "saved_filename(spec)" in inspect.getsource(research.ResearchLoop._iterate)


_NAMESAKE = """
strategy:
  name: same_name_v1
  universe: {{symbols: [SPY], timeframe: 1D}}
  entry: rsi(14) < {threshold}
  exit: {{stop_loss: {{type: atr, multiplier: 2.0, period: 14}}, max_holding_bars: 20}}
  sizing: {{risk_per_trade: 0.005}}
"""
