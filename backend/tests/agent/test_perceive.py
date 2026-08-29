"""05 D2 — perception over real bars. And the replay guarantee, D7.

Every bar here is one of the 125 daily SPY bars captured from Alpaca on
2026-08-26, replayed through `RecordedMarketData`. Nothing opens a socket.

Two families of test, and the second is the one that carries specs/05's biggest
claim.

**Perception is honest.** Anything the engines could not measure comes back
`None`, and `None` never renders as a plausible number. The whole shape of this
is a response to a real failure: the first live cycle labelled an implied
volatility *level* `iv_rank`, and the model reasoned faithfully to a conclusion
the input did not support.

**A journalled day replays to an identical order set.** That is what makes the
record evidence rather than narration (specs/06 D6). If it ever fails, something
impure has leaked into a step that is supposed to be pure — which is why the
replay test is worth more than its size suggests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent import Choice, Stage, build_candidates, run_cycle
from alphagate.agent.candidates import vertical_credit_spreads
from alphagate.agent.cycle import CycleRecord
from alphagate.agent.iv_store import IvHistoryStore
from alphagate.agent.perceive import (
    HISTORY_DAYS,
    OPTIONS_HISTORY_FLOOR,
    atm_implied_volatility,
    atr_percent,
    perceive,
)
from alphagate.agent.prompt import build_user_message
from alphagate.agent.proposer import Proposer, RecordedProposer
from alphagate.agent.trend import MIN_CONFIDENCE, read_trend
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.trend import TrendPhase
from alphagate.marketdata import RecordedMarketData
from alphagate.options import OptionContract, OptionQuote, Right
from alphagate.risk import DEFAULT_LIMITS, PortfolioSnapshot
from tests.agent.conftest import StubProposer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
SPY: Ticker = ticker("SPY")
AS_OF = datetime(2026, 8, 26, 18, 0, tzinfo=UTC)
EQUITY = Decimal(100_000)


@pytest.fixture
def data() -> RecordedMarketData:
    return RecordedMarketData(directory=FIXTURES)


def chain(data: RecordedMarketData) -> Mapping[OptionContract, OptionQuote]:
    return data.option_chain(
        SPY, expiry_from=date(2026, 9, 4), expiry_to=date(2026, 9, 11), right="put"
    )


class TestPerceptionIsReal:
    def test_it_reads_the_captured_history(self, data: RecordedMarketData) -> None:
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert len(result.bars) == 125

    def test_atr_comes_from_the_core_indicator(self, data: RecordedMarketData) -> None:
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert result.read.atr_pct is not None
        assert Decimal("0.1") < result.read.atr_pct < Decimal(5)

    def test_realised_volatility_is_computed_from_real_closes(
        self, data: RecordedMarketData
    ) -> None:
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert result.volatility.realised is not None
        assert 0.01 < result.volatility.realised < 1.0

    def test_iv_vs_hv_is_exactly_computable_today(self, data: RecordedMarketData) -> None:
        """The field that carries "is premium rich" while `iv_rank` cannot."""
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert result.read.iv_vs_hv is not None
        assert result.read.iv_vs_hv > 0

    def test_hv_rank_needs_no_options_entitlement(self, data: RecordedMarketData) -> None:
        """Stock bars are enough, which is why it is available and `iv_rank` is not."""
        result = perceive(data, SPY, as_of=AS_OF, chain=None)
        assert result.read.hv_rank is not None
        assert 0 <= result.read.hv_rank <= 100


class TestPerceptionIsHonest:
    def test_no_chain_means_no_implied_volatility(self, data: RecordedMarketData) -> None:
        result = perceive(data, SPY, as_of=AS_OF, chain=None)
        assert result.volatility.implied is None
        assert result.read.iv_vs_hv is None
        assert not result.is_tradeable

    def test_no_history_means_no_iv_rank(self, data: RecordedMarketData) -> None:
        """The regression. `None`, not a middling default — a default here is a
        claim about how rich premium is."""
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert result.read.iv_rank is None
        assert result.volatility.observations == 0

    def test_an_unmeasured_field_never_renders_as_a_number(
        self, data: RecordedMarketData
    ) -> None:
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        view = json.loads(build_user_message(result.read, ()))["market_read"]
        assert view["iv_rank_0_100"] == "unmeasured"
        assert view["iv_percentile_0_100"] == "unmeasured"

    def test_a_rank_is_refused_if_it_is_not_a_rank(self, data: RecordedMarketData) -> None:
        """`MarketRead` refuses a value outside 0-100 in a rank field, so a
        level cannot be smuggled into one again."""
        from alphagate.agent.model import MarketRead
        from alphagate.core.errors import InvariantViolation

        with pytest.raises(InvariantViolation, match="must be a rank"):
            MarketRead(
                underlying=SPY, as_of=AS_OF, spot=Decimal(765), iv_rank=Decimal("157.9")
            )

    def test_the_options_history_floor_is_encoded(self) -> None:
        """Alpaca serves no options data before February 2024. Asking for more
        returns a shorter series, not an error."""
        assert date(2024, 2, 5) == OPTIONS_HISTORY_FLOOR
        assert HISTORY_DAYS == 180

    def test_atm_iv_is_none_without_greeks(self) -> None:
        assert atm_implied_volatility({}, Decimal(765)) is None

    def test_atr_is_none_on_a_cold_start(self) -> None:
        assert atr_percent(()) is None


class TestTrend:
    """specs/05 D2's reuse dividend: a state machine's output, not raw OHLC."""

    def test_it_measures_a_trend_from_real_bars(self, data: RecordedMarketData) -> None:
        bars = data.daily_bars(SPY, start=date(2026, 2, 27), end=date(2026, 8, 26))
        result = read_trend(bars)
        assert result.is_measured
        assert result.state is not None
        assert result.state.state in set(TrendPhase)
        assert result.bars_consumed == 125

    def test_the_state_explains_itself(self, data: RecordedMarketData) -> None:
        """Reason codes, not just a direction — which is the whole argument for
        reusing the engine rather than asking a model to eyeball a chart."""
        bars = data.daily_bars(SPY, start=date(2026, 2, 27), end=date(2026, 8, 26))
        state = read_trend(bars).state
        assert state is not None
        assert len(state.reason_codes) >= 3

    def test_no_bars_is_unmeasured_not_neutral(self) -> None:
        result = read_trend(())
        assert not result.is_measured
        assert result.reason == "no bars"
        assert "unmeasured" in result.describe()

    def test_a_cold_engine_is_unmeasured(self, data: RecordedMarketData) -> None:
        """Five bars in, the engine *does* emit a state — at confidence 0.00.

        Worth knowing, because it is not what the name "warmup" suggests: the
        engine reports early and tells you how much of the evidence it could
        read, and the confidence floor here is what actually refuses it. A
        caller that trusted `state is not None` alone would put a trend built
        from nothing into a prompt.
        """
        bars = data.daily_bars(SPY, start=date(2026, 8, 20), end=date(2026, 8, 26))
        result = read_trend(bars)
        assert not result.is_measured
        assert "confidence" in result.reason

    def test_confidence_climbs_with_history(self, data: RecordedMarketData) -> None:
        """The floor is a real gate, not a formality: 5 bars fail it, 18 fail
        it, 40 pass. Measured at the fixture, not assumed."""
        reads = {
            len(bars): read_trend(bars).is_measured
            for bars in (
                data.daily_bars(SPY, start=date(2026, 8, 20), end=date(2026, 8, 26)),
                data.daily_bars(SPY, start=date(2026, 8, 1), end=date(2026, 8, 26)),
                data.daily_bars(SPY, start=date(2026, 5, 1), end=date(2026, 8, 26)),
            )
        }
        assert reads == {5: False, 18: False, 81: True}

    def test_a_thin_evidence_read_is_refused(self, data: RecordedMarketData) -> None:
        """"A state produced from a third of the evidence is not a weaker
        opinion, it is an opinion about something else." """
        bars = data.daily_bars(SPY, start=date(2026, 2, 27), end=date(2026, 8, 26))
        assert not read_trend(bars, minimum_confidence=1.01).is_measured
        assert MIN_CONFIDENCE == 0.5

    def test_a_fresh_engine_every_call(self, data: RecordedMarketData) -> None:
        """Reusing one would make today's trend depend on which bars yesterday's
        process happened to consume."""
        bars = data.daily_bars(SPY, start=date(2026, 2, 27), end=date(2026, 8, 26))
        assert read_trend(bars).state == read_trend(bars).state

    def test_the_prompt_projects_it_rather_than_dumping_a_repr(
        self, data: RecordedMarketData
    ) -> None:
        """v4. The `repr` carried the evidence hash and a mappingproxy into the
        prompt — hundreds of tokens, and an invitation to reason about a hash."""
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        view = json.loads(build_user_message(result.read, ()))["market_read"]
        assert set(view["trend"]) == {
            "state",
            "timeframe",
            "strength_0_100",
            "confidence_0_1",
            "reasons",
        }
        assert "input_snapshot_hash" not in json.dumps(view)
        assert "mappingproxy" not in json.dumps(view)


class TestIvHistoryStore:
    def test_it_starts_empty(self, tmp_path: Path) -> None:
        assert IvHistoryStore(directory=tmp_path).observations(SPY) == []

    def test_one_observation_per_day(self, tmp_path: Path) -> None:
        """"Sixteen cycles a day would make a 20-observation history a day and a
        half of one afternoon's weather." """
        store = IvHistoryStore(directory=tmp_path)
        assert store.record(SPY, date(2026, 8, 26), 0.12)
        assert not store.record(SPY, date(2026, 8, 26), 0.13)
        assert store.record(SPY, date(2026, 8, 27), 0.13)
        assert store.observations(SPY) == [0.12, 0.13]

    def test_it_refuses_a_non_positive_observation(self, tmp_path: Path) -> None:
        store = IvHistoryStore(directory=tmp_path)
        assert not store.record(SPY, date(2026, 8, 26), 0.0)
        assert not store.record(SPY, date(2026, 8, 26), float("nan"))

    def test_a_truncated_tail_is_skipped(self, tmp_path: Path) -> None:
        store = IvHistoryStore(directory=tmp_path)
        store.record(SPY, date(2026, 8, 26), 0.12)
        path = store.path_for(SPY)
        path.write_text(path.read_text(encoding="utf-8") + '{"day": "2026-08', encoding="utf-8")
        assert store.observations(SPY) == [0.12]

    def test_seeding_without_the_entitlement_adds_nothing_and_raises_nothing(
        self, tmp_path: Path, data: RecordedMarketData
    ) -> None:
        """403 from an unsigned OPRA agreement is a configuration fact, not a
        failure of this cycle."""
        from alphagate.options import OptionContract

        store = IvHistoryStore(directory=tmp_path)
        added = store.seed_from_option_bars(
            data,
            SPY,
            OptionContract(SPY, date(2026, 12, 18), Decimal(770), Right.CALL),
            {},
            start=date(2026, 2, 27),
            end=date(2026, 8, 26),
        )
        assert added == 0
        assert store.observations(SPY) == []

    def test_perception_uses_the_history_once_it_exists(
        self, tmp_path: Path, data: RecordedMarketData
    ) -> None:
        """The path that lights up the moment OPRA is signed."""
        store = IvHistoryStore(directory=tmp_path)
        for day in range(1, 25):
            store.record(SPY, date(2026, 7, day), 0.10 + 0.005 * day)
        result = perceive(
            data,
            SPY,
            as_of=AS_OF,
            chain=chain(data),
            history=store.observations(SPY),
        )
        assert result.read.iv_rank is not None
        assert 0 <= result.read.iv_rank <= 100
        assert result.volatility.observations == 24


class TestReplay:
    """specs/05 D7 and specs/06 D6 — test plan item 6."""

    def _cycle(self, data: RecordedMarketData, proposer: Proposer) -> CycleRecord:
        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        quotes = chain(data)
        structures = vertical_credit_spreads(
            quotes, right=Right.PUT, width=Decimal(5), as_of=AS_OF
        )
        candidates = build_candidates(
            structures, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=AS_OF
        )
        from alphagate.agent.model import Setup

        return run_cycle(
            read=result.read,
            setup=Setup(SPY, "replay", "bullish", "fixture"),
            candidates=candidates,
            portfolio=PortfolioSnapshot(
                equity=EQUITY, positions=(), drawdown_pct=Decimal(0), fills_today=0
            ),
            limits=DEFAULT_LIMITS,
            as_of=AS_OF,
            proposer=proposer,
            sequence=1,
        )

    def test_a_journalled_day_replays_to_the_same_outcome(
        self, data: RecordedMarketData
    ) -> None:
        """Identical proposal, identical verdict, identical stage.

        Note the stage this actually reaches on the captured chain: `VETOED`,
        on `net_delta_budget`. That is the Gate refusing a real trade on real
        data, and it makes the test stronger rather than weaker — replay has to
        reproduce the refusal too, not just the trades that went through.
        """
        choice = Choice(2, "third on the menu", 0.6)
        live = self._cycle(data, StubProposer(choice=choice))
        assert live.proposal is not None

        replay = self._cycle(data, RecordedProposer({live.cycle_id: choice}))
        assert replay.proposal is not None
        assert replay.proposal.structure == live.proposal.structure
        assert replay.proposal.quantity == live.proposal.quantity
        assert replay.proposal.risk == live.proposal.risk
        assert replay.stage is live.stage
        assert replay.verdict == live.verdict
        assert replay.veto_reasons == live.veto_reasons

    def test_the_menu_no_longer_ranks_the_gates_refusals_to_the_top(
        self, data: RecordedMarketData
    ) -> None:
        """The fix the live agent forced.

        Ranking by return on risk puts the strike closest to the money first,
        which is also the highest delta — so index 0 was systematically the one
        candidate the Gate would refuse, the model chose it every cycle, and was
        vetoed every cycle. Every candidate on the menu now fits the delta band
        at its own size.
        """
        quotes = chain(data)
        structures = vertical_credit_spreads(
            quotes, right=Right.PUT, width=Decimal(5), as_of=AS_OF
        )
        low, high = DEFAULT_LIMITS.scaled_delta_band(EQUITY)
        for candidate in build_candidates(
            structures, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=AS_OF
        ):
            greeks = candidate.risk.net_greeks
            assert greeks is not None
            assert low <= greeks.delta * candidate.quantity <= high

    def test_unfiltered_the_top_candidate_would_have_been_vetoed(
        self, data: RecordedMarketData
    ) -> None:
        """The bug, pinned. With the delta filter switched off by an absurd
        band, the highest-return candidate breaches the real one."""
        from dataclasses import replace

        quotes = chain(data)
        structures = vertical_credit_spreads(
            quotes, right=Right.PUT, width=Decimal(5), as_of=AS_OF
        )
        unfiltered = build_candidates(
            structures,
            limits=replace(DEFAULT_LIMITS, delta_band=(-1e9, 1e9)),
            equity=EQUITY,
            as_of=AS_OF,
        )
        _, high = DEFAULT_LIMITS.scaled_delta_band(EQUITY)
        top = unfiltered[0].risk.net_greeks
        assert top is not None
        assert abs(top.delta * unfiltered[0].quantity) > high

    def test_the_gate_remains_authoritative(self, data: RecordedMarketData) -> None:
        """The filter shortens the menu; it does not take the Gate's job. A
        hostile book still gets a veto, and the journal still shows it."""
        from alphagate.agent.model import Setup

        result = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        structures = vertical_credit_spreads(
            chain(data), right=Right.PUT, width=Decimal(5), as_of=AS_OF
        )
        candidates = build_candidates(
            structures, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=AS_OF
        )
        record = run_cycle(
            read=result.read,
            setup=Setup(SPY, "replay", "bullish", "fixture"),
            candidates=candidates,
            portfolio=PortfolioSnapshot(
                equity=EQUITY,
                positions=(),
                drawdown_pct=Decimal("0.20"),
                fills_today=0,
            ),
            limits=DEFAULT_LIMITS,
            as_of=AS_OF,
            proposer=StubProposer(choice=Choice(0, "top", 0.6)),
            sequence=1,
        )
        assert record.stage is Stage.VETOED
        assert "drawdown_killswitch" in record.veto_reasons

    def test_the_menu_the_replay_saw_is_the_menu_the_model_saw(
        self, data: RecordedMarketData
    ) -> None:
        """Indices are the model's entire vocabulary. If the menu re-orders
        between the run and the replay, "index 2" means a different trade."""
        first = self._cycle(data, StubProposer(choice=Choice(None, "no", 0.0)))
        second = self._cycle(data, StubProposer(choice=Choice(None, "no", 0.0)))
        assert [c.summarise() for c in first.candidates] == [
            c.summarise() for c in second.candidates
        ]

    def test_perception_is_deterministic_over_the_same_payloads(
        self, data: RecordedMarketData
    ) -> None:
        a = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        b = perceive(data, SPY, as_of=AS_OF, chain=chain(data))
        assert a.read == b.read
        assert a.volatility == b.volatility

    def test_an_unrecorded_cycle_shortens_the_order_set_visibly(
        self, data: RecordedMarketData
    ) -> None:
        """"Not a different one quietly, and not a crash halfway through." """
        replay = self._cycle(data, RecordedProposer({}))
        assert replay.stage is Stage.DECLINED
        assert replay.proposal is None
