"""05 D1 step 3, D4, D6, D7 — the menu. Test plan items 4, 7, 8.

The menu is where three of specs/05's guarantees stop being prose:

* a candidate the risk budget cannot fund is **never shown** (D4);
* a candidate priced off stale quotes is **dropped before the model sees it**,
  not after (D6);
* the same chain produces the same menu, in the same order, every time (D7).

The ordering property is the least obvious and the most load-bearing. Indices
are the model's entire vocabulary; if the menu re-orders between the run that
produced a journal line and the run that replays it, "index 3" means a different
trade and the replay claim is worthless.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from alphagate.agent import MENU_LIMIT, build_candidates, size_for
from alphagate.agent.candidates import (
    SHORTLIST_SPREAD_PCT,
    net_return_on_risk,
    spreads_by_delta,
    vertical_credit_spreads,
)
from alphagate.options import Greeks, OptionContract, OptionQuote, OptionStructure, Right, Side
from alphagate.risk import DEFAULT_LIMITS, RiskLimits
from tests.agent.conftest import EQUITY, NOW, contract, menu, put_chain, quote, risk_for

Built = list[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]]


def spreads(
    quotes: Mapping[OptionContract, OptionQuote] | None = None, *, width: str = "5"
) -> Built:
    return vertical_credit_spreads(
        quotes if quotes is not None else put_chain(),
        right=Right.PUT,
        width=Decimal(width),
        as_of=NOW,
    )


class TestBuildingTheMenu:
    def test_it_finds_every_five_wide_pair(self) -> None:
        """Six strikes, five 5-wide pairs."""
        assert len(spreads()) == 5

    def test_every_structure_is_a_genuine_credit(self) -> None:
        """A put credit spread sells the higher strike. Getting it backwards
        builds a debit and labels it a credit — which `OptionStructure` refuses,
        so the direction is asserted by construction."""
        for structure, _ in spreads():
            short = next(leg for leg in structure.legs if leg.side.value == "sell")
            long = next(leg for leg in structure.legs if leg.side.value == "buy")
            assert short.contract.strike > long.contract.strike

    def test_indices_are_dense_and_zero_based(self) -> None:
        candidates = menu()
        assert [c.index for c in candidates] == list(range(len(candidates)))

    def test_the_menu_is_bounded(self) -> None:
        """"A longer menu is not a better menu — it is a longer prompt with more
        ways to be wrong, and the tail of it is never chosen." """
        assert len(menu()) <= MENU_LIMIT

    def test_it_is_ranked_by_return_on_risk_net_of_the_spread(self) -> None:
        """Not raw return on risk. See `net_return_on_risk` — ranking on the raw
        figure put the widest market first, systematically."""
        candidates = menu()
        scores = [net_return_on_risk(c.risk) for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_the_crossing_cost_actually_moves_the_ranking(self) -> None:
        """A tight 26% beats a wide 30%: "a 30% return that costs 6% to enter is
        not a better trade than a 26% return that costs 1%"."""
        quotes = put_chain()
        wide = contract("752")
        quotes[wide] = quote(wide, "3.05", "3.40")  # same mid, much worse market
        ranked = build_candidates(
            spreads(quotes), limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW
        )
        top_strikes = {str(leg.contract.strike) for leg in ranked[0].structure.legs}
        assert "752" not in top_strikes, "the wide market must not rank first"

    def test_the_menu_shows_the_model_the_metric_it_was_ranked_by(self) -> None:
        summary = menu()[0].summarise()
        assert "return_on_risk_after_spread" in summary
        assert "return_on_risk" in summary

    def test_a_width_with_no_pairs_yields_nothing(self) -> None:
        assert spreads(width="7") == []


class TestSizingD4:
    """specs/05 test plan item 4."""

    def test_the_rule_is_floor_of_budget_over_max_loss(self, limits: RiskLimits) -> None:
        """1% of 100k is 1,000; a 440 spread fits twice, not 2.27 times."""
        risk = risk_for(("752", "747"))
        assert risk.max_loss == Decimal("440.000")
        assert size_for(risk, limits, EQUITY) == 2

    def test_exactly_at_the_budget(self, limits: RiskLimits) -> None:
        """440 into a 440 budget is one unit, not zero and not two."""
        equity = Decimal(44_000)  # 1% = 440
        assert size_for(risk_for(("752", "747")), limits, equity) == 1

    def test_just_under_the_budget_is_zero(self, limits: RiskLimits) -> None:
        equity = Decimal(43_900)  # 1% = 439, one cent short of a unit
        assert size_for(risk_for(("752", "747")), limits, equity) == 0

    def test_just_over_the_budget_is_still_one(self, limits: RiskLimits) -> None:
        equity = Decimal(87_900)  # 1% = 879, just short of two units
        assert size_for(risk_for(("752", "747")), limits, equity) == 1

    def test_it_never_rounds_up(self, limits: RiskLimits) -> None:
        """Rounding up would put the order over the limit and hand the Gate a
        veto that sizing could have avoided. A control routinely tripped by our
        own code stops being read."""
        risk = risk_for(("752", "747"))
        for equity in (Decimal(87_000), Decimal(87_999), Decimal(88_000)):
            quantity = size_for(risk, limits, equity)
            assert risk.max_loss * quantity <= limits.max_trade_loss(equity)

    def test_zero_quantity_candidates_never_reach_the_menu(self, limits: RiskLimits) -> None:
        """D4: showing a model something it cannot legally trade and relying on
        it not to pick that one is not a control."""
        tiny = build_candidates(spreads(), limits=limits, equity=Decimal(5_000), as_of=NOW)
        assert tiny == ()

    def test_every_candidate_on_the_menu_is_fundable(self, limits: RiskLimits) -> None:
        for candidate in menu():
            assert candidate.quantity >= 1
            assert candidate.risk.max_loss * candidate.quantity <= limits.max_trade_loss(EQUITY)


class TestStaleQuotesDropEarlyD6:
    """specs/05 test plan item 8."""

    def test_a_fresh_chain_produces_a_menu(self) -> None:
        assert len(menu(age=0)) > 0

    def test_a_stale_chain_produces_nothing(self) -> None:
        """Dropped here, before proposal — not shown and then rejected."""
        assert menu(age=90) == ()

    def test_the_boundary_is_the_gates_own_limit(self, limits: RiskLimits) -> None:
        assert len(menu(age=int(limits.max_quote_age))) > 0
        assert menu(age=int(limits.max_quote_age) + 1) == ()

    def test_one_stale_leg_drops_only_its_own_spreads(self) -> None:
        """`compute_risk` takes the oldest leg's age, so a single stale strike
        poisons exactly the two spreads that use it — and nothing else."""
        quotes = put_chain()
        stale = contract("747")
        quotes[stale] = quote(stale, "2.60", "2.65", age=300)
        candidates = build_candidates(
            spreads(quotes), limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW
        )
        strikes = {
            str(leg.contract.strike) for c in candidates for leg in c.structure.legs
        }
        assert "747" not in strikes
        assert len(candidates) == 3


class TestFilters:
    def test_a_wide_market_is_shortlisted_out(self) -> None:
        quotes = put_chain()
        wide = contract("752")
        quotes[wide] = quote(wide, "2.50", "4.00")
        candidates = build_candidates(
            spreads(quotes), limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW
        )
        assert all(c.risk.worst_spread_pct <= SHORTLIST_SPREAD_PCT for c in candidates)

    def test_the_shortlist_is_looser_than_the_gate(self) -> None:
        """Deliberate. If they were equal the Gate would never veto on
        liquidity, and a control that never fires is one nobody can tell works."""
        assert DEFAULT_LIMITS.max_spread_pct < SHORTLIST_SPREAD_PCT

    def test_an_expiry_outside_the_window_is_dropped(self, limits: RiskLimits) -> None:
        far = replace(limits, dte_range=(30, 45))
        assert build_candidates(spreads(), limits=far, equity=EQUITY, as_of=NOW) == ()

    def test_missing_greeks_do_not_block_the_menu(self) -> None:
        """The Gate refuses an open with unknown exposure (specs/03 D4); the
        menu does not pre-empt that. Filtering here would hide a veto the
        journal should show."""
        assert len(menu(greeks=None)) > 0


class TestDeterminismD7:
    """specs/05 test plan item 7."""

    def test_the_same_chain_gives_the_same_menu(self) -> None:
        assert menu() == menu()

    def test_indices_are_stable_across_runs(self) -> None:
        first = [(c.index, str(c.structure)) for c in menu()]
        second = [(c.index, str(c.structure)) for c in menu()]
        assert first == second

    def test_input_order_does_not_change_the_menu(self) -> None:
        """A dict iteration order leaking into the menu would re-index the
        model's whole vocabulary between runs."""
        forward = spreads()
        backward = list(reversed(forward))
        assert build_candidates(
            forward, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW
        ) == build_candidates(backward, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        equity=st.integers(min_value=20_000, max_value=5_000_000),
        age=st.integers(min_value=0, max_value=59),
    )
    def test_building_is_a_pure_function(self, equity: int, age: int) -> None:
        built = spreads(put_chain(age=age))
        a = build_candidates(built, limits=DEFAULT_LIMITS, equity=Decimal(equity), as_of=NOW)
        b = build_candidates(built, limits=DEFAULT_LIMITS, equity=Decimal(equity), as_of=NOW)
        assert a == b

    @settings(max_examples=100)
    @given(equity=st.integers(min_value=1_000, max_value=10_000_000))
    def test_sizing_never_exceeds_the_budget(self, equity: int) -> None:
        risk = risk_for(("752", "747"))
        quantity = size_for(risk, DEFAULT_LIMITS, Decimal(equity))
        assert quantity >= 0
        assert risk.max_loss * quantity <= DEFAULT_LIMITS.max_trade_loss(Decimal(equity))

    def test_a_later_as_of_makes_the_same_chain_stale(self) -> None:
        """The only impurity is the clock, and it is an argument."""
        built = spreads()
        fresh = build_candidates(built, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW)
        later = build_candidates(
            built,
            limits=DEFAULT_LIMITS,
            equity=EQUITY,
            as_of=NOW + timedelta(minutes=5),
        )
        assert fresh != ()
        assert later == ()


class TestTheModelFacingView:
    def test_it_carries_no_occ_symbols(self) -> None:
        """"A prompt that contains well-formed OCC symbols is a prompt that
        teaches a model to emit one." """
        for candidate in menu():
            blob = str(candidate.summarise())
            assert "SPY26" not in blob
            assert "P0074" not in blob

    def test_money_is_rendered_as_strings(self) -> None:
        summary = menu()[0].summarise()
        for key in ("net_premium", "max_loss", "max_profit"):
            assert isinstance(summary[key], str)

    def test_it_states_the_quantity_the_model_does_not_choose(self) -> None:
        summary = menu()[0].summarise()
        assert summary["quantity"] == menu()[0].quantity

    def test_unknown_greeks_render_as_null_not_zero(self) -> None:
        summary = menu(greeks=None)[0].summarise()
        assert summary["net_delta"] is None
        assert summary["net_vega"] is None


@pytest.fixture
def limits() -> RiskLimits:
    return DEFAULT_LIMITS


class TestDeltaBudget:
    """The filter a live run forced. specs/05 D4's discipline, applied to delta."""

    def test_every_candidate_fits_the_band_at_its_own_size(self) -> None:
        low, high = DEFAULT_LIMITS.scaled_delta_band(EQUITY)
        for candidate in menu():
            greeks = candidate.risk.net_greeks
            assert greeks is not None
            assert low <= greeks.delta * candidate.quantity <= high

    def test_an_existing_book_delta_narrows_the_menu(self) -> None:
        """A candidate is judged on the exposure it would leave, not the
        exposure it carries."""
        _, high = DEFAULT_LIMITS.scaled_delta_band(EQUITY)
        # Leave room for a small candidate and not a large one, so the filter is
        # visibly graded rather than all-or-nothing.
        loaded = build_candidates(
            spreads(), limits=DEFAULT_LIMITS, equity=EQUITY, as_of=NOW,
            book_delta=high - 9.0,
        )
        assert 0 < len(loaded) < len(menu())

    def test_unknown_greeks_are_left_for_the_gate(self) -> None:
        """"Silently filtering it out here would hide a data-quality problem
        behind an empty menu." `known_greeks` should veto it, visibly."""
        assert len(menu(greeks=None)) > 0


class TestSpreadsByDelta:
    """specs/07 D5's wing selection: the book names deltas, not points.

    Every quote here is built by hand, at one expiry, so each test controls
    exactly the deltas the selection has to choose between rather than
    inheriting the ladder `put_chain` was shaped for a points-width scan.
    """

    ANCHOR = 0.16
    TOLERANCE = 0.06
    WIDTH = 0.08

    @staticmethod
    def _quote(
        strike: str, delta: float | None, *, right: Right = Right.PUT
    ) -> tuple[OptionContract, OptionQuote]:
        c = contract(strike, right)
        greeks = None if delta is None else Greeks(delta, 0.01, -0.02, 0.05, -0.01, 0.18)
        return c, quote(c, "1.00", "1.05", greeks=greeks)

    def _chain(
        self, *pairs: tuple[str, float | None], right: Right = Right.PUT
    ) -> dict[OptionContract, OptionQuote]:
        return dict(self._quote(strike, delta, right=right) for strike, delta in pairs)

    def _select(
        self, quotes: dict[OptionContract, OptionQuote], *, right: Right = Right.PUT
    ) -> list[tuple[OptionStructure, object]]:
        return spreads_by_delta(
            quotes,
            right=right,
            anchor_delta=self.ANCHOR,
            anchor_tolerance=self.TOLERANCE,
            width_delta=self.WIDTH,
            as_of=NOW,
        )

    def test_exact_anchor_match_wins_the_short_leg(self) -> None:
        quotes = self._chain(("760", -0.16), ("755", -0.30), ("745", -0.08))
        built = self._select(quotes)
        assert len(built) == 1
        structure, _ = built[0]
        short = next(leg for leg in structure.legs if leg.side is Side.SELL)
        assert short.contract.strike == Decimal("760")

    def test_nearest_within_tolerance_beats_a_farther_candidate(self) -> None:
        """0.20 is 0.04 from the anchor; 0.10 is 0.06 from it — both inside
        tolerance, but 0.20 is the nearer one and must win."""
        quotes = self._chain(("760", -0.20), ("758", -0.10), ("745", -0.08))
        built = self._select(quotes)
        short = next(leg for leg in built[0][0].legs if leg.side is Side.SELL)
        assert short.contract.strike == Decimal("760")

    def test_outside_tolerance_produces_no_structure(self) -> None:
        """0.30 is 0.14 from a 0.16 anchor with a 0.06 tolerance — too far, and
        there is nothing else to anchor on."""
        quotes = self._chain(("760", -0.30), ("745", -0.08))
        assert self._select(quotes) == []

    def test_missing_greeks_are_not_ranked_at_delta_zero(self) -> None:
        """A `None`-greeks quote must not win by being mistaken for the money
        strike. The nearest strike that actually carries a delta must be
        chosen instead of the one with no greeks at all."""
        quotes = self._chain(("760", None), ("758", -0.17), ("745", -0.08))
        built = self._select(quotes)
        short = next(leg for leg in built[0][0].legs if leg.side is Side.SELL)
        assert short.contract.strike == Decimal("758")

    def test_a_wing_with_no_greeks_yields_no_structure(self) -> None:
        """The short leg resolves fine; the only possible wing has no delta to
        rank it by, so there is no fallback to a points width — nothing."""
        quotes = self._chain(("760", -0.16), ("745", None))
        assert self._select(quotes) == []

    def test_the_wing_must_be_strictly_further_out_of_the_money(self) -> None:
        """765 is a dead-on 0.08 match — closer than any real wing could be —
        but it sits *above* the anchor, which for a put is nearer the money,
        not further from it. It must be excluded regardless of how good the
        delta match is, and the genuine (if imperfect) wing must win instead."""
        quotes = self._chain(
            ("760", -0.16),  # anchor
            ("765", -0.08),  # exact width match, wrong side of the anchor
            ("745", -0.03),  # the only legitimate, further-OTM wing
        )
        built = self._select(quotes)
        long_leg = next(leg for leg in built[0][0].legs if leg.side is Side.BUY)
        assert long_leg.contract.strike == Decimal("745")

    def test_calls_select_the_higher_strike_as_the_wing(self) -> None:
        """The put case moves OTM downward; calls move OTM upward, and the same
        selection has to hold with the direction reversed."""
        quotes = self._chain(
            ("760", 0.16), ("770", 0.08), ("750", 0.30), right=Right.CALL
        )
        built = self._select(quotes, right=Right.CALL)
        assert len(built) == 1
        structure, _ = built[0]
        short = next(leg for leg in structure.legs if leg.side is Side.SELL)
        long_leg = next(leg for leg in structure.legs if leg.side is Side.BUY)
        assert short.contract.strike == Decimal("760")
        assert long_leg.contract.strike == Decimal("770")

    def test_the_menu_feeds_build_candidates_unchanged(self) -> None:
        """The output shape must be exactly what `vertical_credit_spreads`
        produces, since both are handed straight to `build_candidates`."""
        short_c = contract("760")
        long_c = contract("745")
        # Priced like a genuine credit: the near-the-money short leg costs more
        # than the further-OTM long leg, which is what makes the net premium
        # positive and lets sizing put at least one contract on the menu.
        quotes = {
            short_c: quote(
                short_c, "3.20", "3.25", greeks=Greeks(-0.16, 0.01, -0.02, 0.05, -0.01, 0.18)
            ),
            long_c: quote(
                long_c, "1.25", "1.29", greeks=Greeks(-0.08, 0.01, -0.02, 0.05, -0.01, 0.18)
            ),
        }
        built = self._select(quotes)
        # A 15-wide spread costs more than 1% of `EQUITY` to fund one contract
        # of; a larger account is used here purely so sizing does not zero the
        # candidate out before this test gets to look at it.
        funded_equity = Decimal(500_000)
        candidates = build_candidates(built, limits=DEFAULT_LIMITS, equity=funded_equity, as_of=NOW)
        assert len(candidates) == 1
        assert candidates[0].structure.legs[0].contract.strike in {
            Decimal("760"),
            Decimal("745"),
        }
