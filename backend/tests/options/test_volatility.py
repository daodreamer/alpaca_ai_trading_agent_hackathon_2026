"""Black–Scholes inversion, realised volatility, and IV rank.

The rank tests exist because of a specific, real mistake. The first live agent
cycle handed the model an implied-volatility *level* (15.79) in a field called
`iv_rank`; the model read "IV rank is low (15.79)" and reasoned correctly from a
number that meant something else. So this file pins the distinction hard: a rank
is a position inside a trailing range, it lives in [0, 1], and when there is not
enough history to compute one the answer is `None` — never a middling default,
because a default here is a claim about how rich premium is.

The Black–Scholes tests are mostly round-trips and identities. A pricing formula
tested against hand-picked expected values tests the arithmetic once; tested
against put–call parity and against its own inverse, it tests the relationships
that have to hold for every input.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import ClassVar

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from alphagate.core.errors import InvariantViolation
from alphagate.options.blackscholes import (
    MAX_VOL,
    call_price,
    implied_volatility,
    put_price,
    time_to_expiry_years,
)
from alphagate.options.volatility import (
    MIN_HISTORY,
    iv_percentile,
    iv_rank,
    iv_vs_hv,
    realised_volatility,
    summarise_volatility,
)

SPOT = 100.0
YEAR = 1.0


def closes(values: list[str]) -> list[Decimal]:
    return [Decimal(v) for v in values]


class TestPricing:
    def test_an_at_the_money_call_is_worth_about_0_4_sigma(self) -> None:
        """The standard approximation: ATM call ≈ 0.4 · S · σ · √T."""
        price = call_price(SPOT, SPOT, 0.20, YEAR)
        assert price == pytest.approx(0.4 * SPOT * 0.20, rel=0.05)

    def test_put_call_parity_holds(self) -> None:
        """C - P = S - K·e^(-rT). The identity the put is derived from, asserted
        rather than assumed."""
        for strike in (80.0, 100.0, 120.0):
            call = call_price(SPOT, strike, 0.25, 0.5, 0.04)
            put = put_price(SPOT, strike, 0.25, 0.5, 0.04)
            assert call - put == pytest.approx(SPOT - strike * math.exp(-0.04 * 0.5))

    def test_price_rises_with_volatility(self) -> None:
        """Monotone in vol — which is what makes bisection sufficient."""
        prices = [call_price(SPOT, SPOT, vol, YEAR) for vol in (0.1, 0.2, 0.4, 0.8)]
        assert prices == sorted(prices)

    def test_an_expired_option_is_worth_its_intrinsic(self) -> None:
        assert call_price(110.0, 100.0, 0.3, 0.0) == pytest.approx(10.0)
        assert put_price(90.0, 100.0, 0.3, 0.0) == pytest.approx(10.0)

    def test_a_deep_out_of_the_money_call_is_nearly_worthless(self) -> None:
        assert call_price(SPOT, 300.0, 0.15, 0.05) < 0.01


class TestInversion:
    @pytest.mark.parametrize("vol", [0.12, 0.20, 0.45, 1.20])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
    def test_it_recovers_the_volatility_it_priced_with(
        self, vol: float, strike: float
    ) -> None:
        """The round-trip that matters: price(σ) then invert, get σ back."""
        price = call_price(SPOT, strike, vol, 0.5, 0.04)
        recovered = implied_volatility(price, SPOT, strike, 0.5, is_call=True, rate=0.04)
        assert recovered is not None
        assert recovered == pytest.approx(vol, abs=1e-4)

    @pytest.mark.parametrize("strike", [80.0, 125.0])
    def test_a_price_with_no_time_value_refuses_rather_than_guesses(
        self, strike: float
    ) -> None:
        """The bug this file found.

        A deep in-the-money call at 5% volatility has a time value of about
        1e-12. With an absolute convergence tolerance the bisection "converged"
        on its first midpoint and returned 0.04 — a confidently wrong number,
        headed straight for the series an IV rank is computed from. There is no
        information about volatility in such a price, and the honest answer is
        to say so.
        """
        price = call_price(SPOT, strike, 0.05, 0.5, 0.04)
        assert implied_volatility(price, SPOT, strike, 0.5, is_call=True, rate=0.04) is None

    def test_the_same_strikes_invert_fine_once_there_is_time_value(self) -> None:
        """The floor excludes flat prices, not deep strikes."""
        for strike in (80.0, 125.0):
            price = call_price(SPOT, strike, 0.20, 0.5, 0.04)
            recovered = implied_volatility(
                price, SPOT, strike, 0.5, is_call=True, rate=0.04
            )
            assert recovered == pytest.approx(0.20, abs=1e-4)

    def test_it_works_for_puts_too(self) -> None:
        price = put_price(SPOT, 95.0, 0.30, 0.25, 0.04)
        recovered = implied_volatility(price, SPOT, 95.0, 0.25, is_call=False, rate=0.04)
        assert recovered is not None
        assert recovered == pytest.approx(0.30, abs=1e-4)

    def test_a_price_at_intrinsic_has_no_implied_volatility(self) -> None:
        """"A fabricated volatility would enter the very series the rank is
        computed from." `None`, not a clamped floor."""
        assert implied_volatility(10.0, 110.0, 100.0, 0.5, is_call=True) is None

    def test_a_price_below_intrinsic_has_none_either(self) -> None:
        """A crossed or stale print. A real fact about the print."""
        assert implied_volatility(5.0, 110.0, 100.0, 0.5, is_call=True) is None

    def test_a_price_above_the_underlying_has_none(self) -> None:
        assert implied_volatility(120.0, 100.0, 90.0, 0.5, is_call=True) is None

    @pytest.mark.parametrize(
        ("price", "spot", "strike", "years"),
        [(0.0, 100.0, 100.0, 1.0), (5.0, 0.0, 100.0, 1.0), (5.0, 100.0, 100.0, 0.0)],
    )
    def test_degenerate_inputs_return_none(
        self, price: float, spot: float, strike: float, years: float
    ) -> None:
        assert implied_volatility(price, spot, strike, years, is_call=True) is None

    def test_an_absurd_volatility_is_out_of_bracket(self) -> None:
        """Above 500% we are fitting a model to something that is not an option."""
        price = call_price(SPOT, SPOT, MAX_VOL * 2, YEAR)
        assert implied_volatility(price, SPOT, SPOT, YEAR, is_call=True) is None

    @settings(max_examples=200)
    @given(
        vol=st.floats(min_value=0.02, max_value=2.0),
        moneyness=st.floats(min_value=0.7, max_value=1.4),
        days=st.integers(min_value=5, max_value=365),
    )
    def test_the_round_trip_is_a_property(
        self, vol: float, moneyness: float, days: int
    ) -> None:
        strike = SPOT * moneyness
        years = time_to_expiry_years(days)
        price = call_price(SPOT, strike, vol, years)
        assume(price > 1e-6)
        recovered = implied_volatility(price, SPOT, strike, years, is_call=True)
        if recovered is not None:
            assert recovered == pytest.approx(vol, abs=1e-3)


class TestTimeToExpiry:
    def test_it_uses_calendar_days(self) -> None:
        """Matching `OptionContract.days_to_expiry`. A 252-day year here and a
        calendar count there would put a systematic ~1.4x error into every
        inverted volatility."""
        assert time_to_expiry_years(365) == pytest.approx(1.0)
        assert time_to_expiry_years(30) == pytest.approx(30 / 365)

    def test_an_expired_contract_is_zero_not_negative(self) -> None:
        assert time_to_expiry_years(-5) == 0.0


class TestRealisedVolatility:
    def test_a_flat_series_has_no_volatility(self) -> None:
        assert realised_volatility(closes(["100"] * 25)) == pytest.approx(0.0)

    def test_it_is_annualised(self) -> None:
        """A steady 1% daily swing annualises to roughly 16%."""
        prices = []
        price = Decimal(100)
        for index in range(30):
            price = price * (Decimal("1.01") if index % 2 == 0 else Decimal("0.99"))
            prices.append(price)
        result = realised_volatility(prices)
        assert result is not None
        assert 0.10 < result < 0.30

    def test_too_short_a_series_returns_none(self) -> None:
        """Not zero. A volatility nobody could measure is not a calm market."""
        assert realised_volatility(closes(["100", "101", "102"])) is None

    def test_a_non_positive_close_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="must be positive"):
            realised_volatility(closes(["100"] * 20 + ["0"]))


class TestIvRank:
    """A rank is a position in a range. It is not a level."""

    HISTORY: ClassVar[list[float]] = [0.10 + 0.01 * i for i in range(MIN_HISTORY)]
    """0.10 .. 0.29, evenly spaced, so the expected ranks are checkable by eye."""

    def test_at_the_bottom_of_the_range_is_zero(self) -> None:
        assert iv_rank(0.10, self.HISTORY) == pytest.approx(0.0)

    def test_at_the_top_of_the_range_is_one(self) -> None:
        assert iv_rank(0.29, self.HISTORY) == pytest.approx(1.0)

    def test_in_the_middle_is_a_half(self) -> None:
        assert iv_rank(0.195, self.HISTORY) == pytest.approx(0.5)

    def test_it_is_clamped_outside_the_historical_range(self) -> None:
        assert iv_rank(0.50, self.HISTORY) == pytest.approx(1.0)
        assert iv_rank(0.01, self.HISTORY) == pytest.approx(0.0)

    def test_a_level_is_not_a_rank(self) -> None:
        """The regression this module exists for.

        An implied volatility of 15.79% is a *low level*. Whether it is a low
        rank depends entirely on the history it is ranked against: against a
        calm regime the same level is at the very top, against a wild one it is
        at the very bottom.
        """
        calm = [0.05 + 0.001 * i for i in range(MIN_HISTORY)]  # 5.0%..6.9%
        wild = [0.15 + 0.02 * i for i in range(MIN_HISTORY)]  # 15%..53%
        level = 0.1579
        assert iv_rank(level, calm) == pytest.approx(1.0)
        assert iv_rank(level, wild) == pytest.approx(0.0, abs=0.05)

    def test_too_little_history_gives_none(self) -> None:
        """"A rank computed from four observations is noise wearing a
        percentile's clothing." """
        assert iv_rank(0.20, [0.1, 0.2, 0.3, 0.4]) is None

    def test_a_flat_history_has_no_inside(self) -> None:
        assert iv_rank(0.20, [0.20] * MIN_HISTORY) is None

    def test_non_finite_history_points_are_dropped(self) -> None:
        polluted = [*self.HISTORY, float("nan"), float("inf"), -1.0]
        assert iv_rank(0.195, polluted) == pytest.approx(0.5)


class TestIvPercentile:
    def test_it_counts_the_history_below(self) -> None:
        history = [0.10 + 0.01 * i for i in range(MIN_HISTORY)]
        assert iv_percentile(0.20, history) == pytest.approx(0.5)

    def test_it_is_robust_where_rank_is_not(self) -> None:
        """One panic spike moves the rank a long way and the percentile barely
        at all — which is the whole argument for carrying both."""
        calm = [0.15] * 19 + [0.16]
        spiked = [*calm, 0.90]
        assert iv_rank(0.16, spiked) == pytest.approx(0.013, abs=0.01)
        assert iv_percentile(0.16, spiked) >= 0.9  # type: ignore[operator]

    def test_too_little_history_gives_none(self) -> None:
        assert iv_percentile(0.20, [0.1, 0.2]) is None


class TestIvVsHv:
    def test_above_one_means_options_price_more_than_delivered(self) -> None:
        assert iv_vs_hv(0.24, 0.12) == pytest.approx(2.0)

    def test_an_unknown_realised_makes_the_ratio_unknown(self) -> None:
        """"Reporting 1.0 would say options are fairly priced, which is a claim,
        not a missing value." """
        assert iv_vs_hv(0.24, None) is None

    def test_a_zero_realised_is_not_divided_by(self) -> None:
        assert iv_vs_hv(0.24, 0.0) is None


class TestSummary:
    def test_it_reports_what_it_could_and_could_not_compute(self) -> None:
        history = [0.10 + 0.01 * i for i in range(MIN_HISTORY)]
        read = summarise_volatility(0.195, closes(["100"] * 25), history)
        assert read.rank == pytest.approx(0.5)
        assert read.percentile is not None
        assert read.observations == MIN_HISTORY
        assert read.is_complete

    def test_no_implied_means_nothing_downstream_is_invented(self) -> None:
        read = summarise_volatility(None, closes(["100"] * 25), [0.1] * 40)
        assert read.implied is None
        assert read.rank is None
        assert read.ratio is None
        assert not read.is_complete

    def test_a_thin_history_is_visible_rather_than_shaky(self) -> None:
        """The count is recorded, so "rank unavailable" and "rank from 3 points"
        are different rows in the journal."""
        read = summarise_volatility(0.20, closes(["100"] * 25), [0.1, 0.2, 0.3])
        assert read.rank is None
        assert read.observations == 3
