"""Search-cost deflation has to mean the same thing at every bar size.

The deflated Sharpe subtracts what a best-of-N search would have produced by
luck alone: ``sqrt(2 ln N / n)`` in per-period units, annualised. Two inputs
decide the size of that subtraction, and both were wrong for anything but daily
bars.

``n`` was the *bar count*. Six years of 5-minute bars is about 118,000 bars
against 1,500 daily ones, so the term shrank by a factor of nine -- even though
the two samples cover the same six years and therefore contain the same amount
of independent evidence about the market.

The annualisation was the literal ``sqrt(252)``, while the observed Sharpe is
annualised from the bar spacing itself. So the two halves of one subtraction
were denominated in different units.

Both errors point the same way: on intraday bars the detector would deflate
about seventy-eight times too little. Intraday research has *more* room to
overfit, not less, and the honest scorer this project is built around would have
quietly stopped objecting.
"""

from __future__ import annotations

import math

import pytest

from aqr.backtest.metrics import periods_per_year
from aqr.validation.overfitting import deflated_sharpe

DAILY = 252.0
FIVE_MIN = 252.0 * 6.5 * 12  # 19,656 five-minute bars in a trading year


class TestAnnualisationIsExplicit:
    def test_the_default_is_daily(self) -> None:
        assert deflated_sharpe(1.0, 100, 1000) == deflated_sharpe(1.0, 100, 1000, DAILY)

    def test_a_faster_bar_deflates_more_per_bar_not_less(self) -> None:
        """The subtraction is in per-period units before annualising, so a
        higher annualisation factor makes the same per-bar luck worth more."""
        slow = deflated_sharpe(2.0, 500, 1000, DAILY)
        fast = deflated_sharpe(2.0, 500, 1000, FIVE_MIN)
        assert fast < slow

    def test_the_two_halves_share_units(self) -> None:
        observed, trials, sample = 2.0, 500, 5000
        expected = observed - math.sqrt(2 * math.log(trials) / sample) * math.sqrt(FIVE_MIN)
        assert deflated_sharpe(observed, trials, sample, FIVE_MIN) == pytest.approx(expected)


class TestSameCalendarSpanSameDeflation:
    def test_daily_and_intraday_over_one_span_deflate_alike(self) -> None:
        """Six years is six years. Slicing it more finely does not buy more
        evidence about whether an edge is real, and a detector that thinks it
        does will wave through anything tested on 1-minute bars."""
        years = 6.0
        daily = deflated_sharpe(2.0, 500, int(years * DAILY), DAILY)
        intraday = deflated_sharpe(2.0, 500, int(years * FIVE_MIN), FIVE_MIN)

        assert daily == pytest.approx(intraday, abs=0.02)

    def test_the_old_behaviour_would_have_under_deflated_by_an_order_of_magnitude(
        self,
    ) -> None:
        years = 6.0
        correct = deflated_sharpe(2.0, 500, int(years * FIVE_MIN), FIVE_MIN)
        as_if_daily = deflated_sharpe(2.0, 500, int(years * FIVE_MIN), DAILY)
        # The bug made an inflated Sharpe survive nearly untouched.
        assert (2.0 - as_if_daily) < (2.0 - correct) / 5


class TestPeriodsPerYearIsShared:
    def test_the_metric_layer_exposes_the_factor_it_uses(self) -> None:
        # One definition, used by both the Sharpe and the deflation. Two copies
        # of this number is how the units drifted apart in the first place.
        import numpy as np

        daily = np.arange(300, dtype=np.int64) * 86_400
        assert periods_per_year(daily) == pytest.approx(252.0, rel=0.05)

    def test_five_minute_spacing_is_recognised(self) -> None:
        import numpy as np

        intraday = np.arange(300, dtype=np.int64) * 300
        assert periods_per_year(intraday) == pytest.approx(FIVE_MIN, rel=0.05)


class TestDegenerateInputs:
    def test_one_trial_deflates_nothing(self) -> None:
        assert deflated_sharpe(1.5, 1, 1000, FIVE_MIN) == 1.5

    def test_an_empty_sample_deflates_nothing(self) -> None:
        assert deflated_sharpe(1.5, 100, 0, FIVE_MIN) == 1.5

    def test_a_non_positive_annualisation_falls_back_to_daily(self) -> None:
        assert deflated_sharpe(1.5, 100, 1000, 0.0) == deflated_sharpe(1.5, 100, 1000, DAILY)
