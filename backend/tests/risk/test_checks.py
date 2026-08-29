"""03 D4 — every check, one at a time, with its boundary pinned.

Each check gets three cases: one that passes, one that vetoes, and one sitting
exactly on the limit. The third is the one worth writing. "> versus >=" is the
kind of edit that survives review and shows up during a drawdown, and the only
defence is a test that fails the moment the comparison flips.

**The convention, stated once and asserted everywhere below: a value exactly at
its limit passes.** Two checks are counters rather than magnitudes —
`position_count` and `daily_trade_cap` — and for those "at the limit" means the
cap is already reached, so they veto. `drawdown_killswitch` also vetoes at the
threshold, because the spec says it trips *at* the limit, not past it.

Checks are called directly here rather than through `evaluate`. A test that goes
through the Gate cannot tell a veto from `liquidity` apart from a veto from
`portfolio_heat` without reading strings, and this file is about the checks
themselves.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.options import Greeks, StructureRisk
from alphagate.risk import CheckResult, Intent, RiskLimits
from alphagate.risk.checks import (
    Context,
    daily_trade_cap,
    defined_risk,
    drawdown_killswitch,
    expiry_window,
    fresh_quotes,
    known_greeks,
    liquidity,
    net_delta_budget,
    net_vega_budget,
    per_trade_loss,
    portfolio_heat,
    position_count,
    underlying_concentration,
)
from tests.risk.conftest import (
    EQUITY,
    MSFT,
    NOW,
    forge_risk,
    make_risk,
    position,
    proposal,
    snapshot,
    spread_risk,
)


def ctx(
    limits: RiskLimits,
    *,
    risk: StructureRisk | None = None,
    quantity: int = 1,
    intent: Intent = Intent.OPEN,
    portfolio: object | None = None,
    as_of: datetime = NOW,
) -> Context:
    return Context(
        proposal=proposal(risk=risk, quantity=quantity, intent=intent),
        portfolio=snapshot() if portfolio is None else portfolio,  # type: ignore[arg-type]
        limits=limits,
        as_of=as_of,
    )


def test_the_baseline_passes_every_check(limits: RiskLimits) -> None:
    """If this fails, every other test in the file is testing two things."""
    base = ctx(limits)
    from alphagate.risk.checks import CHECKS

    failures = [check(base) for check in CHECKS if not check(base).passed]
    assert not failures, [f"{f.name}: {f.detail}" for f in failures]


class TestDefinedRisk:
    def test_a_real_structure_always_passes(self, limits: RiskLimits) -> None:
        assert defined_risk(ctx(limits)).passed

    def test_the_domain_refuses_to_build_an_undefined_loss(self) -> None:
        """The primary defence is specs/02 D4, not this check."""
        with pytest.raises(InvariantViolation, match="max_loss must be positive"):
            make_risk(max_loss=Decimal(0))

    @pytest.mark.parametrize("bad", [Decimal(0), Decimal(-1), Decimal("NaN"), Decimal("Infinity")])
    def test_it_vetoes_a_loss_that_is_not_finite_and_positive(
        self, limits: RiskLimits, bad: Decimal
    ) -> None:
        result = defined_risk(ctx(limits, risk=forge_risk(max_loss=bad)))
        assert not result.passed

    def test_the_smallest_positive_loss_passes(self, limits: RiskLimits) -> None:
        """Boundary: positive is positive, however small."""
        assert defined_risk(ctx(limits, risk=forge_risk(max_loss=Decimal("0.01")))).passed


class TestKnownGreeks:
    def test_present_greeks_pass(self, limits: RiskLimits) -> None:
        assert known_greeks(ctx(limits)).passed

    def test_an_open_with_unknown_exposure_is_vetoed(self, limits: RiskLimits) -> None:
        result = known_greeks(ctx(limits, risk=spread_risk(greeks=None)))
        assert not result.passed
        assert result.observed == "missing"

    @pytest.mark.parametrize("intent", [Intent.CLOSE, Intent.ROLL])
    def test_only_opens_are_held_to_it(self, limits: RiskLimits, intent: Intent) -> None:
        """specs/03 D4 scopes this one to `intent is OPEN`, literally.

        A roll still has to state its exposure, but that is the delta and vega
        budgets' job — they refuse an unknown net outright.
        """
        assert known_greeks(ctx(limits, risk=spread_risk(greeks=None), intent=intent)).passed


class TestFreshQuotes:
    def test_a_current_quote_passes(self, limits: RiskLimits) -> None:
        assert fresh_quotes(ctx(limits)).passed

    def test_a_stale_quote_is_vetoed(self, limits: RiskLimits) -> None:
        result = fresh_quotes(ctx(limits, risk=spread_risk(age=61)))
        assert not result.passed
        assert result.observed == pytest.approx(61.0)

    def test_exactly_at_the_age_limit_passes(self, limits: RiskLimits) -> None:
        """Inclusive. 60.0s old is fresh; 60.1s is not."""
        assert fresh_quotes(ctx(limits, risk=spread_risk(age=60))).passed
        assert not fresh_quotes(ctx(limits, risk=make_risk(quote_age_seconds=60.1))).passed

    def test_a_quote_from_the_future_is_not_stale(self, limits: RiskLimits) -> None:
        """Clock skew reads as negative age (specs/02 D2) and must not veto."""
        assert fresh_quotes(ctx(limits, risk=make_risk(quote_age_seconds=-5.0))).passed


class TestPerTradeLoss:
    def test_a_small_order_passes(self, limits: RiskLimits) -> None:
        assert per_trade_loss(ctx(limits)).passed  # 350 against a 1,000 limit

    def test_an_oversized_order_is_vetoed(self, limits: RiskLimits) -> None:
        result = per_trade_loss(ctx(limits, quantity=3))  # 1,050 against 1,000
        assert not result.passed
        assert result.observed == Decimal(1050)

    def test_exactly_at_the_limit_passes(self, limits: RiskLimits) -> None:
        """1% of $35,000 is $350, which is exactly this spread's maximum loss."""
        at = snapshot(equity=Decimal(35_000))
        just_under = snapshot(equity=Decimal(34_999))
        assert per_trade_loss(ctx(limits, portfolio=at)).passed
        assert not per_trade_loss(ctx(limits, portfolio=just_under)).passed


class TestPortfolioHeat:
    def test_an_empty_book_passes(self, limits: RiskLimits) -> None:
        assert portfolio_heat(ctx(limits)).passed

    def test_a_hot_book_is_vetoed(self, limits: RiskLimits) -> None:
        book = snapshot(positions=(position(max_loss="4651"),))
        assert not portfolio_heat(ctx(limits, portfolio=book)).passed

    def test_exactly_at_the_limit_passes(self, limits: RiskLimits) -> None:
        """4,650 already on the book plus 350 proposed is exactly 5% of 100k."""
        book = snapshot(positions=(position(max_loss="4650"),))
        result = portfolio_heat(ctx(limits, portfolio=book))
        assert result.passed
        assert result.observed == Decimal(5000)

    def test_heat_counts_the_proposal_not_just_the_book(self, limits: RiskLimits) -> None:
        """The check is about the book *after* the fill, not before it."""
        book = snapshot(positions=(position(max_loss="4900"),))
        assert not portfolio_heat(ctx(limits, portfolio=book)).passed


class TestPositionCount:
    def test_room_on_the_book_passes(self, limits: RiskLimits) -> None:
        book = snapshot(positions=tuple(position(underlying=MSFT) for _ in range(7)))
        assert position_count(ctx(limits, portfolio=book)).passed

    def test_at_the_cap_it_vetoes(self, limits: RiskLimits) -> None:
        """A counter, so "at the limit" means the cap is already used up."""
        book = snapshot(positions=tuple(position(underlying=MSFT) for _ in range(8)))
        result = position_count(ctx(limits, portfolio=book))
        assert not result.passed
        assert result.observed == 8


class TestUnderlyingConcentration:
    def test_a_diversified_book_passes(self, limits: RiskLimits) -> None:
        book = snapshot(positions=(position(underlying=MSFT, max_loss="1900"),))
        assert underlying_concentration(ctx(limits, portfolio=book)).passed

    def test_piling_into_one_name_is_vetoed(self, limits: RiskLimits) -> None:
        book = snapshot(positions=(position(max_loss="1651"),))
        assert not underlying_concentration(ctx(limits, portfolio=book)).passed

    def test_exactly_at_the_limit_passes(self, limits: RiskLimits) -> None:
        """1,650 in AAPL plus 350 proposed is exactly 2% of 100k."""
        book = snapshot(positions=(position(max_loss="1650"),))
        assert underlying_concentration(ctx(limits, portfolio=book)).passed

    def test_other_underlyings_do_not_count(self, limits: RiskLimits) -> None:
        book = snapshot(positions=(position(underlying=MSFT, max_loss="1900"),))
        result = underlying_concentration(ctx(limits, portfolio=book))
        assert result.observed == Decimal(350)


class TestGreekBudgets:
    def test_a_neutral_book_passes(self, limits: RiskLimits) -> None:
        assert net_delta_budget(ctx(limits)).passed
        assert net_vega_budget(ctx(limits)).passed

    def test_delta_exactly_on_the_band_passes(self, limits: RiskLimits) -> None:
        _, high = limits.scaled_delta_band(EQUITY)
        at = make_risk(net_greeks=Greeks(high, 0.0, 0.0, 0.0, 0.0, 0.25))
        beyond = make_risk(net_greeks=Greeks(high * 1.01, 0.0, 0.0, 0.0, 0.0, 0.25))
        assert net_delta_budget(ctx(limits, risk=at)).passed
        assert not net_delta_budget(ctx(limits, risk=beyond)).passed

    def test_the_band_is_two_sided(self, limits: RiskLimits) -> None:
        low, _ = limits.scaled_delta_band(EQUITY)
        short = make_risk(net_greeks=Greeks(low * 1.01, 0.0, 0.0, 0.0, 0.0, 0.25))
        assert not net_delta_budget(ctx(limits, risk=short)).passed

    def test_vega_exactly_on_the_band_passes(self, limits: RiskLimits) -> None:
        _, high = limits.scaled_vega_band(EQUITY)
        at = make_risk(net_greeks=Greeks(0.0, 0.0, 0.0, high, 0.0, 0.25))
        beyond = make_risk(net_greeks=Greeks(0.0, 0.0, 0.0, high * 1.01, 0.0, 0.25))
        assert net_vega_budget(ctx(limits, risk=at)).passed
        assert not net_vega_budget(ctx(limits, risk=beyond)).passed

    def test_quantity_multiplies_the_proposed_exposure(self, limits: RiskLimits) -> None:
        _, high = limits.scaled_delta_band(EQUITY)
        half = make_risk(net_greeks=Greeks(high * 0.6, 0.0, 0.0, 0.0, 0.0, 0.25))
        assert net_delta_budget(ctx(limits, risk=half)).passed
        assert not net_delta_budget(ctx(limits, risk=half, quantity=2)).passed

    def test_an_unknown_book_exposure_vetoes(self, limits: RiskLimits) -> None:
        """`None` is not zero. A book that cannot state its delta is not neutral."""
        book = snapshot(positions=(position(greeks=None),))
        result = net_delta_budget(ctx(limits, portfolio=book))
        assert not result.passed
        assert result.observed is None
        assert "the book" in result.detail

    def test_an_unknown_proposal_exposure_vetoes(self, limits: RiskLimits) -> None:
        result = net_delta_budget(ctx(limits, risk=spread_risk(greeks=None)))
        assert not result.passed
        assert "the proposal" in result.detail

    def test_the_band_scales_with_equity(self, limits: RiskLimits) -> None:
        """±0.30 per $1k: a 30-delta position is at the edge on 100k, over on 50k."""
        risk = make_risk(net_greeks=Greeks(30.0, 0.0, 0.0, 0.0, 0.0, 0.25))
        small = snapshot(equity=Decimal(50_000))
        assert net_delta_budget(ctx(limits, risk=risk)).passed
        assert not net_delta_budget(ctx(limits, risk=risk, portfolio=small)).passed


class TestLiquidity:
    def test_a_tight_market_passes(self, limits: RiskLimits) -> None:
        assert liquidity(ctx(limits)).passed

    def test_a_wide_market_is_vetoed(self, limits: RiskLimits) -> None:
        result = liquidity(ctx(limits, risk=spread_risk(spread="0.08")))
        assert not result.passed

    def test_exactly_at_the_limit_passes(self, limits: RiskLimits) -> None:
        assert liquidity(ctx(limits, risk=make_risk(worst_spread_pct=Decimal("0.05")))).passed
        assert not liquidity(
            ctx(limits, risk=make_risk(worst_spread_pct=Decimal("0.0501")))
        ).passed


class TestExpiryWindow:
    @pytest.mark.parametrize("dte", [3, 10, 21])
    def test_inside_the_window_passes(self, limits: RiskLimits, dte: int) -> None:
        """Inclusive on both ends."""
        assert expiry_window(ctx(limits, risk=make_risk(days_to_expiry=dte))).passed

    @pytest.mark.parametrize("dte", [-1, 0, 2, 22, 45])
    def test_outside_the_window_is_vetoed(self, limits: RiskLimits, dte: int) -> None:
        """0DTE is excluded by the lower bound, not by a special case."""
        assert not expiry_window(ctx(limits, risk=make_risk(days_to_expiry=dte))).passed


class TestDrawdownKillswitch:
    def test_a_healthy_account_passes(self, limits: RiskLimits) -> None:
        assert drawdown_killswitch(ctx(limits, portfolio=snapshot(drawdown="0.04"))).passed

    def test_it_trips_at_the_threshold_not_past_it(self, limits: RiskLimits) -> None:
        """The one check whose boundary is exclusive on the safe side."""
        assert drawdown_killswitch(ctx(limits, portfolio=snapshot(drawdown="0.0499"))).passed
        assert not drawdown_killswitch(ctx(limits, portfolio=snapshot(drawdown="0.05"))).passed

    def test_the_latch_survives_a_recovery(self, limits: RiskLimits) -> None:
        """Earning a little back does not silently re-arm the strategy."""
        recovered = snapshot(drawdown="0.01", killswitch_tripped=True)
        result = drawdown_killswitch(ctx(limits, portfolio=recovered))
        assert not result.passed
        assert "re-armed by hand" in result.detail


class TestDailyTradeCap:
    def test_under_the_cap_passes(self, limits: RiskLimits) -> None:
        assert daily_trade_cap(ctx(limits, portfolio=snapshot(fills_today=14))).passed

    def test_at_the_cap_it_vetoes(self, limits: RiskLimits) -> None:
        assert not daily_trade_cap(ctx(limits, portfolio=snapshot(fills_today=15))).passed


class TestCheckResults:
    def test_a_passing_check_still_reports_its_numbers(self, limits: RiskLimits) -> None:
        """The dashboard renders near-misses, so `observed` survives a pass."""
        book = snapshot(positions=(position(max_loss="4600"),))
        result = portfolio_heat(ctx(limits, portfolio=book))
        assert result.passed
        assert result.observed == Decimal(4950)
        assert result.limit == Decimal(5000)

    def test_every_check_names_itself_after_the_spec(self, limits: RiskLimits) -> None:
        from alphagate.risk.checks import CHECKS

        expected = [
            "defined_risk",
            "known_greeks",
            "fresh_quotes",
            "per_trade_loss",
            "portfolio_heat",
            "position_count",
            "underlying_concentration",
            "net_delta_budget",
            "net_vega_budget",
            "liquidity",
            "expiry_window",
            "drawdown_killswitch",
            "daily_trade_cap",
        ]
        assert [check(ctx(limits)).name for check in CHECKS] == expected

    def test_results_are_values(self) -> None:
        a = CheckResult("liquidity", True, "fine", Decimal("0.02"), Decimal("0.05"))
        b = CheckResult("liquidity", True, "fine", Decimal("0.02"), Decimal("0.05"))
        assert a == b


class TestLimitsConfiguration:
    """D5 — the competition defaults, asserted rather than assumed."""

    def test_the_defaults_are_the_spec(self, limits: RiskLimits) -> None:
        assert limits.max_trade_loss(EQUITY) == Decimal(1000)
        assert limits.max_portfolio_loss(EQUITY) == Decimal(5000)
        assert limits.max_per_underlying(EQUITY) == Decimal(2000)
        assert limits.max_open_structures == 8
        assert limits.max_daily_trades == 15
        assert limits.max_quote_age == 60.0
        assert limits.dte_range == (3, 21)
        assert limits.max_spread_pct == Decimal("0.05")
        assert limits.max_drawdown_pct == Decimal("0.05")

    def test_zero_dte_is_outside_the_configured_window(self, limits: RiskLimits) -> None:
        """The headline strategy claim of D5, as a test rather than a comment."""
        assert limits.dte_range[0] > 0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_trade_loss_pct", Decimal(0)),
            ("max_portfolio_loss_pct", Decimal(-1)),
            ("max_open_structures", 0),
            ("max_daily_trades", 0),
            ("max_quote_age", 0.0),
        ],
    )
    def test_a_limit_cannot_be_disabled(
        self, limits: RiskLimits, field: str, value: object
    ) -> None:
        """"Configurable, but never disabled" — D4's second table heading."""
        from dataclasses import replace

        with pytest.raises(InvariantViolation):
            replace(limits, **{field: value})

    def test_an_inverted_range_is_refused(self, limits: RiskLimits) -> None:
        from dataclasses import replace

        with pytest.raises(InvariantViolation, match="inverted"):
            replace(limits, dte_range=(21, 3))
        with pytest.raises(InvariantViolation, match="inverted"):
            replace(limits, delta_band=(0.30, -0.30))


class TestSnapshotInvariants:
    def test_equity_must_be_positive(self) -> None:
        """Every budgeted limit is a fraction of it; zero equity is not a book."""
        with pytest.raises(InvariantViolation, match="equity must be positive"):
            snapshot(equity=Decimal(0))

    def test_missing_greeks_poison_the_aggregate(self) -> None:
        book = snapshot(positions=(position(), position(greeks=None)))
        assert book.net_delta is None
        assert book.net_vega is None

    def test_exposure_is_per_underlying(self) -> None:
        book = snapshot(
            positions=(position(max_loss="100"), position(underlying=MSFT, max_loss="900"))
        )
        assert book.exposure_to(position().underlying) == Decimal(100)
        assert book.open_risk == Decimal(1000)


class TestProposalInvariants:
    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="quantity must be positive"):
            proposal(quantity=0)

    def test_it_needs_an_identity(self) -> None:
        with pytest.raises(InvariantViolation, match="proposal_id"):
            proposal(proposal_id="  ")

    def test_the_rationale_is_carried_verbatim(self) -> None:
        """Evidence for the journal, never input to a check."""
        assert "IV rank" in proposal().rationale


def test_days_to_expiry_is_calendar_days_from_the_supplied_date() -> None:
    """Sanity on the fixture itself: 2026-09-11 is 10 days after 2026-09-01."""
    assert spread_risk().days_to_expiry == 10
    assert (date(2026, 9, 11) - datetime(2026, 9, 1, tzinfo=UTC).date()).days == 10
