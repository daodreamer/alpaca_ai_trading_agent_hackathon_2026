"""05 D1 step 2 and D8 — the screen, and the cadence.

Both are pure, which is the point of both. A loop that decides when to run while
running can only be tested by waiting; a function that returns the times can be
tested against a half day and a holiday in microseconds. A screen that consulted
a model could only be tested by mocking one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from alphagate.agent.earnings import (
    ETF_UNDERLYINGS,
    NoEarningsCalendar,
    StaticEarningsCalendar,
    earnings_within,
)
from alphagate.agent.model import MarketRead
from alphagate.agent.schedule import (
    CYCLE_INTERVAL,
    FIRST_CYCLE_DELAY,
    MARKET_TZ,
    CycleKind,
    next_slot,
    session_slots,
)
from alphagate.agent.screen import BIAS_BEARISH, BIAS_BULLISH, BIAS_NEUTRAL, DefaultScreen
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from tests.agent.conftest import SPY, read

AAPL = ticker("AAPL")
OPEN = datetime(2026, 8, 26, 13, 30, tzinfo=UTC)  # 09:30 ET
CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)  # 16:00 ET
HALF_DAY_CLOSE = datetime(2026, 8, 26, 17, 0, tzinfo=UTC)  # 13:00 ET


class FakeTrend:
    """Minimal stand-in: the screen reads `state.value` and `strength`."""

    def __init__(self, phase: str, strength: float = 60.0) -> None:
        self.state = type("Phase", (), {"value": phase})()
        self.strength = strength
        self.confidence = 1.0


def complete_read(**overrides: object) -> MarketRead:
    base = {
        "underlying": SPY,
        "as_of": OPEN,
        "spot": Decimal(765),
        "atr_pct": Decimal("0.81"),
        "iv_vs_hv": Decimal("1.07"),
        "earnings_within_dte": False,
        "trend": FakeTrend("BULLISH"),
    }
    base.update(overrides)
    return MarketRead(**base)  # type: ignore[arg-type]


class TestSchedule:
    def test_the_first_cycle_is_twenty_minutes_after_the_open(self) -> None:
        """"Not on the open itself — the first minutes' quotes are the widest of
        the day and `worst_spread_pct` would veto most candidates anyway." """
        slots = session_slots(OPEN, CLOSE)
        assert slots[0].at == OPEN + FIRST_CYCLE_DELAY
        assert slots[0].local.strftime("%H:%M") == "09:50"

    def test_it_runs_every_fifteen_minutes(self) -> None:
        slots = session_slots(OPEN, CLOSE)
        gaps = {b.at - a.at for a, b in pairwise(slots)}
        assert gaps == {CYCLE_INTERVAL}

    def test_a_full_session_has_the_expected_count(self) -> None:
        """09:50 to 15:50 inclusive, every 15 minutes."""
        slots = session_slots(OPEN, CLOSE)
        assert len(slots) == 25
        assert slots[-1].local.strftime("%H:%M") == "15:50"

    def test_nothing_runs_at_or_after_the_bell(self) -> None:
        """"A cycle at exactly 16:00:00 would read a market that has already
        stopped and place an order that cannot fill." """
        assert all(slot.at < CLOSE for slot in session_slots(OPEN, CLOSE))

    def test_the_last_slots_are_exits_only(self) -> None:
        """A `day` limit placed at 15:58 that does not fill simply expires."""
        slots = session_slots(OPEN, CLOSE)
        assert slots[-1].kind is CycleKind.EXITS_ONLY
        assert not slots[-1].kind.may_open
        assert slots[-2].kind is CycleKind.FULL

    def test_exits_are_never_scheduled_away(self) -> None:
        """specs/03 D4 through the scheduler: every slot may close."""
        assert all(slot.kind in set(CycleKind) for slot in session_slots(OPEN, CLOSE))
        assert len([s for s in session_slots(OPEN, CLOSE) if not s.kind.may_open]) >= 1

    def test_a_half_day_is_shorter_not_wrong(self) -> None:
        """A 13:00 close is a real thing three times a year, and a hardcoded
        16:00 would have the agent proposing into a closed market."""
        slots = session_slots(OPEN, HALF_DAY_CLOSE)
        assert len(slots) == 13
        assert slots[-1].local.strftime("%H:%M") == "12:50"
        assert all(slot.at < HALF_DAY_CLOSE for slot in slots)

    def test_sequences_are_dense_and_stable(self) -> None:
        """They feed `cycle_id_for`, so the third cycle of a day must have the
        same id on a replay as it had live."""
        slots = session_slots(OPEN, CLOSE)
        assert [slot.sequence for slot in slots] == list(range(len(slots)))
        assert session_slots(OPEN, CLOSE) == session_slots(OPEN, CLOSE)

    def test_next_slot_is_strictly_after(self) -> None:
        """"A runner that wakes a millisecond early must not run the same slot
        twice" — that would be two journal lines claiming one decision."""
        slots = session_slots(OPEN, CLOSE)
        assert next_slot(slots, slots[0].at) is slots[1]
        assert next_slot(slots, slots[0].at - timedelta(microseconds=1)) is slots[0]

    def test_next_slot_is_none_once_the_session_is_done(self) -> None:
        assert next_slot(session_slots(OPEN, CLOSE), CLOSE) is None

    def test_naive_bounds_are_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="tz-aware"):
            session_slots(datetime(2026, 8, 26, 9, 30), CLOSE)  # noqa: DTZ001

    def test_an_inverted_session_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="closes"):
            session_slots(CLOSE, OPEN)

    def test_the_market_timezone_is_eastern(self) -> None:
        assert str(MARKET_TZ) == "America/New_York"


class TestScreen:
    def test_a_complete_bullish_read_produces_a_setup(self) -> None:
        setup = DefaultScreen().screen(complete_read())
        assert setup is not None
        assert setup.bias == BIAS_BULLISH
        assert "BULLISH" in setup.name.upper()

    def test_a_bearish_trend_flips_the_bias(self) -> None:
        setup = DefaultScreen().screen(complete_read(trend=FakeTrend("BEARISH")))
        assert setup is not None
        assert setup.bias == BIAS_BEARISH

    def test_a_neutral_trend_argues_for_neither_side(self) -> None:
        """"A real answer and not a shrug: the caller builds a two-sided
        structure or nothing." """
        assert DefaultScreen.bias_of(complete_read(trend=FakeTrend("NEUTRAL"))) == BIAS_NEUTRAL

    def test_an_unmeasured_trend_produces_no_setup(self) -> None:
        screen = DefaultScreen()
        assert screen.screen(complete_read(trend=None)) is None
        assert screen.explain(complete_read(trend=None)) == "trend unmeasured"

    def test_earnings_inside_the_window_produces_no_setup(self) -> None:
        screen = DefaultScreen()
        assert screen.screen(complete_read(earnings_within_dte=True)) is None
        assert "earnings inside" in screen.explain(complete_read(earnings_within_dte=True))

    def test_an_unchecked_earnings_date_also_produces_no_setup(self) -> None:
        """The distinction that costs money: `None` is not `False`."""
        screen = DefaultScreen()
        assert screen.screen(complete_read(earnings_within_dte=None)) is None
        assert "no earnings calendar" in screen.explain(complete_read(earnings_within_dte=None))

    def test_unmeasured_volatility_produces_no_setup(self) -> None:
        assert DefaultScreen().screen(complete_read(iv_vs_hv=None)) is None
        assert DefaultScreen().screen(complete_read(atr_pct=None)) is None

    def test_a_weak_trend_produces_no_setup(self) -> None:
        screen = DefaultScreen()
        weak = complete_read(trend=FakeTrend("BULLISH", strength=10.0))
        assert screen.screen(weak) is None
        assert "strength below" in screen.explain(weak)

    def test_the_refusal_reason_is_recorded_not_raised(self) -> None:
        """`NO_SETUP` entries are the majority, and "why didn't it trade at
        14:30?" is the question they exist to answer."""
        assert DefaultScreen().explain(complete_read()) == "setup found"

    def test_it_consults_no_model(self) -> None:
        import inspect

        parameters = set(inspect.signature(DefaultScreen.screen).parameters)
        assert parameters == {"self", "read"}

    def test_it_is_deterministic(self) -> None:
        read_ = complete_read()
        first, second = DefaultScreen().screen(read_), DefaultScreen().screen(read_)
        assert first == second


class TestEarnings:
    def test_an_etf_never_reports(self) -> None:
        """A fact about the instrument, not a lookup."""
        assert SPY in ETF_UNDERLYINGS
        assert (
            earnings_within(
                NoEarningsCalendar(), SPY, as_of=date(2026, 8, 26), through=date(2026, 9, 30)
            )
            is False
        )

    def test_an_unknown_single_name_is_unknown_not_absent(self) -> None:
        """"We did not look" must never render as "there is none"."""
        assert (
            earnings_within(
                NoEarningsCalendar(), AAPL, as_of=date(2026, 8, 26), through=date(2026, 9, 30)
            )
            is None
        )

    def test_a_hand_maintained_date_inside_the_window(self) -> None:
        calendar = StaticEarningsCalendar({AAPL: (date(2026, 9, 3),)})
        assert (
            earnings_within(
                calendar, AAPL, as_of=date(2026, 8, 26), through=date(2026, 9, 30)
            )
            is True
        )

    def test_a_hand_maintained_date_outside_the_window(self) -> None:
        calendar = StaticEarningsCalendar({AAPL: (date(2026, 11, 3),)})
        assert (
            earnings_within(
                calendar, AAPL, as_of=date(2026, 8, 26), through=date(2026, 9, 30)
            )
            is False
        )

    def test_a_known_name_with_no_upcoming_date_is_unknown(self) -> None:
        """A calendar that has run out is not a calendar that says "never"."""
        calendar = StaticEarningsCalendar({AAPL: (date(2026, 1, 3),)})
        assert (
            earnings_within(
                calendar, AAPL, as_of=date(2026, 8, 26), through=date(2026, 9, 30)
            )
            is None
        )

    def test_the_next_date_is_the_next_one(self) -> None:
        calendar = StaticEarningsCalendar(
            {AAPL: (date(2026, 11, 3), date(2026, 9, 3), date(2026, 1, 3))}
        )
        assert calendar.next_earnings(AAPL, date(2026, 8, 26)) == date(2026, 9, 3)

    def test_perception_defaults_to_the_fail_closed_calendar(self) -> None:
        """No calendar configured means every single name reads `unmeasured`."""
        assert not NoEarningsCalendar().is_known(AAPL)
        assert NoEarningsCalendar().is_known(SPY)


def test_a_read_is_only_complete_when_earnings_were_checked() -> None:
    """`is_complete` gates the whole pipeline, and it requires an explicit
    `False` — neither an event nor an unchecked calendar is a book to sell
    premium into."""
    assert complete_read().is_complete
    assert not complete_read(earnings_within_dte=None).is_complete
    assert not complete_read(earnings_within_dte=True).is_complete


def test_the_fixture_read_still_builds() -> None:
    assert read().underlying == SPY
    assert read().as_of.tzinfo is UTC


class TestWatchlist:
    """specs/05 D8. One place, reviewable, and honest about what it can trade."""

    def test_the_universe_is_spy_alone(self) -> None:
        """specs/07 D2. One underlying, and it is a decision rather than a stub.

        SPY is the only instrument that satisfies all four of that section's
        requirements at once: a market inside the Gate's 5% limit, no earnings
        by construction, weekly expiries, and free history to backtest on.
        """
        from alphagate.agent import WATCHLIST

        assert [str(entry.symbol) for entry in WATCHLIST] == ["SPY"]

    def test_every_name_has_a_width_that_suits_its_price(self) -> None:
        """Five dollars is a sensible wing on a $700 index and most of a $40
        stock. Fixed per name because strike ladders are quoted in increments,
        and a computed width lands between strikes that do not exist."""
        from alphagate.agent import WATCHLIST

        assert all(entry.spread_width > 0 for entry in WATCHLIST)

    def test_every_watchlist_name_answers_earnings_structurally(self) -> None:
        """specs/07 test plan 9, and the reason D2's universe is what it is.

        Asserted against `ETF_UNDERLYINGS` rather than against the string
        "SPY", so a name added without a hand-checked report date fails here
        rather than passing by not being the one the test was written for.
        Alpaca serves no earnings calendar, and an earnings print inside the
        holding period is the single most common way a defined-risk premium
        sale becomes a maximum loss.
        """
        from alphagate.agent import WATCHLIST
        from alphagate.agent.earnings import ETF_UNDERLYINGS

        assert {entry.symbol for entry in WATCHLIST} <= ETF_UNDERLYINGS

    def test_the_whole_watchlist_is_tradeable_and_nothing_is_sitting_out(self) -> None:
        """specs/07 test plan 10. Under D2 an empty return is never expected.

        `tradeable_today()` is still called rather than assumed, so if the
        calendar ever stops answering for an entry the watchlist shrinks
        visibly instead of the agent idling through a day with nothing to do.
        """
        from alphagate.agent import WATCHLIST, tradeable_today
        from alphagate.agent.watchlist import next_earnings_gaps

        assert {e.symbol for e in tradeable_today()} == {e.symbol for e in WATCHLIST}
        assert next_earnings_gaps() == ()

    def test_an_empty_calendar_declines_rather_than_guesses(self) -> None:
        """"An empty mapping that declines is honest; a populated one that was
        guessed at is not." """
        from alphagate.agent import COMPETITION_EARNINGS

        assert COMPETITION_EARNINGS.dates == {}

    def test_a_single_name_needs_a_date_before_the_screen_can_admit_it(self) -> None:
        """The mechanism D2 depends on, tested where it lives.

        No single name is on the watchlist, so this exercises the calendar
        rather than the list: unknown until a date is filled in, known
        afterwards. That is the one line of work a name outside
        `ETF_UNDERLYINGS` would cost, and it is a line a person has to sign
        for.
        """
        assert not StaticEarningsCalendar({}).is_known(AAPL)
        assert StaticEarningsCalendar({AAPL: (date(2026, 10, 30),)}).is_known(AAPL)
