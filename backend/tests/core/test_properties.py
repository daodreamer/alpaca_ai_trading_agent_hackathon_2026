"""Property-based invariants — specs/14-testing.md.

These check the claims that must hold for *every* input, not just the examples
picked in the unit tests.
"""

from __future__ import annotations

import itertools
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from hypothesis import assume, given
from hypothesis import strategies as st

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import ticker
from alphagate.core.levels import Zone
from alphagate.core.numeric import price
from alphagate.core.time_model import SessionKind, Timeframe
from alphagate.core.trend import TrendPhase, is_valid_transition

AAPL = ticker("AAPL")
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
SESSION_DATE = date(2026, 8, 13)

prices = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("100000"),
    allow_nan=False,
    allow_infinity=False,
    places=8,
)


@given(prices)
def test_price_quantization_is_idempotent(raw: Decimal) -> None:
    assert price(price(raw)) == price(raw)


@given(prices, prices)
def test_zone_contains_its_own_bounds_and_midpoint(a: Decimal, b: Decimal) -> None:
    low, high = sorted((a, b))
    zone = Zone(low=low, high=high)
    assert zone.contains(zone.low)
    assert zone.contains(zone.high)
    assert zone.contains(zone.midpoint)
    assert zone.width >= 0


@given(st.lists(prices, min_size=4, max_size=4))
def test_bar_accepts_exactly_the_ohlc_consistent_orderings(values: list[Decimal]) -> None:
    """A bar is constructible iff the OHLC invariants hold — no wider, no narrower."""
    open_, high, low, close = values
    consistent = high >= max(open_, close) and low <= min(open_, close) and high >= low

    def build() -> Bar:
        return Bar(
            symbol=AAPL,
            timeframe=Timeframe.H1,
            start_time_utc=SESSION_OPEN,
            end_time_utc=SESSION_OPEN + timedelta(hours=1),
            session_date=SESSION_DATE,
            session=SessionKind.REGULAR,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=Decimal(0),
            source="fake",
            feed=Feed.SYNTHETIC,
            adjustment_mode=AdjustmentMode.UNADJUSTED,
            is_final=True,
        )

    if consistent:
        bar = build()
        assert bar.high >= bar.low
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low <= min(bar.open, bar.close)
    else:
        try:
            build()
        except Exception:  # any rejection is acceptable here
            return
        raise AssertionError("inconsistent OHLC was accepted")


@given(st.integers(min_value=0, max_value=3599))
def test_bar_membership_partitions_its_window(offset_seconds: int) -> None:
    """Every instant in [start, end) is in the bar; end itself is in the next one."""
    first = Bar(
        symbol=AAPL,
        timeframe=Timeframe.H1,
        start_time_utc=SESSION_OPEN,
        end_time_utc=SESSION_OPEN + timedelta(hours=1),
        session_date=SESSION_DATE,
        session=SessionKind.REGULAR,
        open=price("100"),
        high=price("100"),
        low=price("100"),
        close=price("100"),
        volume=Decimal(0),
        source="fake",
        feed=Feed.SYNTHETIC,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        is_final=True,
    )
    instant = SESSION_OPEN + timedelta(seconds=offset_seconds)
    assert first.contains(instant)
    assert not first.contains(first.end_time_utc)


phases = st.sampled_from(list(TrendPhase))


@given(phases, phases)
def test_transition_validity_is_total_and_reflexive(a: TrendPhase, b: TrendPhase) -> None:
    assert isinstance(is_valid_transition(a, b), bool)
    assert is_valid_transition(a, a)


@given(phases, phases)
def test_direction_flips_require_passing_through_neutral(a: TrendPhase, b: TrendPhase) -> None:
    """The hysteresis guarantee: no single transition crosses from bull to bear."""
    assume(a in TrendPhase.ladder() and b in TrendPhase.ladder())
    assume(a.direction.is_directional and b.direction.is_directional)
    assume(a.direction is not b.direction)
    assert not is_valid_transition(a, b)


def test_every_ladder_phase_is_reachable_from_every_other() -> None:
    """One step at a time, but never a dead end."""
    ladder = TrendPhase.ladder()
    for start, goal in itertools.product(ladder, repeat=2):
        seen = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for nxt in ladder:
                if nxt not in seen and is_valid_transition(current, nxt):
                    seen.add(nxt)
                    frontier.append(nxt)
        assert goal in seen
