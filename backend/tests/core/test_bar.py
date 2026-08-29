"""Canonical Bar — specs/03-market-data.md and ADR 0004."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from alphagate.core.bar import AdjustmentMode, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.time_model import SessionKind, Timeframe

from .conftest import SESSION_CLOSE, SESSION_OPEN, make_bar


class TestOhlcInvariants:
    def test_accepts_a_valid_bar(self) -> None:
        bar = make_bar()
        assert bar.close == Decimal("104.00000000")

    def test_high_must_be_at_least_open_and_close(self) -> None:
        with pytest.raises(InvariantViolation, match="high"):
            make_bar(open="100", high="99", low="98", close="98.5")

    def test_low_must_be_at_most_open_and_close(self) -> None:
        with pytest.raises(InvariantViolation, match="low"):
            make_bar(open="100", high="105", low="101", close="104")

    def test_high_must_be_at_least_low(self) -> None:
        with pytest.raises(InvariantViolation):
            make_bar(open="100", high="99", low="100", close="99.5")

    def test_volume_must_not_be_negative(self) -> None:
        with pytest.raises(InvariantViolation, match="volume"):
            make_bar(volume="-1")

    def test_zero_volume_is_valid(self) -> None:
        assert make_bar(volume="0").volume == Decimal("0.00000000")

    def test_prices_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="positive"):
            make_bar(open="0", high="105", low="0", close="104")

    def test_flat_bar_is_valid(self) -> None:
        bar = make_bar(open="100", high="100", low="100", close="100")
        assert bar.high == bar.low

    def test_trade_count_must_not_be_negative(self) -> None:
        with pytest.raises(InvariantViolation, match="trade_count"):
            make_bar(trade_count=-1)

    def test_vwap_must_lie_within_the_bar_range(self) -> None:
        with pytest.raises(InvariantViolation, match="vwap"):
            make_bar(vwap="106")

    def test_vwap_at_the_boundary_is_valid(self) -> None:
        assert make_bar(vwap="105").vwap == Decimal("105.00000000")


class TestPriceTypes:
    def test_prices_are_quantized_decimals(self) -> None:
        bar = make_bar(open="100.123456789")
        assert bar.open == Decimal("100.12345679")

    def test_float_prices_are_rejected(self) -> None:
        with pytest.raises(TypeError):
            make_bar(open=100.0)


class TestTimeInvariants:
    def test_end_must_follow_start(self) -> None:
        with pytest.raises(InvariantViolation, match="end_time_utc"):
            make_bar(end_time_utc=SESSION_OPEN)

    def test_naive_timestamps_are_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            make_bar(start_time_utc=datetime(2026, 8, 13, 13, 30))  # noqa: DTZ001

    def test_span_may_not_exceed_the_nominal_duration(self) -> None:
        with pytest.raises(InvariantViolation, match="nominal"):
            make_bar(end_time_utc=SESSION_OPEN + timedelta(hours=2))

    def test_truncated_session_close_bar_is_valid_and_final(self) -> None:
        # ADR 0004 D5: a 6.5h session on 1h ends with a 30-minute bar.
        bar = make_bar(
            start_time_utc=SESSION_CLOSE - timedelta(minutes=30),
            end_time_utc=SESSION_CLOSE,
        )
        assert bar.is_truncated
        assert bar.is_final

    def test_full_length_bar_is_not_truncated(self) -> None:
        assert not make_bar().is_truncated

    def test_daily_bar_spans_the_session_not_the_calendar_day(self) -> None:
        bar = make_bar(
            timeframe=Timeframe.D1,
            start_time_utc=SESSION_OPEN,
            end_time_utc=SESSION_CLOSE,
        )
        assert bar.duration == timedelta(hours=6, minutes=30)
        assert bar.is_truncated  # shorter than the 1-day nominal, by construction


class TestMembership:
    def test_contains_is_half_open(self) -> None:
        bar = make_bar()
        assert bar.contains(SESSION_OPEN)
        assert bar.contains(SESSION_OPEN + timedelta(minutes=59, seconds=59))
        assert not bar.contains(bar.end_time_utc)
        assert not bar.contains(SESSION_OPEN - timedelta(microseconds=1))

    def test_a_print_at_the_boundary_belongs_to_the_next_bar(self) -> None:
        first = make_bar()
        second = make_bar(
            start_time_utc=first.end_time_utc,
            end_time_utc=first.end_time_utc + timedelta(hours=1),
        )
        boundary = first.end_time_utc
        assert not first.contains(boundary)
        assert second.contains(boundary)

    def test_contains_accepts_an_instant_in_another_timezone(self) -> None:
        bar = make_bar()  # 13:30–14:30 UTC
        eastern = timezone(timedelta(hours=-4))
        assert bar.contains(datetime(2026, 8, 13, 9, 45, tzinfo=eastern))  # 13:45 UTC
        assert not bar.contains(datetime(2026, 8, 13, 10, 45, tzinfo=eastern))  # 14:45 UTC

    def test_contains_rejects_a_naive_instant(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            make_bar().contains(datetime(2026, 8, 13, 13, 45))  # noqa: DTZ001


class TestIdentity:
    def test_key_matches_the_canonical_uniqueness_tuple(self) -> None:
        bar = make_bar()
        assert bar.key == (
            bar.symbol,
            bar.timeframe,
            bar.start_time_utc,
            bar.feed,
            bar.adjustment_mode,
        )

    def test_same_window_on_a_different_feed_is_a_different_bar(self) -> None:
        assert make_bar(feed=Feed.IEX).key != make_bar(feed=Feed.SIP).key

    def test_same_window_with_a_different_adjustment_is_a_different_bar(self) -> None:
        unadjusted = make_bar(adjustment_mode=AdjustmentMode.UNADJUSTED)
        adjusted = make_bar(adjustment_mode=AdjustmentMode.ADJUSTED)
        assert unadjusted.key != adjusted.key

    def test_revision_does_not_change_identity(self) -> None:
        assert make_bar(revision=0).key == make_bar(revision=3).key

    def test_revision_must_not_be_negative(self) -> None:
        with pytest.raises(InvariantViolation, match="revision"):
            make_bar(revision=-1)


class TestImmutability:
    def test_bar_is_frozen(self) -> None:
        bar = make_bar()
        with pytest.raises(dataclasses.FrozenInstanceError):
            bar.close = Decimal("1")  # type: ignore[misc]

    def test_finalize_returns_a_new_final_bar(self) -> None:
        partial = make_bar(is_final=False)
        final = partial.finalized()
        assert not partial.is_final
        assert final.is_final
        assert final.key == partial.key

    def test_finalizing_a_final_bar_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="final"):
            make_bar(is_final=True).finalized()

    def test_revised_bumps_the_revision(self) -> None:
        original = make_bar(revision=1)
        revised = original.revised(close="103")
        assert revised.revision == 2
        assert revised.close == Decimal("103.00000000")
        assert revised.key == original.key

    def test_a_revision_must_still_satisfy_the_ohlc_invariants(self) -> None:
        # A correction that pushes close above high is a bad correction.
        with pytest.raises(InvariantViolation, match="high"):
            make_bar().revised(close="106")

    def test_revised_rejects_identity_fields(self) -> None:
        with pytest.raises(InvariantViolation, match="identity"):
            make_bar().revised(feed=Feed.SIP)


class TestSession:
    def test_extended_hours_bar_keeps_its_trading_day(self) -> None:
        bar = make_bar(
            session=SessionKind.PRE,
            start_time_utc=SESSION_OPEN - timedelta(hours=2),
            end_time_utc=SESSION_OPEN - timedelta(hours=1),
        )
        assert bar.session is SessionKind.PRE
        assert bar.session_date == make_bar().session_date
