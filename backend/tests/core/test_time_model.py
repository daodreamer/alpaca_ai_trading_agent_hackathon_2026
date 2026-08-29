"""ADR 0004 — timeframes, sessions, and the tz-aware-UTC rule."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.time_model import SessionKind, SessionWindow, Timeframe, ensure_utc


class TestTimeframe:
    def test_canonical_codes_match_the_spec(self) -> None:
        assert [tf.code for tf in Timeframe] == [
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "4h",
            "1D",
            "1W",
        ]

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("1m", timedelta(minutes=1)),
            ("15m", timedelta(minutes=15)),
            ("1h", timedelta(hours=1)),
            ("4h", timedelta(hours=4)),
            ("1D", timedelta(days=1)),
            ("1W", timedelta(days=7)),
        ],
    )
    def test_nominal_duration(self, code: str, expected: timedelta) -> None:
        assert Timeframe.from_code(code).nominal_duration == expected

    def test_from_code_rejects_unknown(self) -> None:
        with pytest.raises(InvariantViolation, match="2h"):
            Timeframe.from_code("2h")

    def test_from_code_is_case_sensitive_for_1d_vs_1m(self) -> None:
        # "1M" (month) is not a canonical timeframe and must not resolve to 1m.
        with pytest.raises(InvariantViolation):
            Timeframe.from_code("1M")

    @pytest.mark.parametrize(
        ("code", "intraday"),
        [("1m", True), ("4h", True), ("1D", False), ("1W", False)],
    )
    def test_is_intraday(self, code: str, intraday: bool) -> None:
        assert Timeframe.from_code(code).is_intraday is intraday


class TestEnsureUtc:
    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            ensure_utc(datetime(2026, 8, 13, 13, 30), field="start_time_utc")  # noqa: DTZ001

    def test_normalizes_other_zones_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-4))
        result = ensure_utc(datetime(2026, 8, 13, 9, 30, tzinfo=eastern), field="t")
        assert result == datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
        assert result.tzinfo is UTC

    def test_passes_through_utc(self) -> None:
        moment = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
        assert ensure_utc(moment, field="t") is moment


class TestSessionWindow:
    def test_requires_open_before_close(self) -> None:
        moment = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
        with pytest.raises(InvariantViolation):
            SessionWindow(
                kind=SessionKind.REGULAR,
                open_utc=moment,
                close_utc=moment,
                session_date=date(2026, 8, 13),
            )

    def test_contains_is_half_open(self) -> None:
        window = SessionWindow(
            kind=SessionKind.REGULAR,
            open_utc=datetime(2026, 8, 13, 13, 30, tzinfo=UTC),
            close_utc=datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
            session_date=date(2026, 8, 13),
        )
        assert window.contains(datetime(2026, 8, 13, 13, 30, tzinfo=UTC))
        assert window.contains(datetime(2026, 8, 13, 19, 59, 59, tzinfo=UTC))
        assert not window.contains(datetime(2026, 8, 13, 20, 0, tzinfo=UTC))
        assert not window.contains(datetime(2026, 8, 13, 13, 29, tzinfo=UTC))

    def test_duration(self) -> None:
        window = SessionWindow(
            kind=SessionKind.REGULAR,
            open_utc=datetime(2026, 8, 13, 13, 30, tzinfo=UTC),
            close_utc=datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
            session_date=date(2026, 8, 13),
        )
        assert window.duration == timedelta(hours=6, minutes=30)
