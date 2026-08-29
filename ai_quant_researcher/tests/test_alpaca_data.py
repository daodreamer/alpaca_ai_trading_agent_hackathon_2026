"""The Alpaca bar adapter.

No test here touches the network. The HTTP client is injected, which is the
only reason a paginating adapter with point-in-time semantics is testable at
all: the interesting behaviour is what it does across a page boundary, on an
empty window, and on a bar the market had not finished printing -- none of which
a live endpoint reproduces on demand.

The property this file exists to pin is the last one. Alpaca stamps a daily bar
with the session's *opening* instant, so a naive adapter reports a bar as
knowable hours before it closed. That is look-ahead arriving through the data
layer, where none of the backtester's defences can see it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aqr.data.alpaca import AlpacaProvider, alpaca_timeframe

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 10, tzinfo=UTC)


class _Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    """Records every request and replays a scripted list of responses."""

    def __init__(self, *responses: _Response) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str]) -> _Response:
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if not self._responses:
            raise AssertionError("the adapter asked for more pages than were scripted")
        return self._responses.pop(0)


def _bar(day: int, *, hour: int = 5, close: float = 100.0) -> dict[str, Any]:
    return {
        "t": f"2024-01-{day:02d}T{hour:02d}:00:00Z",
        "o": close - 1.0,
        "h": close + 2.0,
        "l": close - 2.0,
        "c": close,
        "v": 1_000_000,
    }


def _page(bars: list[dict[str, Any]], token: str | None = None) -> _Response:
    return _Response({"bars": {"AAPL": bars}, "next_page_token": token})


def _provider(*responses: _Response, **kwargs: Any) -> tuple[AlpacaProvider, _StubClient]:
    client = _StubClient(*responses)
    provider = AlpacaProvider(
        key_id="PKTEST", secret="s3cret", client=client, sleep=lambda _s: None, **kwargs
    )
    return provider, client


class TestTimeframeMapping:
    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [("1m", "1Min"), ("5m", "5Min"), ("15m", "15Min"), ("1h", "1Hour"), ("1D", "1Day")],
    )
    def test_maps_every_timeframe_the_rest_of_the_system_uses(
        self, ours: str, theirs: str
    ) -> None:
        assert alpaca_timeframe(ours) == theirs

    def test_refuses_a_timeframe_it_cannot_map(self) -> None:
        # Silently substituting a different timeframe would produce a backtest
        # that is internally consistent and answers a question nobody asked.
        with pytest.raises(ValueError, match="4h"):
            alpaca_timeframe("4h")


class TestPagination:
    def test_follows_the_page_token_until_it_is_exhausted(self) -> None:
        provider, client = _provider(
            _page([_bar(2), _bar(3)], token="page-2"),
            _page([_bar(4), _bar(5)], token=None),
        )
        bars = provider.load("AAPL", START, END, "1D")

        assert len(bars) == 4
        assert len(client.calls) == 2
        assert "page_token" not in client.calls[0]["params"]
        assert client.calls[1]["params"]["page_token"] == "page-2"

    def test_a_repeated_page_token_raises_instead_of_looping_forever(self) -> None:
        provider, _ = _provider(
            _page([_bar(2)], token="same"),
            _page([_bar(3)], token="same"),
        )
        with pytest.raises(RuntimeError, match="repeated"):
            provider.load("AAPL", START, END, "1D")


class TestPointInTime:
    def test_a_daily_bar_is_not_knowable_until_its_session_closes(self) -> None:
        """Alpaca stamps a daily bar at the session open. Trusting that stamp as
        the availability time hands the backtester tomorrow's close today."""
        provider, _ = _provider(_page([_bar(2, hour=5)]))
        bars = provider.load("AAPL", START, END, "1D")

        event = int(datetime(2024, 1, 2, 5, tzinfo=UTC).timestamp())
        assert int(bars.event_time[0]) == event
        assert int(bars.available_time[0]) > event

    def test_an_intraday_bar_becomes_available_when_it_closes(self) -> None:
        provider, _ = _provider(
            _Response(
                {
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2024-01-02T14:30:00Z",
                                "o": 1.0,
                                "h": 2.0,
                                "l": 0.5,
                                "c": 1.5,
                                "v": 10,
                            }
                        ]
                    },
                    "next_page_token": None,
                }
            )
        )
        bars = provider.load("AAPL", START, END, "5m")
        gap = int(bars.available_time[0]) - int(bars.event_time[0])
        assert gap == 5 * 60


class TestRequest:
    def test_sends_the_credentials_as_headers_and_never_in_the_url(self) -> None:
        provider, client = _provider(_page([_bar(2)]))
        provider.load("AAPL", START, END, "1D")
        call = client.calls[0]

        assert call["headers"]["APCA-API-KEY-ID"] == "PKTEST"
        assert call["headers"]["APCA-API-SECRET-KEY"] == "s3cret"
        assert "s3cret" not in call["url"]
        assert "s3cret" not in str(call["params"])

    def test_asks_for_adjusted_bars_by_default(self) -> None:
        """An unadjusted 4:1 split is a -75% return the strategy never took."""
        provider, client = _provider(_page([_bar(2)]))
        provider.load("AAPL", START, END, "1D")
        assert client.calls[0]["params"]["adjustment"] == "all"

    def test_reports_which_feed_and_adjustment_produced_the_bars(self) -> None:
        provider, _ = _provider(_page([_bar(2)]))
        assert provider.dataset_version("1D") == "alpaca:sip:all:rth:1D"

    def test_defaults_to_the_consolidated_tape_not_one_venue(self) -> None:
        """IEX is a few percent of the volume, and its daily history has holes
        big enough to swallow a bear market -- 634 days, in the pull that
        prompted this test."""
        provider, client = _provider(_page([_bar(2)]))
        provider.load("AAPL", START, END, "1D")
        assert client.calls[0]["params"]["feed"] == "sip"


class TestFailures:
    def test_an_empty_window_says_so_rather_than_returning_nothing(self) -> None:
        provider, _ = _provider(_Response({"bars": {}, "next_page_token": None}))
        with pytest.raises(ValueError, match="no bars"):
            provider.load("AAPL", START, END, "1D")

    def test_an_http_error_surfaces_the_status_without_the_key(self) -> None:
        provider, _ = _provider(_Response({"message": "forbidden"}, status=403))
        with pytest.raises(RuntimeError) as exc:
            provider.load("AAPL", START, END, "1D")
        assert "403" in str(exc.value)
        assert "s3cret" not in str(exc.value)

    def test_out_of_order_bars_are_sorted_rather_than_rejected(self) -> None:
        # Pages are chronological in practice, but Bars refuses non-increasing
        # timestamps, and a hard failure on a cosmetic ordering quirk would cost
        # a whole pull.
        provider, _ = _provider(_page([_bar(4), _bar(2), _bar(3)]))
        bars = provider.load("AAPL", START, END, "1D")
        assert list(bars.event_time) == sorted(bars.event_time)

    def test_a_duplicate_timestamp_across_pages_is_dropped(self) -> None:
        provider, _ = _provider(
            _page([_bar(2), _bar(3)], token="page-2"),
            _page([_bar(3), _bar(4)], token=None),
        )
        bars = provider.load("AAPL", START, END, "1D")
        assert len(bars) == 3


class TestRegularHours:
    """Alpaca serves extended hours and has no flag to say otherwise.

    A 15-minute SPY pull came back with 59 bars per session where 09:30-16:00
    holds 26, the first stamped 09:00Z (04:00 New York) and the last 23:45Z.
    Those bars are thin, their spreads are a multiple of the session's, and the
    backtester treats every bar alike -- so a strategy tested on them is
    measured against a market that was not open.
    """

    def test_intraday_bars_outside_the_session_are_dropped(self) -> None:
        premarket = {
            "t": "2024-01-02T09:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10,
        }
        regular = {
            "t": "2024-01-02T15:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10,
        }
        after = {
            "t": "2024-01-02T23:45:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10,
        }
        provider, _ = _provider(
            _Response({"bars": {"AAPL": [premarket, regular, after]}, "next_page_token": None})
        )
        bars = provider.load("AAPL", START, END, "15m")
        assert len(bars) == 1
        assert bars.timestamps[0].hour == 15

    def test_extended_hours_can_be_asked_for_deliberately(self) -> None:
        premarket = {
            "t": "2024-01-02T09:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10,
        }
        client = _StubClient(
            _Response({"bars": {"AAPL": [premarket]}, "next_page_token": None})
        )
        provider = AlpacaProvider(
            key_id="PKTEST", secret="s3cret", client=client, regular_hours_only=False
        )
        assert len(provider.load("AAPL", START, END, "15m")) == 1

    def test_daily_bars_are_untouched_by_the_session_filter(self) -> None:
        # A daily bar is stamped at the session open, which is outside 09:30 in
        # New York. Filtering one on its stamp would delete every daily series.
        provider, _ = _provider(_page([_bar(2, hour=5)]))
        assert len(provider.load("AAPL", START, END, "1D")) == 1

    def test_the_session_choice_is_recorded_in_the_dataset_version(self) -> None:
        provider, _ = _provider(_page([_bar(2)]))
        assert "rth" in provider.dataset_version("15m")


class TestRateLimiting:
    """429 is not an error, it is an instruction to wait.

    Building the sealed cache is 682 sequential requests, and treating a rate
    limit as a hard failure left 41 tickers permanently unfetchable no matter how
    often the pull was repeated -- each retry re-hit the limit at the same point.
    A ticker that never arrives is a hole in the universe, and a hole in the
    universe is a different strategy.
    """

    def test_a_rate_limit_is_retried_rather_than_raised(self) -> None:
        provider, client = _provider(
            _Response({}, status=429),
            _page([_bar(2)]),
        )
        bars = provider.load("AAPL", START, END)
        assert len(bars) == 1
        assert len(client.calls) == 2

    def test_the_wait_grows_between_attempts(self) -> None:
        waits: list[float] = []
        client = _StubClient(
            _Response({}, status=429),
            _Response({}, status=429),
            _page([_bar(2)]),
        )
        provider = AlpacaProvider(
            key_id="PKTEST", secret="s3cret", client=client, sleep=waits.append
        )
        provider.load("AAPL", START, END)
        assert waits == sorted(waits) and waits[0] < waits[-1]

    def test_it_gives_up_rather_than_retrying_forever(self) -> None:
        provider, _ = _provider(*[_Response({}, status=429) for _ in range(9)], max_retries=3)
        with pytest.raises(RuntimeError, match="429"):
            provider.load("AAPL", START, END)

    def test_a_server_error_is_retried_too(self) -> None:
        provider, client = _provider(_Response({}, status=503), _page([_bar(2)]))
        assert len(provider.load("AAPL", START, END)) == 1
        assert len(client.calls) == 2

    def test_a_permission_error_is_not_retried(self) -> None:
        """403 means the account cannot have these bars. Waiting does not help,
        and retrying turns one clear failure into a slow one."""
        provider, client = _provider(_Response({}, status=403))
        with pytest.raises(RuntimeError, match="403"):
            provider.load("AAPL", START, END)
        assert len(client.calls) == 1
