"""The IBKR request probe.

A live ``reqHistoricalData`` can fail for at least four unrelated reasons --
``ADJUSTED_LAST`` refusing an explicit end date, a missing market-data
subscription, a duration the bar size does not accept, a timeout that is simply
too short -- and every one of them surfaces as the same pair of messages:

    reqHistoricalData: Timeout for Stock(...)
    Error 366: No historical data query found for ticker id

Guessing between four causes one round trip at a time is the slow way. The probe
asks all of them at once and reports which combinations returned bars, so the
next change is informed by a result rather than by a hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from aqr.data.ibkr import probe_requests


@dataclass
class _Bar:
    date: Any
    open: float = 99.0
    high: float = 101.0
    low: float = 98.0
    close: float = 100.0
    volume: float = 1e6


class _ScriptedIB:
    """Answers only the request shapes it was told to accept."""

    def __init__(self, accept: list[tuple[str, bool]], error: str = "Timeout") -> None:
        self.accept = set(accept)
        self.error = error
        self.asked: list[dict[str, Any]] = []

    def isConnected(self) -> bool:  # noqa: N802
        return True

    def connect(self, *a: Any, **kw: Any) -> None:
        raise AssertionError("the probe must not open its own connection")

    def disconnect(self) -> None:
        pass

    def qualifyContracts(self, contract: Any) -> list[Any]:  # noqa: N802
        return [contract]

    def reqHistoricalData(self, contract: Any, **kw: Any) -> list[_Bar]:  # noqa: N802
        self.asked.append(dict(kw))
        key = (kw["whatToShow"], bool(kw["endDateTime"]))
        if key not in self.accept:
            raise TimeoutError(self.error)
        return [_Bar(date=date(2024, 1, 2))]


END = datetime(2026, 8, 27, tzinfo=UTC)


class TestProbe:
    def test_it_tries_both_adjusted_and_unadjusted_with_and_without_an_end_date(self) -> None:
        ib = _ScriptedIB(accept=[])
        results = probe_requests(ib, "AAPL", end=END)

        shapes = {(r.what_to_show, r.dated) for r in results}
        assert ("ADJUSTED_LAST", True) in shapes
        assert ("ADJUSTED_LAST", False) in shapes
        assert ("TRADES", True) in shapes
        assert ("TRADES", False) in shapes

    def test_a_working_combination_is_reported_as_working(self) -> None:
        # The hypothesis that prompted the probe: ADJUSTED_LAST is documented to
        # need an empty end date, while TRADES accepts one.
        ib = _ScriptedIB(accept=[("ADJUSTED_LAST", False), ("TRADES", True), ("TRADES", False)])
        results = probe_requests(ib, "AAPL", end=END)

        by_shape = {(r.what_to_show, r.dated): r for r in results}
        assert by_shape[("ADJUSTED_LAST", False)].bars == 1
        assert by_shape[("ADJUSTED_LAST", True)].bars == 0
        assert by_shape[("ADJUSTED_LAST", True)].error

    def test_one_failing_shape_does_not_end_the_probe(self) -> None:
        """The whole point is to learn about all four, so the first timeout
        cannot be allowed to stop the other three."""
        ib = _ScriptedIB(accept=[("TRADES", False)])
        results = probe_requests(ib, "AAPL", end=END)
        assert len(results) >= 4
        assert any(r.bars > 0 for r in results)

    def test_the_error_is_recorded_against_the_shape_that_caused_it(self) -> None:
        ib = _ScriptedIB(accept=[], error="No security definition")
        results = probe_requests(ib, "AAPL", end=END)
        assert all("No security definition" in (r.error or "") for r in results)

    def test_it_never_places_or_reads_an_order(self) -> None:
        # A probe that touched account state would be a different tool with a
        # different risk profile. The scripted handle raises on anything else.
        ib = _ScriptedIB(accept=[("TRADES", False)])
        probe_requests(ib, "AAPL", end=END)
        for asked in ib.asked:
            assert set(asked) <= {
                "endDateTime",
                "durationStr",
                "barSizeSetting",
                "whatToShow",
                "useRTH",
                "formatDate",
                "timeout",
            }

    def test_a_summary_line_names_the_shape(self) -> None:
        ib = _ScriptedIB(accept=[("TRADES", True)])
        results = probe_requests(ib, "AAPL", end=END)
        text = "\n".join(str(r) for r in results)
        assert "ADJUSTED_LAST" in text and "TRADES" in text
        assert "end date" in text


class TestDurationProbe:
    """How much history one request will actually answer.

    IBKR's published step table says a "1 day" bar size accepts "1 Y". Practice
    is widely reported to be more generous. The difference decides whether
    ADJUSTED_LAST -- which the live probe confirmed refuses an explicit end date,
    and therefore cannot be chunked -- can reach past one year at all.

    Not a question worth answering from documentation when one request answers
    it for this account, on this build, today.
    """

    def test_it_asks_for_progressively_longer_windows(self) -> None:
        from aqr.data.ibkr import probe_durations

        ib = _ScriptedIB(accept=[("ADJUSTED_LAST", False)])
        results = probe_durations(ib, "AAPL")
        asked = [r["durationStr"] for r in ib.asked]
        assert "1 Y" in asked
        assert any(d.startswith("20 ") or d.startswith("30 ") for d in asked)
        assert len(results) == len(asked)

    def test_it_reports_the_bar_count_for_each_window(self) -> None:
        from aqr.data.ibkr import probe_durations

        ib = _ScriptedIB(accept=[("ADJUSTED_LAST", False)])
        results = probe_durations(ib, "AAPL")
        assert all(r.bars >= 0 for r in results)

    def test_it_always_sends_an_empty_end_date(self) -> None:
        # The shape ADJUSTED_LAST requires. Sending a dated request here would
        # measure the wrong thing.
        from aqr.data.ibkr import probe_durations

        ib = _ScriptedIB(accept=[("ADJUSTED_LAST", False)])
        probe_durations(ib, "AAPL")
        assert all(not r["endDateTime"] for r in ib.asked)

    def test_a_rejected_window_does_not_stop_the_rest(self) -> None:
        from aqr.data.ibkr import probe_durations

        ib = _ScriptedIB(accept=[])
        results = probe_durations(ib, "AAPL")
        assert len(results) >= 4
        assert all(r.error for r in results)
