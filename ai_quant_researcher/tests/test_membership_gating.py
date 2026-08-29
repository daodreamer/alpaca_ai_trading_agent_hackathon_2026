"""Trading only what was in the index at the time.

Truncating each symbol's bars to its membership intervals would get most of this
for free: a name with no bars before it joined cannot be ranked, held or
benchmarked, and the delisting exit already fires when the bars stop. That
handles a name that joins once and either stays or leaves once, which is the
overwhelming majority.

It does not handle re-entry, and re-entry is where the shortcut turns into the
bias it was meant to remove. A company dropped from the index in 2019 and read-
mitted in 2021 keeps trading throughout. Its bars have no gap, so bar-presence
says "member" for the whole span, and the book would hold it during two years
when it was not in the universe -- picked, necessarily, because we know it came
back.

So membership is passed in explicitly and intersected with bar presence, and the
two reasons a holding becomes untradable are unified:

* its bars stop      -> `delisted`
* it leaves the index -> `left_universe`

Both are forced exits at the last price at which the position could have been
sold. Neither is a decision the strategy made, so both are reported with their
own reason rather than as a rebalance.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.portfolio import _carry_forward, _membership_mask, run_portfolio
from aqr.data.bars import Bars
from aqr.data.universes import Interval, Membership, PointInTimeUniverse
from aqr.dsl.schema import spec_from_dict

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
N = 400
START = datetime(2016, 1, 4, tzinfo=UTC)
T0 = int(START.timestamp())


def _bars(symbol: str, drift: float, n: int = N) -> Bars:
    t = np.arange(T0, T0 + n * 86_400, 86_400, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1e6),
    )


def _data() -> dict[str, Bars]:
    # AAA is the strongest, so the ranking wants to hold it. That is the point:
    # the gate has to override the ranking, not merely agree with it.
    return {s: _bars(s, 0.0009 - 0.0002 * i) for i, s in enumerate(SYMBOLS)}


def _day(step: int) -> date:
    return datetime.fromtimestamp(T0 + step * 86_400, tz=UTC).date()


def _spec(**over: object):
    body: dict[str, object] = {
        "name": "xs_probe",
        "mode": "portfolio",
        "rank_by": "roc(20)",
        "hold": 2,
        "rebalance_every": 10,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "sleeve": {"budget": 0.20, "idle": "benchmark"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def _membership(aaa_intervals: tuple[Interval, ...]) -> PointInTimeUniverse:
    members = [Membership("AAA", "AAA", aaa_intervals)]
    members += [
        Membership(s, s, (Interval(_day(0), None),)) for s in SYMBOLS if s != "AAA"
    ]
    return PointInTimeUniverse(
        name="probe",
        source="test",
        window=(_day(0), _day(N - 1)),
        members=tuple(members),
    )


def _run(membership: PointInTimeUniverse | None = None, **over: object):
    return run_portfolio(_spec(**over), _data(), CONFIG, membership=membership)


# --------------------------------------------------------------------------


def test_without_membership_nothing_changes() -> None:
    """Every result in the registry was measured without one. Passing no
    membership must behave exactly as before."""
    a = _run()
    b = run_portfolio(_spec(), _data(), CONFIG)
    assert np.array_equal(a.equity, b.equity)


def test_a_name_is_not_held_before_it_joins() -> None:
    """Bars exist before a company enters an index. Holding it then is buying on
    the strength of a listing decision nobody had made yet."""
    joined = 200
    result = _run(_membership((Interval(_day(joined), None),)))
    for step in range(30, joined):
        assert "AAA" not in result.weights_at(step), f"held at step {step}, before joining"


def test_it_is_held_after_it_joins() -> None:
    """The gate must not simply exclude the name forever -- that would be a
    different bias with the same symptoms."""
    joined = 200
    result = _run(_membership((Interval(_day(joined), None),)))
    held = [s for s in range(joined + 3, N - 1) if "AAA" in result.weights_at(s)]
    assert held, "AAA was never held even after joining"


def test_leaving_the_index_forces_an_exit() -> None:
    left = 200
    result = _run(_membership((Interval(_day(0), _day(left)),)))
    exits = [t for t in result.trades if t.symbol == "AAA"]
    assert exits
    assert any(t.exit_reason == "left_universe" for t in exits), [t.exit_reason for t in exits]


def test_the_forced_exit_is_not_reported_as_a_rebalance() -> None:
    """The strategy did not choose it. Counting it as a rebalance would put a
    decision nobody made into the turnover and the exit-reason breakdown."""
    result = _run(_membership((Interval(_day(0), _day(200)),)))
    forced = [t for t in result.trades if t.exit_reason == "left_universe"]
    assert forced
    assert all(t.symbol == "AAA" for t in forced)


def test_re_entry_leaves_the_gap_empty() -> None:
    """The case bar-presence cannot see: the company keeps trading throughout, so
    its bars have no hole. Only the membership table knows it was out."""
    out, back = 150, 300
    result = _run(_membership((Interval(_day(0), _day(out)), Interval(_day(back), None))))
    for step in range(out + 2, back):
        assert "AAA" not in result.weights_at(step), f"held at step {step}, while out"
    later = [s for s in range(back + 3, N - 1) if "AAA" in result.weights_at(s)]
    assert later, "AAA was never held again after rejoining"


def test_a_non_member_is_not_in_the_peer_set() -> None:
    """The deeper contamination. `rs_rank` against a peer set containing names
    that were not in the index is not a number a researcher could have computed
    at the time, whichever name is being ranked."""
    joined = 200
    with_gate = _run(_membership((Interval(_day(joined), None),)), rank_by="rs_rank(20)")
    without = _run(rank_by="rs_rank(20)")
    early = 60
    assert with_gate.core_weights_at(early) != without.core_weights_at(early)


def test_the_benchmark_holds_only_members() -> None:
    """Otherwise the strategy is measured against an index that owned something
    nobody could have owned."""
    joined = 200
    gated = _run(_membership((Interval(_day(joined), None),)))
    ungated = _run()
    assert not np.allclose(gated.benchmark_equity, ungated.benchmark_equity)


class TestTheMaskIsAShortcutForAskingEveryDay:
    """The mask is filled by binary search over the session calendar rather than
    by asking each session whether it is inside an interval.

    That was worth doing -- asking per session cost 1.5 million Python calls per
    backtest -- but a range fill is a different computation from a predicate,
    and the two agree only if the half-open convention survives the translation.
    These pin that they agree, on the cases where a boundary error hides: the
    first and last day of a span, a re-entry, and a name the table never heard
    of.
    """

    @staticmethod
    def _calendar(days: list[date]) -> np.ndarray:
        return np.array([d.toordinal() for d in days], dtype=np.int64)

    def test_it_agrees_with_member_on_over_every_session(self) -> None:
        table = PointInTimeUniverse(
            name="t",
            source="test",
            window=(date(2020, 1, 1), date(2020, 3, 1)),
            members=(
                Membership(
                    ticker="RE",
                    name="re-entrant",
                    intervals=(
                        Interval(date(2020, 1, 10), date(2020, 1, 20)),
                        Interval(date(2020, 2, 5), None),
                    ),
                ),
            ),
        )
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
        mask = _membership_mask(table, "RE", self._calendar(days))
        expected = np.array([table.by_ticker("RE").member_on(d) for d in days])
        np.testing.assert_array_equal(mask, expected)

    def test_the_interval_is_half_open_at_both_ends(self) -> None:
        """Start is a member day, end is not. A closed end would keep a name for
        one session after it left, on the session its removal was effective."""
        table = PointInTimeUniverse(
            name="t",
            source="test",
            window=(date(2020, 1, 1), date(2020, 2, 1)),
            members=(
                Membership(
                    ticker="X",
                    name="x",
                    intervals=(Interval(date(2020, 1, 10), date(2020, 1, 20)),),
                ),
            ),
        )
        days = [date(2020, 1, 9), date(2020, 1, 10), date(2020, 1, 19), date(2020, 1, 20)]
        mask = _membership_mask(table, "X", self._calendar(days))
        assert list(mask) == [False, True, True, False]

    def test_a_ticker_the_table_never_heard_of_is_never_a_member(self) -> None:
        table = PointInTimeUniverse(
            name="t",
            source="test",
            window=(date(2020, 1, 1), date(2020, 2, 1)),
            members=(
                Membership("X", "x", (Interval(date(2020, 1, 1), None),)),
            ),
        )
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(5)]
        assert not _membership_mask(table, "NOPE", self._calendar(days)).any()


class TestCarryingThePriceForward:
    """A held name with no bar today is marked at its last known close.

    That answer is now precomputed per symbol instead of rescanning the
    symbol's history on every mark. The rescan was quadratic in the run length,
    but it was also the definition, so these pin that the array form gives the
    same answer the rescan did -- including before the first bar, where there is
    no close to carry.
    """

    @staticmethod
    def _rescan(closes: np.ndarray, step: int, opens: np.ndarray) -> float:
        """The per-step form this replaced, kept here as the specification."""
        value = closes[step]
        if not np.isnan(value):
            return float(value)
        prior = closes[: step + 1]
        known = prior[~np.isnan(prior)]
        if known.size:
            return float(known[-1])
        value = opens[step]
        return float(value) if not np.isnan(value) else 0.0

    def test_it_matches_the_rescan_it_replaced(self) -> None:
        nan = float("nan")
        closes = np.array([nan, nan, 10.0, nan, nan, 12.0, nan, 11.0])
        opens = np.array([nan, 9.5, 9.9, 10.1, 10.2, 11.8, 12.1, 10.9])
        carried = _carry_forward(closes, opens)
        expected = [self._rescan(closes, i, opens) for i in range(closes.size)]
        np.testing.assert_array_equal(carried, np.array(expected))

    def test_before_the_first_close_it_falls_back_to_the_open_then_to_zero(self) -> None:
        nan = float("nan")
        carried = _carry_forward(
            np.array([nan, nan, 10.0]), np.array([nan, 9.5, 9.9])
        )
        assert list(carried) == [0.0, 9.5, 10.0]

    def test_a_gap_is_marked_at_the_last_close_not_at_zero(self) -> None:
        nan = float("nan")
        carried = _carry_forward(np.array([10.0, nan, nan]), np.array([9.9, nan, nan]))
        assert list(carried) == [10.0, 10.0, 10.0]
