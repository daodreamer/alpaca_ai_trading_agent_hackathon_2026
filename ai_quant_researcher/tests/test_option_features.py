"""Option features for the DSL — specs/10-options-research.md D6, test plan 4-5.

Two properties matter more than the arithmetic, and most of this file is about
them rather than about any one formula:

**No look-ahead.** A feature's value at bar ``i`` may only be a function of
data timestamped at or before bar ``i``. ``test_truncating_*`` asserts this
directly, by removing every row dated after a cutoff and checking that nothing
at or before the cutoff moved -- the same causality argument
``test_causality.py`` makes for the bar registry, made here for the option one.

**A missing feature is `NaN`, and `NaN` is `False`, never an exception.** D6's
forward-fill is bounded at 5 calendar days precisely so a rule cannot silently
evaluate on IV that is a fortnight old; the boundary is asserted to the day.

The chain-based fixtures reuse ``build_world``'s ``chain_row`` helper from
``test_option_engine.py`` rather than reinventing a quote-row builder.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.dsl.expr import Compare, ParseError, evaluate, parse
from aqr.features.engine import FeatureKey
from aqr.options.chain import ChainIndex
from aqr.options.engine import OptionBacktestConfig, run_option_backtest
from aqr.options.features import (
    MAX_FORWARD_FILL_DAYS,
    OptionFeatureFrame,
    VolatilityHistory,
    resolve_entry_feature,
)
from aqr.options.spec import Anchor, Cadence, DteTarget, OptionSizing, OptionSpec, StructureSpec
from tests.test_option_engine import chain_row, make_underlying

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def daily_bars(n: int, start: date = date(2023, 1, 2)) -> Bars:
    """Calendar-consecutive days, not trading days.

    Forward-fill here is bounded in *calendar* days (D6), so a fixture with no
    weekend gaps makes the day-count arithmetic in each assertion checkable by
    eye -- ``bar i`` is unambiguously ``start + i`` days.
    """
    days = [start + timedelta(days=i) for i in range(n)]
    return make_underlying({}, days)


def vol_row(
    session: date,
    *,
    iv_current: str = "0.20",
    iv_year_high: str = "0.40",
    iv_year_low: str = "0.10",
    iv_week_ago: str = "0.19",
    iv_month_ago: str = "0.18",
    hv_current: str = "0.15",
) -> dict[str, str]:
    """One ``volatility_history`` row, vendor column names, vendor blank convention."""
    return {
        "date": session.isoformat(),
        "act_symbol": "SPY",
        "hv_current": hv_current,
        "hv_week_ago": "",
        "hv_month_ago": "",
        "hv_year_high": "",
        "hv_year_high_date": "",
        "hv_year_low": "",
        "hv_year_low_date": "",
        "iv_current": iv_current,
        "iv_week_ago": iv_week_ago,
        "iv_month_ago": iv_month_ago,
        "iv_year_high": iv_year_high,
        "iv_year_high_date": "",
        "iv_year_low": iv_year_low,
        "iv_year_low_date": "",
    }


def _quote_row(
    session: date, expiry: date, strike: float, right: str, *, delta: float, iv: float
) -> dict[str, str]:
    row = chain_row(session, expiry, strike, right, bid=1.0, ask=1.0, delta=delta)
    row["vol"] = f"{iv:.4f}"
    return row


def term_chain_rows(session: date) -> list[dict[str, str]]:
    """One session's ladder at ~14, ~28 and ~49 DTE, with ATM (0.50-delta) and
    25-delta legs on both sides -- everything ``atm_iv``, ``term_slope`` and
    ``skew_25d`` need from one session."""
    rows: list[dict[str, str]] = []
    for dte, atm_iv, put25_iv, call25_iv in (
        (14, 0.16, 0.20, 0.14),
        (28, 0.18, 0.24, 0.15),
        (49, 0.21, 0.27, 0.17),
    ):
        expiry = session + timedelta(days=dte)
        rows.append(_quote_row(session, expiry, 400.0, "call", delta=0.50, iv=atm_iv))
        rows.append(_quote_row(session, expiry, 380.0, "put", delta=-0.25, iv=put25_iv))
        rows.append(_quote_row(session, expiry, 420.0, "call", delta=0.25, iv=call25_iv))
    return rows


# --------------------------------------------------------------------------- #
# The combined feature table (D5: "only the feature table changes")
# --------------------------------------------------------------------------- #


class TestCombinedResolution:
    def test_option_features_resolve(self) -> None:
        for name in ("iv_rank", "iv_hv_spread", "iv_current", "hv_current", "term_slope",
                      "skew_25d"):
            assert resolve_entry_feature(name).arity == 0
        assert resolve_entry_feature("iv_change").arity == 1
        assert resolve_entry_feature("atm_iv").arity == 1

    def test_bar_features_still_resolve_through_the_combined_table(self) -> None:
        assert resolve_entry_feature("close").arity == 0
        assert resolve_entry_feature("sma").arity == 1

    def test_unknown_feature_raises_naming_the_closest_match(self) -> None:
        with pytest.raises(KeyError, match="iv_rank"):
            resolve_entry_feature("iv_ran")

    def test_the_equity_dsl_parse_is_unaffected(self) -> None:
        """The default `parse()` -- what every equity caller uses -- must not
        gain `iv_rank` just because this module exists somewhere in the import
        graph. Only a caller that opts into `resolve_entry_feature` sees it."""
        with pytest.raises(ParseError, match="unknown feature"):
            parse("iv_rank() > 50")

    def test_an_entry_expression_can_mix_both_vocabularies(self) -> None:
        node = parse("iv_rank() > 50 and close > sma(200)", resolve_feature=resolve_entry_feature)
        assert isinstance(node, object)  # parses at all; evaluated below


# --------------------------------------------------------------------------- #
# volatility_history features: arithmetic
# --------------------------------------------------------------------------- #


class TestVolatilityFeatures:
    def test_iv_rank_is_the_normalised_position_between_the_year_extremes(self) -> None:
        bars = daily_bars(3)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.25", iv_year_high="0.45", iv_year_low="0.05")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        value = frame.get(FeatureKey("iv_rank"))[0]
        # (0.25 - 0.05) / (0.45 - 0.05) * 100 = 50.0
        assert value == pytest.approx(50.0)

    def test_iv_hv_spread_is_current_iv_minus_current_hv(self) -> None:
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.22", hv_current="0.14")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        assert frame.get(FeatureKey("iv_hv_spread"))[0] == pytest.approx(0.08)

    def test_iv_current_and_hv_current_are_the_raw_levels(self) -> None:
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.31", hv_current="0.19")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        assert frame.get(FeatureKey("iv_current"))[0] == pytest.approx(0.31)
        assert frame.get(FeatureKey("hv_current"))[0] == pytest.approx(0.19)

    def test_iv_change_five_reads_the_week_ago_column(self) -> None:
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.25", iv_week_ago="0.20")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        assert frame.get(FeatureKey("iv_change", (5.0,)))[0] == pytest.approx(0.05)

    def test_iv_change_twenty_one_reads_the_month_ago_column(self) -> None:
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.25", iv_month_ago="0.15")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        assert frame.get(FeatureKey("iv_change", (21.0,)))[0] == pytest.approx(0.10)

    def test_iv_change_refuses_an_n_the_table_cannot_answer(self) -> None:
        """No interpolation: the vendor carries exactly two lookbacks."""
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows([vol_row(date(2023, 1, 2))])
        frame = OptionFeatureFrame(bars, volatility=vol)
        with pytest.raises(ValueError, match="5.*21|21.*5"):
            frame.get(FeatureKey("iv_change", (10.0,)))

    def test_blank_extremes_make_iv_rank_nan_but_not_iv_current(self) -> None:
        """15 vendor rows carry blank year extremes (D0). That must not be
        coerced to 0 -- a blank low read as 0 would report SPY at the top of a
        range that was never measured."""
        bars = daily_bars(2)
        vol = VolatilityHistory.from_rows(
            [vol_row(date(2023, 1, 2), iv_current="0.20", iv_year_high="", iv_year_low="")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        assert np.isnan(frame.get(FeatureKey("iv_rank"))[0])
        assert frame.get(FeatureKey("iv_current"))[0] == pytest.approx(0.20)

    def test_a_feature_that_needs_volatility_data_errors_loudly_without_it(self) -> None:
        """Missing the whole table is a wiring bug, not a quiet NaN -- the same
        distinction `FeatureFrame.get` draws for a cross-sectional feature with
        no universe."""
        frame = OptionFeatureFrame(daily_bars(3))
        with pytest.raises(ValueError, match="volatility"):
            frame.get(FeatureKey("iv_rank"))


# --------------------------------------------------------------------------- #
# The forward-fill bound (D6)
# --------------------------------------------------------------------------- #


class TestForwardFillBound:
    def test_a_value_is_carried_forward_up_to_the_bound(self) -> None:
        start = date(2023, 1, 2)
        bars = daily_bars(MAX_FORWARD_FILL_DAYS + 5, start=start)
        vol = VolatilityHistory.from_rows(
            [vol_row(start, iv_current="0.30", iv_year_high="0.50", iv_year_low="0.10")]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        values = frame.get(FeatureKey("iv_rank"))
        for offset in range(MAX_FORWARD_FILL_DAYS + 1):  # 0..5 inclusive: within the bound
            assert values[offset] == pytest.approx(50.0), offset
        for offset in range(MAX_FORWARD_FILL_DAYS + 1, MAX_FORWARD_FILL_DAYS + 5):
            assert np.isnan(values[offset]), offset

    def test_the_2019_weekly_era_does_not_silently_carry_a_fortnight(self) -> None:
        """The failure D6 names by name: an unbounded fill would answer a rule
        with IV up to 13 days stale between two Saturday snapshots a week apart
        two weeks apart."""
        start = date(2019, 2, 9)
        bars = daily_bars(20, start=start)
        two_weeks = start + timedelta(days=14)
        vol = VolatilityHistory.from_rows(
            [
                vol_row(start, iv_current="0.10", iv_year_high="0.20", iv_year_low="0.0"),
                vol_row(two_weeks, iv_current="0.18", iv_year_high="0.20", iv_year_low="0.0"),
            ]
        )
        frame = OptionFeatureFrame(bars, volatility=vol)
        values = frame.get(FeatureKey("iv_rank"))
        assert np.isnan(values[7])  # the fortnight's midpoint: too stale from either row
        assert np.isnan(values[13])  # the day before the second row: still too stale
        assert values[14] == pytest.approx(90.0)  # the second row lands exactly here


# --------------------------------------------------------------------------- #
# option_chain features: atm_iv, term_slope, skew_25d
# --------------------------------------------------------------------------- #


class TestChainFeatures:
    def test_atm_iv_reads_the_half_delta_contract_in_the_nearest_bucket(self) -> None:
        session = date(2023, 1, 2)
        bars = daily_bars(2, start=session)
        chain = ChainIndex.from_rows(term_chain_rows(session))
        frame = OptionFeatureFrame(bars, chain=chain)
        assert frame.get(FeatureKey("atm_iv", (14.0,)))[0] == pytest.approx(0.16)
        assert frame.get(FeatureKey("atm_iv", (28.0,)))[0] == pytest.approx(0.18)
        assert frame.get(FeatureKey("atm_iv", (49.0,)))[0] == pytest.approx(0.21)

    def test_term_slope_is_49_dte_atm_minus_14_dte_atm(self) -> None:
        session = date(2023, 1, 2)
        bars = daily_bars(2, start=session)
        chain = ChainIndex.from_rows(term_chain_rows(session))
        frame = OptionFeatureFrame(bars, chain=chain)
        assert frame.get(FeatureKey("term_slope"))[0] == pytest.approx(0.21 - 0.16)

    def test_skew_25d_is_put_iv_minus_call_iv_at_25_delta(self) -> None:
        session = date(2023, 1, 2)
        bars = daily_bars(2, start=session)
        chain = ChainIndex.from_rows(term_chain_rows(session))
        frame = OptionFeatureFrame(bars, chain=chain)
        # ~28 DTE bucket: put 0.24, call 0.15
        assert frame.get(FeatureKey("skew_25d"))[0] == pytest.approx(0.24 - 0.15)

    def test_a_session_missing_the_named_leg_is_nan_not_a_crash(self) -> None:
        session = date(2023, 1, 2)
        bars = daily_bars(2, start=session)
        expiry = session + timedelta(days=28)
        rows = [chain_row(session, expiry, 400.0, "call", bid=1.0, ask=1.0, delta=0.50)]
        chain = ChainIndex.from_rows(rows)
        frame = OptionFeatureFrame(bars, chain=chain)
        assert np.isnan(frame.get(FeatureKey("skew_25d"))[0])  # no 25-delta legs at all

    def test_a_feature_that_needs_the_chain_errors_loudly_without_it(self) -> None:
        frame = OptionFeatureFrame(daily_bars(3))
        with pytest.raises(ValueError, match="chain"):
            frame.get(FeatureKey("atm_iv", (28.0,)))


# --------------------------------------------------------------------------- #
# No look-ahead (test plan 4)
# --------------------------------------------------------------------------- #


class TestNoLookAhead:
    def test_truncating_volatility_rows_after_a_cutoff_does_not_move_earlier_bars(self) -> None:
        start = date(2023, 1, 2)
        bars = daily_bars(20, start=start)
        extremes = {"iv_year_high": "0.30", "iv_year_low": "0.00"}
        rows = [
            vol_row(start, iv_current="0.10", **extremes),
            vol_row(start + timedelta(days=5), iv_current="0.15", **extremes),
            vol_row(start + timedelta(days=10), iv_current="0.20", **extremes),
            vol_row(start + timedelta(days=15), iv_current="0.29", **extremes),
        ]
        full = OptionFeatureFrame(bars, volatility=VolatilityHistory.from_rows(rows))
        cutoff = start + timedelta(days=12)
        truncated = OptionFeatureFrame(
            bars, volatility=VolatilityHistory.from_rows(rows, before=cutoff)
        )
        full_values = full.get(FeatureKey("iv_rank"))
        truncated_values = truncated.get(FeatureKey("iv_rank"))
        cutoff_index = (cutoff - start).days
        for i in range(cutoff_index):
            left, right = full_values[i], truncated_values[i]
            assert (np.isnan(left) and np.isnan(right)) or left == pytest.approx(right), i

    def test_truncating_chain_sessions_after_a_cutoff_does_not_move_earlier_bars(self) -> None:
        start = date(2023, 1, 2)
        bars = daily_bars(20, start=start)
        rows: list[dict[str, str]] = []
        for offset in (0, 5, 10, 15):
            rows += term_chain_rows(start + timedelta(days=offset))
        full = OptionFeatureFrame(bars, chain=ChainIndex.from_rows(rows))
        cutoff = start + timedelta(days=12)
        truncated = OptionFeatureFrame(bars, chain=ChainIndex.from_rows(rows, before=cutoff))
        full_values = full.get(FeatureKey("term_slope"))
        truncated_values = truncated.get(FeatureKey("term_slope"))
        cutoff_index = (cutoff - start).days
        for i in range(cutoff_index):
            left, right = full_values[i], truncated_values[i]
            assert (np.isnan(left) and np.isnan(right)) or left == pytest.approx(right), i


# --------------------------------------------------------------------------- #
# NaN propagates to False, never to a crash
# --------------------------------------------------------------------------- #


def test_a_nan_feature_makes_a_comparison_false_not_an_exception() -> None:
    start = date(2023, 1, 2)
    bars = daily_bars(10, start=start)
    # No volatility data at all for bars 0..4; a row lands on bar 5.
    rich_day = start + timedelta(days=5)
    vol = VolatilityHistory.from_rows(
        [vol_row(rich_day, iv_current="0.60", iv_year_high="0.60", iv_year_low="0.00")]
    )
    frame = OptionFeatureFrame(bars, volatility=vol)
    node = parse("iv_rank() > 50", resolve_feature=resolve_entry_feature)
    assert isinstance(node, Compare)
    mask = evaluate(node, frame)
    assert mask.dtype == bool
    assert not mask[:5].any(), "no data yet: must be False, not raise"
    assert mask[5]


# --------------------------------------------------------------------------- #
# Integration: the engine can evaluate a mixed entry expression
# --------------------------------------------------------------------------- #


def _mixed_spec(**overrides: object) -> OptionSpec:
    fields: dict[str, object] = {
        "name": "test_iv_rank_gate",
        "underlying": "SPY",
        "entry": "iv_rank() > 50",
        "structure": StructureSpec(
            type="put_credit_spread",
            dte=DteTarget(target=28, tolerance=10),
            anchor=Anchor(delta=0.16, tolerance=0.06),
            width_points=10.0,
        ),
        "sizing": OptionSizing(risk_per_trade=0.01, max_concurrent=5),
        "cadence": Cadence(min_sessions_between_entries=1),
    }
    fields.update(overrides)
    return OptionSpec(**fields)  # type: ignore[arg-type]


def test_run_option_backtest_gates_entries_on_iv_rank() -> None:
    """Two adjacent sessions; only one has a rich IV rank *at its decision
    bar* (D2: the bar before the fill, not the fill session itself). Proof the
    engine actually reads `volatility=` and folds it into the same signal
    array as `close`, without breaking the one-bar decision-to-fill offset.
    """
    from tests.test_option_engine import build_world

    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(80)]
    days = [d for d in days if d.weekday() < 5]
    rich, poor = days[5], days[6]
    chain, bars = build_world(sessions=[rich, poor], days=days)
    # rich's decision bar is days[4]; poor's decision bar is days[5] (== rich,
    # a date that happens to also be a session, which is irrelevant here --
    # only its role as a *decision bar* for `poor` matters). An earlier row at
    # days[0] keeps `iv_rank`'s observed warm-up at 1 bar rather than 5: like
    # `sma`, the *first* valid bar of a feature is conservatively excluded
    # from trading (FeatureFrame.warmup's "+1"), and without this row that
    # exclusion would swallow days[4] itself.
    vol_rows = [
        vol_row(days[0], iv_current="0.10", iv_year_high="0.60", iv_year_low="0.00"),
        vol_row(days[4], iv_current="0.55", iv_year_high="0.60", iv_year_low="0.00"),
        vol_row(days[5], iv_current="0.05", iv_year_high="0.60", iv_year_low="0.00"),
    ]
    volatility = VolatilityHistory.from_rows(vol_rows)
    result = run_option_backtest(
        _mixed_spec(), chain, bars, OptionBacktestConfig(), volatility=volatility
    )
    assert len(result.option_trades) == 1
    assert result.option_trades[0].entry_session == rich


def test_run_option_backtest_still_works_with_a_bar_only_entry() -> None:
    """The equity path through the same engine must be untouched: an entry
    that never mentions an option feature needs no `volatility=` at all."""
    from tests.test_option_engine import build_world, credit_spread_spec

    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(80)]
    days = [d for d in days if d.weekday() < 5]
    chain, bars = build_world(sessions=days[5:6], days=days)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    assert len(result.option_trades) == 1


# --------------------------------------------------------------------------- #
# Against the real cache (specs/10 D0) -- skipped if it is not pulled
# --------------------------------------------------------------------------- #
#
# The unit tests above prove the arithmetic on numbers a reader can check by
# hand. This section runs the same code against the real SPY cache, which is
# what actually finds a bug -- a coverage gap, a tolerance too tight, a
# forward-fill edge nobody hand-built a fixture for. Skipped rather than
# failed when the cache is absent, matching test_option_cache_claims.py: the
# cache is gigabytes and is not in git.

ROOT = Path(__file__).resolve().parents[1]
_CHAIN = ROOT / "data-options" / "option_chain" / "SPY.csv"
_VOLATILITY = ROOT / "data-options" / "volatility_history" / "SPY.csv"
_UNDERLYING = ROOT / "data-options-underlying" / "1D" / "SPY.csv"

_cache_present = _CHAIN.exists() and _VOLATILITY.exists() and _UNDERLYING.exists()


def _real_bars() -> Bars:
    with _UNDERLYING.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stamps = np.array(
        [int(datetime.fromisoformat(r["timestamp"]).astimezone(UTC).timestamp()) for r in rows],
        dtype=np.int64,
    )
    close = np.array([float(r["close"]) for r in rows], dtype=np.float64)
    return Bars(
        symbol="SPY",
        timeframe="1D",
        event_time=stamps,
        open=np.array([float(r["open"]) for r in rows], dtype=np.float64),
        high=np.array([float(r["high"]) for r in rows], dtype=np.float64),
        low=np.array([float(r["low"]) for r in rows], dtype=np.float64),
        close=close,
        volume=np.array([float(r["volume"]) for r in rows], dtype=np.float64),
    )


def _real_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def real_frame() -> OptionFeatureFrame:
    if not _cache_present:
        pytest.skip("no option cache; `uv run aqr options-pull` then `aqr options-embargo`")
    bars = _real_bars()
    volatility = VolatilityHistory.from_rows(_real_rows(_VOLATILITY))
    chain = ChainIndex.from_rows(_real_rows(_CHAIN))
    return OptionFeatureFrame(bars, volatility=volatility, chain=chain)


@pytest.mark.skipif(not _cache_present, reason="no option cache")
class TestRealCache:
    def test_iv_rank_is_bounded_and_mostly_low(self, real_frame: OptionFeatureFrame) -> None:
        """specs/10 D8: median 18.5, exceeds 50 on 17.8% of *sessions* (not
        bars). Bar-grid quartiles will differ a little because forward-fill
        stretches each session's value across the bars it covers -- asserted
        loosely, as a sanity bound rather than a restatement of D8's number.
        """
        values = real_frame.get(FeatureKey("iv_rank"))
        finite = values[~np.isnan(values)]
        assert finite.size > 1000
        assert np.nanmin(finite) >= -1e-6
        assert np.nanmax(finite) <= 100 + 1e-6
        median = float(np.median(finite))
        assert 5.0 < median < 35.0, f"iv_rank median moved to {median:.1f}"

    def test_term_slope_is_finite_and_usually_small(self, real_frame: OptionFeatureFrame) -> None:
        values = real_frame.get(FeatureKey("term_slope"))
        finite = values[~np.isnan(values)]
        assert finite.size > 1000
        median = float(np.median(finite))
        assert -0.05 < median < 0.05, f"term_slope median moved to {median:.4f}"

    def test_a_mixed_entry_expression_evaluates_on_the_real_grid(
        self, real_frame: OptionFeatureFrame
    ) -> None:
        node = parse("iv_rank() > 50 and close > sma(200)", resolve_feature=resolve_entry_feature)
        mask = evaluate(node, real_frame)
        assert mask.dtype == bool
        assert 0 < mask.sum() < len(real_frame)
