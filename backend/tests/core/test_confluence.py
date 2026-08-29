"""Multi-timeframe confluence — specs/18-roadmap.md F1, ADR 0024.

The engine synthesises what the per-timeframe engines already decided. It reads
`TrendState`s, structure labels and `Level`s and produces one explainable view;
it computes no indicator, confirms no trend and generates no level.

The invariants tested here are the ones this codebase applies to every score:
contributions sum to the number they explain, unreadable inputs are not read as
zero, and the same inputs produce the same output — ordering included.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.core.confluence import (
    ConfluenceConfig,
    TimeframeInput,
    build_confluence,
)
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import LevelId, Ticker, ticker
from alphagate.core.levels import Level, LevelKind, LevelSource, LevelStatus, Zone
from alphagate.core.structure import StructureLabel
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendDirection, TrendPhase, TrendState

AAPL: Ticker = ticker("AAPL")
NOW = datetime(2026, 11, 25, 21, 0, tzinfo=UTC)
TOLERANCE = 1e-9


def state(
    timeframe: Timeframe,
    phase: TrendPhase,
    *,
    strength: float = 60.0,
    confidence: float = 1.0,
    as_of: datetime = NOW,
) -> TrendState:
    return TrendState(
        symbol=AAPL,
        timeframe=timeframe,
        state=phase,
        strength=strength,
        confidence=confidence,
        as_of=as_of,
        reason_codes=() if phase is TrendPhase.UNKNOWN else ("EMA_ALIGNMENT_BULL",),
        input_snapshot_hash="hash",
    )


def view(
    timeframe: Timeframe,
    phase: TrendPhase | None = None,
    *,
    high: StructureLabel | None = None,
    low: StructureLabel | None = None,
    levels: tuple[Level, ...] = (),
    as_of: datetime = NOW,
) -> TimeframeInput:
    return TimeframeInput(
        timeframe=timeframe,
        trend=None if phase is None else state(timeframe, phase, as_of=as_of),
        swing_high_label=high,
        swing_low_label=low,
        levels=levels,
    )


def level(
    timeframe: Timeframe,
    price: str,
    *,
    kind: LevelKind = LevelKind.SUPPORT,
    half_width: str = "0.10",
    status: LevelStatus = LevelStatus.ACTIVE,
    strength: float = 50.0,
) -> Level:
    centre = Decimal(price)
    margin = Decimal(half_width)
    return Level(
        id=LevelId(f"{timeframe.code}:{kind.value}:{price}"),
        symbol=AAPL,
        timeframe=timeframe,
        kind=kind,
        price=centre,
        zone=Zone(low=centre - margin, high=centre + margin),
        sources=frozenset({LevelSource.SWING_LOW}),
        strength=strength,
        confidence=1.0,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class TestTrendAlignment:
    def test_every_timeframe_bullish_aligns_fully(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.STRONG_BULLISH),
                view(Timeframe.H1, TrendPhase.STRONG_BULLISH),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        assert result.trend_alignment == pytest.approx(1.0)
        assert result.direction is TrendDirection.BULLISH
        assert result.conflicts == ()

    def test_the_ladder_is_what_a_timeframe_contributes(self) -> None:
        """`specs/08-trend.md` already decided that the rung *is* the trend's
        strength as far as anything downstream is concerned, so confluence reads
        the rung rather than inventing a second scale."""
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.WATCH_BULL),
                view(Timeframe.H1, TrendPhase.BULLISH),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        # (1 + 2 + 3) / 3 rungs above NEUTRAL, each over the three-rung span.
        assert result.trend_alignment == pytest.approx((1 / 3 + 2 / 3 + 1.0) / 3)

    def test_opposing_timeframes_cancel_and_are_named(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.STRONG_BEARISH),
                view(Timeframe.H1, TrendPhase.NEUTRAL),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        assert result.trend_alignment == pytest.approx(0.0)
        # Zero from cancellation is not zero from agreement, and the difference
        # is the whole reason anybody looks at this screen.
        assert result.trend_agreement == pytest.approx(0.5)
        assert result.direction is TrendDirection.NEUTRAL

    def test_everything_neutral_is_not_the_same_as_everything_disagreeing(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.NEUTRAL),
                view(Timeframe.H1, TrendPhase.NEUTRAL),
                view(Timeframe.D1, TrendPhase.NEUTRAL),
            ],
            as_of=NOW,
        )
        assert result.trend_alignment == pytest.approx(0.0)
        assert result.trend_agreement is None, "nothing is taking a side to agree about"
        assert result.conflicts == ()

    def test_a_minority_timeframe_is_reported_as_the_conflict(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.BEARISH),
                view(Timeframe.H1, TrendPhase.BULLISH),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        assert result.direction is TrendDirection.BULLISH
        assert result.conflicts == (Timeframe.M5,)
        assert result.trend_agreement == pytest.approx(2 / 3)

    def test_a_lean_inside_the_neutral_band_reads_neutral(self) -> None:
        """One rung of one timeframe out of three is a lean, not a stance."""
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.WATCH_BULL),
                view(Timeframe.H1, TrendPhase.NEUTRAL),
                view(Timeframe.D1, TrendPhase.NEUTRAL),
            ],
            as_of=NOW,
        )
        assert 0 < result.trend_alignment < ConfluenceConfig().neutral_band
        assert result.direction is TrendDirection.NEUTRAL


class TestUnreadableTimeframes:
    def test_a_missing_state_leaves_both_sides_of_the_average(self) -> None:
        """The rule `specs/08-trend.md` sets for evidence, applied to timeframes:
        a timeframe with no opinion must not drag the answer toward neutral."""
        both = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.STRONG_BULLISH),
                view(Timeframe.H1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        with_a_gap = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.STRONG_BULLISH),
                view(Timeframe.H1, TrendPhase.STRONG_BULLISH),
                view(Timeframe.D1, None),
            ],
            as_of=NOW,
        )
        assert with_a_gap.trend_alignment == both.trend_alignment == pytest.approx(1.0)

    def test_an_unknown_phase_is_unreadable_not_neutral(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.STRONG_BULLISH),
                view(Timeframe.D1, TrendPhase.UNKNOWN),
            ],
            as_of=NOW,
        )
        assert result.trend_alignment == pytest.approx(1.0)
        assert result.confidence == pytest.approx(0.5)

    def test_confidence_is_readable_over_requested(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.BULLISH),
                view(Timeframe.M15, TrendPhase.BULLISH),
                view(Timeframe.H1, None),
                view(Timeframe.D1, None),
            ],
            as_of=NOW,
        )
        assert result.confidence == pytest.approx(0.5)

    def test_nothing_readable_leaves_the_alignment_undefined(self) -> None:
        result = build_confluence(
            AAPL,
            [view(Timeframe.M5, None), view(Timeframe.D1, TrendPhase.UNKNOWN)],
            as_of=NOW,
        )
        assert result.trend_alignment is None
        assert result.direction is TrendDirection.UNKNOWN
        assert result.confidence == pytest.approx(0.0)


class TestExplainability:
    def test_contributions_sum_to_the_alignment(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.BEARISH),
                view(Timeframe.H1, TrendPhase.WATCH_BULL),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        assert result.trend_alignment is not None
        total = sum(stance.trend_contribution for stance in result.stances)
        assert total == pytest.approx(result.trend_alignment, abs=TOLERANCE)

    def test_a_timeframe_arguing_the_other_way_contributes_a_negative_number(self) -> None:
        """A statement about the market, not a rounding error — the convention
        `specs/08-trend.md` sets for strength components."""
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.BEARISH),
                view(Timeframe.D1, TrendPhase.STRONG_BULLISH),
            ],
            as_of=NOW,
        )
        by_timeframe = {stance.timeframe: stance for stance in result.stances}
        assert by_timeframe[Timeframe.M5].trend_contribution < 0
        assert by_timeframe[Timeframe.D1].trend_contribution > 0

    def test_an_unreadable_timeframe_still_appears_with_nothing_to_say(self) -> None:
        """A row that vanished would be indistinguishable from one nobody asked
        for, which is the same reason the trend engine reports every evidence
        kind including the ones it could not read."""
        result = build_confluence(
            AAPL, [view(Timeframe.M5, TrendPhase.BULLISH), view(Timeframe.W1, None)], as_of=NOW
        )
        weekly = next(s for s in result.stances if s.timeframe is Timeframe.W1)
        assert weekly.trend_bias is None
        assert weekly.trend_contribution == 0.0
        assert weekly.phase is TrendPhase.UNKNOWN


class TestStructureConfluence:
    def test_higher_highs_and_higher_lows_read_bullish(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, high=StructureLabel.HH, low=StructureLabel.HL),
                view(Timeframe.D1, high=StructureLabel.HH, low=StructureLabel.HL),
            ],
            as_of=NOW,
        )
        assert result.structure_alignment == pytest.approx(1.0)

    def test_a_mixed_pair_takes_no_side(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, high=StructureLabel.HH, low=StructureLabel.LL),
                view(Timeframe.D1, high=StructureLabel.HH, low=StructureLabel.HL),
            ],
            as_of=NOW,
        )
        assert result.structure_alignment == pytest.approx(0.5)

    def test_one_missing_label_makes_the_timeframe_unreadable(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, high=StructureLabel.HH),
                view(Timeframe.D1, high=StructureLabel.LH, low=StructureLabel.LL),
            ],
            as_of=NOW,
        )
        assert result.structure_alignment == pytest.approx(-1.0)

    def test_structure_contributions_sum_to_the_alignment(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, high=StructureLabel.LH, low=StructureLabel.LL),
                view(Timeframe.H1, high=StructureLabel.HH, low=StructureLabel.HL),
                view(Timeframe.D1, high=StructureLabel.HH, low=StructureLabel.LL),
            ],
            as_of=NOW,
        )
        assert result.structure_alignment is not None
        total = sum(stance.structure_contribution for stance in result.stances)
        assert total == pytest.approx(result.structure_alignment, abs=TOLERANCE)

    def test_no_labels_anywhere_leaves_it_undefined(self) -> None:
        result = build_confluence(AAPL, [view(Timeframe.H1), view(Timeframe.D1)], as_of=NOW)
        assert result.structure_alignment is None


class TestLevelClusters:
    def test_levels_at_the_same_price_on_two_timeframes_become_one_cluster(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=(level(Timeframe.H1, "100.00"),)),
                view(Timeframe.D1, levels=(level(Timeframe.D1, "100.05"),)),
            ],
            as_of=NOW,
        )
        assert len(result.clusters) == 1
        cluster = result.clusters[0]
        assert cluster.timeframes == (Timeframe.H1, Timeframe.D1)
        assert cluster.timeframe_count == 2
        assert cluster.price == Decimal("100.025")

    def test_a_cluster_covers_the_union_of_its_zones(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=(level(Timeframe.H1, "100.00", half_width="0.10"),)),
                view(Timeframe.D1, levels=(level(Timeframe.D1, "100.05", half_width="0.20"),)),
            ],
            as_of=NOW,
        )
        cluster = result.clusters[0]
        # 1h covers 99.90–100.10, 1D covers 99.85–100.25; together, the wider band.
        assert cluster.zone_low == Decimal("99.85")
        assert cluster.zone_high == Decimal("100.25")

    def test_levels_far_apart_stay_apart(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=(level(Timeframe.H1, "100.00"),)),
                view(Timeframe.D1, levels=(level(Timeframe.D1, "140.00"),)),
            ],
            as_of=NOW,
        )
        assert [c.timeframe_count for c in result.clusters] == [1, 1]

    def test_support_and_resistance_never_share_a_cluster(self) -> None:
        """A level that was its own opposite would mean nothing — the rule
        `specs/07-levels.md` already sets for candidates within one timeframe."""
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=(level(Timeframe.H1, "100.00"),)),
                view(
                    Timeframe.D1,
                    levels=(level(Timeframe.D1, "100.02", kind=LevelKind.RESISTANCE),),
                ),
            ],
            as_of=NOW,
        )
        assert len(result.clusters) == 2
        assert {c.kind for c in result.clusters} == {LevelKind.SUPPORT, LevelKind.RESISTANCE}

    def test_a_chain_of_near_levels_cannot_merge_without_limit(self) -> None:
        """The cap ADR 0009 puts on single linkage, for the same reason: without
        it "support at 100" silently becomes "support between 99 and 104"."""
        prices = ("100.00", "100.20", "100.40", "100.60", "100.80", "101.00")
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=tuple(level(Timeframe.H1, p) for p in prices)),
            ],
            as_of=NOW,
            config=ConfluenceConfig(
                merge_distance_pct=Decimal("0.003"), max_cluster_width_pct=Decimal("0.005")
            ),
        )
        widths = [c.zone_high - c.zone_low for c in result.clusters]
        assert len(result.clusters) > 1
        assert max(widths) <= Decimal("100") * Decimal("0.005") + Decimal("0.30")

    def test_an_invalidated_level_is_not_in_force(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.H1, levels=(level(Timeframe.H1, "100.00"),)),
                view(
                    Timeframe.D1,
                    levels=(level(Timeframe.D1, "100.02", status=LevelStatus.INVALIDATED),),
                ),
            ],
            as_of=NOW,
        )
        assert [c.timeframe_count for c in result.clusters] == [1]

    def test_clusters_come_back_by_price(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(
                    Timeframe.H1,
                    levels=(
                        level(Timeframe.H1, "130.00"),
                        level(Timeframe.H1, "100.00"),
                        level(Timeframe.H1, "115.00"),
                    ),
                ),
            ],
            as_of=NOW,
        )
        prices = [c.price for c in result.clusters]
        assert prices == sorted(prices)


class TestDeterminism:
    def test_the_order_timeframes_arrive_in_does_not_matter(self) -> None:
        views = [
            view(Timeframe.D1, TrendPhase.BULLISH, levels=(level(Timeframe.D1, "100.05"),)),
            view(Timeframe.M5, TrendPhase.BEARISH, levels=(level(Timeframe.M5, "100.00"),)),
            view(Timeframe.H1, TrendPhase.NEUTRAL),
        ]
        assert build_confluence(AAPL, views, as_of=NOW) == build_confluence(
            AAPL, list(reversed(views)), as_of=NOW
        )

    def test_stances_are_ordered_coarsest_last(self) -> None:
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.D1, TrendPhase.BULLISH),
                view(Timeframe.M5, TrendPhase.BULLISH),
                view(Timeframe.H1, TrendPhase.BULLISH),
            ],
            as_of=NOW,
        )
        assert [s.timeframe for s in result.stances] == [
            Timeframe.M5,
            Timeframe.H1,
            Timeframe.D1,
        ]

    def test_repeating_the_call_produces_an_equal_result(self) -> None:
        views = [view(Timeframe.M5, TrendPhase.BULLISH), view(Timeframe.D1, TrendPhase.BEARISH)]
        assert build_confluence(AAPL, views, as_of=NOW) == build_confluence(AAPL, views, as_of=NOW)


class TestValidation:
    def test_a_timeframe_cannot_be_given_twice(self) -> None:
        with pytest.raises(InvariantViolation, match="1h"):
            build_confluence(
                AAPL,
                [view(Timeframe.H1, TrendPhase.BULLISH), view(Timeframe.H1, TrendPhase.BEARISH)],
                as_of=NOW,
            )

    def test_a_state_for_another_symbol_is_refused(self) -> None:
        other = TimeframeInput(
            timeframe=Timeframe.H1,
            trend=TrendState(
                symbol=ticker("MSFT"),
                timeframe=Timeframe.H1,
                state=TrendPhase.BULLISH,
                strength=50.0,
                confidence=1.0,
                as_of=NOW,
                reason_codes=("EMA_ALIGNMENT_BULL",),
                input_snapshot_hash="hash",
            ),
        )
        with pytest.raises(InvariantViolation, match="MSFT"):
            build_confluence(AAPL, [other], as_of=NOW)

    def test_nothing_at_all_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="at least one"):
            build_confluence(AAPL, [], as_of=NOW)

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="weight"):
            ConfluenceConfig(weights={Timeframe.H1: -1.0})


class TestFreshness:
    def test_the_view_reports_each_timeframes_own_as_of(self) -> None:
        """A weekly state dated last Friday is not stale, it is weekly. The view
        reports the instants and leaves that judgement to the reader."""
        friday = NOW - timedelta(days=3)
        result = build_confluence(
            AAPL,
            [
                view(Timeframe.M5, TrendPhase.BULLISH, as_of=NOW),
                view(Timeframe.W1, TrendPhase.BULLISH, as_of=friday),
            ],
            as_of=NOW,
        )
        by_timeframe = {s.timeframe: s for s in result.stances}
        assert by_timeframe[Timeframe.M5].as_of == NOW
        assert by_timeframe[Timeframe.W1].as_of == friday
