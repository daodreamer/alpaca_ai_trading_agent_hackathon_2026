"""Zone widths, weights and configuration — ADR 0009 D3/D6.

`specs/07-levels.md` requires that these be configuration rather than hidden
constants. That only helps if bad configuration is rejected loudly, which is
most of what this module checks.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine import (
    DEFAULT_SOURCE_QUALITY,
    LevelCandidate,
    LevelConfig,
    StrengthWeights,
    ZoneWidth,
    ZoneWidthMode,
)
from alphagate.core.levels import LevelKind, LevelSource
from tests.core.level_engine.synthetic import candidate

PRICE = Decimal("200")
ATR = Decimal("2.5")


def test_an_absolute_width_ignores_price_and_volatility() -> None:
    width = ZoneWidth.absolute("0.4")

    assert width.resolve(price=PRICE, atr=ATR) == Decimal("0.4")
    assert width.resolve(price=Decimal("5"), atr=None) == Decimal("0.4")


def test_a_percent_width_scales_with_price() -> None:
    width = ZoneWidth.percent("0.5")

    assert width.resolve(price=PRICE, atr=None) == Decimal("1")
    assert width.resolve(price=Decimal("50"), atr=None) == Decimal("0.25")


def test_an_atr_width_scales_with_volatility() -> None:
    width = ZoneWidth.atr("0.25")

    assert width.resolve(price=PRICE, atr=ATR) == Decimal("0.625")


def test_an_atr_width_is_unknowable_before_atr_is_warm() -> None:
    """`None`, never a fabricated width (ADR 0009 D3)."""
    assert ZoneWidth.atr("0.25").resolve(price=PRICE, atr=None) is None


def test_a_negative_width_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="negative"):
        ZoneWidth.absolute("-1")


def test_a_width_carries_a_readable_label() -> None:
    assert ZoneWidth.atr("0.25").label == "ATR:0.25"
    assert ZoneWidth.percent("1").label == "PERCENT:1"


def test_width_modes_are_distinct_values() -> None:
    assert ZoneWidth.absolute("1").mode is ZoneWidthMode.ABSOLUTE
    assert ZoneWidth.percent("1").mode is ZoneWidthMode.PERCENT
    assert ZoneWidth.atr("1").mode is ZoneWidthMode.ATR


# -- weights ----------------------------------------------------------------


def test_the_default_weights_leave_full_version_components_switched_off() -> None:
    """Declared with weight 0 so enabling them is configuration, not a schema change."""
    weights = StrengthWeights()

    assert weights.volume_context == 0.0
    assert weights.multi_timeframe == 0.0
    assert weights.as_mapping()["source_quality"] == 1.0


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="negative"):
        StrengthWeights(touches=-1.0)


def test_weights_that_are_all_zero_are_refused() -> None:
    with pytest.raises(InvariantViolation, match="positive"):
        StrengthWeights(source_quality=0.0, touches=0.0, recency=0.0, confluence=0.0)


def test_the_weight_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        StrengthWeights().as_mapping()["touches"] = 5.0  # type: ignore[index]


# -- config -----------------------------------------------------------------


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.5])
def test_recency_decay_must_be_a_proper_fraction(decay: float) -> None:
    with pytest.raises(InvariantViolation, match="recency_decay"):
        LevelConfig(recency_decay=decay)


def test_decay_of_one_is_allowed_and_is_the_default() -> None:
    assert LevelConfig().recency_decay == 1.0


@pytest.mark.parametrize("field", ["touch_saturation", "confluence_saturation"])
def test_saturations_must_be_at_least_one(field: str) -> None:
    with pytest.raises(InvariantViolation, match=field):
        LevelConfig(**{field: 0})


def test_a_source_quality_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="SWING_HIGH"):
        LevelConfig(source_quality={LevelSource.SWING_HIGH: 1.5})


def test_an_unlisted_source_scores_neutral() -> None:
    config = LevelConfig(source_quality={LevelSource.SWING_HIGH: 1.0})

    assert config.quality_of(LevelSource.YEAR_HIGH) == 0.5


def test_the_default_quality_table_covers_every_mvp_source() -> None:
    assert set(DEFAULT_SOURCE_QUALITY) == set(LevelSource)


# -- candidates -------------------------------------------------------------


def test_a_candidate_price_must_be_positive() -> None:
    with pytest.raises(InvariantViolation, match="positive"):
        candidate("0")


def test_a_candidate_needs_a_timezone_aware_observation_time() -> None:
    with pytest.raises(InvariantViolation, match="timezone"):
        LevelCandidate(
            price="100",
            kind=LevelKind.RESISTANCE,
            source=LevelSource.SWING_HIGH,
            observed_at_utc=datetime(2026, 8, 13, 14, 0),  # noqa: DTZ001 — the point of the test
        )
