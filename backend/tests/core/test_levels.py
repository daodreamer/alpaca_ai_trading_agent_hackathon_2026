"""Level, Zone and UserLevel — specs/04-domain-model.md and specs/07-levels.md."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import LevelId, UserId, UserLevelId, ticker
from alphagate.core.levels import (
    Level,
    LevelKind,
    LevelSource,
    LevelStatus,
    Priority,
    UserLevel,
    Zone,
)
from alphagate.core.time_model import Timeframe

AAPL = ticker("AAPL")
NOW = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)


class TestZone:
    def test_low_must_not_exceed_high(self) -> None:
        with pytest.raises(InvariantViolation, match="zone_low"):
            Zone(low="105", high="100")

    def test_degenerate_zone_is_valid(self) -> None:
        zone = Zone(low="100", high="100")
        assert zone.width == Decimal("0.00000000")

    def test_contains_is_inclusive_at_both_bounds(self) -> None:
        zone = Zone(low="100", high="105")
        assert zone.contains("100")
        assert zone.contains("105")
        assert zone.contains("102.5")
        assert not zone.contains("99.99999999")
        assert not zone.contains("105.00000001")

    def test_midpoint(self) -> None:
        assert Zone(low="100", high="105").midpoint == Decimal("102.50000000")

    def test_width(self) -> None:
        assert Zone(low="100", high="105").width == Decimal("5.00000000")

    def test_bounds_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="positive"):
            Zone(low="0", high="105")

    def test_rejects_float_bounds(self) -> None:
        with pytest.raises(TypeError):
            Zone(low=100.0, high=105.0)  # type: ignore[arg-type]


def make_level(**overrides: object) -> Level:
    defaults: dict[str, object] = {
        "id": LevelId("lvl-1"),
        "symbol": AAPL,
        "timeframe": Timeframe.D1,
        "kind": LevelKind.RESISTANCE,
        "price": "105",
        "zone": Zone(low="104", high="106"),
        "sources": frozenset({LevelSource.SWING_HIGH}),
        "strength": 60.0,
        "confidence": 0.8,
        "status": LevelStatus.ACTIVE,
        "created_at": NOW,
        "updated_at": NOW,
    }
    return Level(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestLevel:
    def test_builds_a_valid_level(self) -> None:
        assert make_level().price == Decimal("105.00000000")

    def test_price_must_lie_within_the_zone(self) -> None:
        # specs/04-domain-model.md, explicit invariant.
        with pytest.raises(InvariantViolation, match="within its zone"):
            make_level(price="110", zone=Zone(low="104", high="106"))

    def test_price_at_a_zone_bound_is_valid(self) -> None:
        assert make_level(price="104", zone=Zone(low="104", high="106"))

    def test_zone_is_optional(self) -> None:
        assert make_level(zone=None).zone is None

    def test_requires_at_least_one_source(self) -> None:
        with pytest.raises(InvariantViolation, match="source"):
            make_level(sources=frozenset())

    @pytest.mark.parametrize("strength", [-0.1, 100.1])
    def test_strength_is_bounded_zero_to_hundred(self, strength: float) -> None:
        with pytest.raises(InvariantViolation, match="strength"):
            make_level(strength=strength)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_is_bounded_zero_to_one(self, confidence: float) -> None:
        with pytest.raises(InvariantViolation, match="confidence"):
            make_level(confidence=confidence)

    def test_updated_at_must_not_precede_created_at(self) -> None:
        with pytest.raises(InvariantViolation, match="updated_at"):
            make_level(updated_at=NOW.replace(hour=13))

    def test_candidate_status_is_distinct_from_active(self) -> None:
        # CLAUDE.md §12 — candidate vs confirmed must be distinguishable.
        assert make_level(status=LevelStatus.CANDIDATE).status is not LevelStatus.ACTIVE

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_level().price = Decimal("1")  # type: ignore[misc]

    def test_effective_zone_collapses_to_the_price_when_absent(self) -> None:
        level = make_level(zone=None, price="105")
        assert level.effective_zone == Zone(low="105", high="105")

    def test_distance_to_is_zero_inside_the_zone(self) -> None:
        assert make_level().distance_to("105") == Decimal("0.00000000")

    def test_distance_to_measures_from_the_nearest_zone_edge(self) -> None:
        level = make_level(zone=Zone(low="104", high="106"))
        assert level.distance_to("110") == Decimal("4.00000000")
        assert level.distance_to("100") == Decimal("4.00000000")


class TestUserLevel:
    def make(self, **overrides: object) -> UserLevel:
        defaults: dict[str, object] = {
            "id": UserLevelId("ul-1"),
            "user_id": UserId("u-1"),
            "symbol": AAPL,
            "timeframe": None,
            "kind": LevelKind.SUPPORT,
            "price": "95",
            "zone": None,
            "note": "weekly demand",
            "priority": Priority.NORMAL,
            "alert_policy": None,
            "active": True,
        }
        return UserLevel(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_builds_a_valid_user_level(self) -> None:
        assert self.make().price == Decimal("95.00000000")

    def test_timeframe_scope_is_optional(self) -> None:
        assert self.make().timeframe is None
        assert self.make(timeframe=Timeframe.H1).timeframe is Timeframe.H1

    def test_price_must_lie_within_the_zone(self) -> None:
        with pytest.raises(InvariantViolation, match="within its zone"):
            self.make(price="90", zone=Zone(low="94", high="96"))

    def test_identity_is_independent_of_any_rendered_chart_object(self) -> None:
        # specs/04-domain-model.md invariant: user-level identity is its own id.
        first = self.make(note="a")
        second = self.make(note="b")
        assert first.id == second.id

    def test_note_is_optional(self) -> None:
        assert self.make(note=None).note is None

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.make().active = False  # type: ignore[misc]
