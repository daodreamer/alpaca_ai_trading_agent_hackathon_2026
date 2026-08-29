"""Volume at price — specs/07-levels.md "Full version", ADR 0025 D1/D2.

The two properties everything else rests on: the profile conserves the volume it
was given exactly, and its grid does not move when the window does. A profile
that quietly loses a fraction of a share per bar would make `volume_context` a
function of how many bars were fetched.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine import ProfileConfig, ZoneWidth, build_volume_profile
from alphagate.core.levels import Zone
from tests.core.level_engine.synthetic import minute_bar

# A one-dollar grid keeps every expected bin edge writable by hand.
DOLLAR = ProfileConfig(bin_width=ZoneWidth.absolute("1"))


def profile(*bars: object, config: ProfileConfig = DOLLAR) -> object:
    return build_volume_profile(bars, config=config)  # type: ignore[arg-type]


# -- the grid ---------------------------------------------------------------


def test_a_flat_bar_puts_all_its_volume_in_one_bin() -> None:
    built = profile(minute_bar(0, high="100.5", low="100.5", volume="900"))

    assert built is not None
    assert built.total_volume == Decimal("900")
    (cell,) = built.bins
    assert (cell.low, cell.high) == (Decimal("100"), Decimal("101"))


def test_a_bar_spanning_bins_is_spread_across_them_in_proportion() -> None:
    """Half the range in each bin means half the volume in each (ADR 0025 D1)."""
    built = profile(minute_bar(0, high="101", low="99", volume="1000"))

    assert built is not None
    assert [cell.volume for cell in built.bins] == [Decimal("500"), Decimal("500")]


def test_the_profile_conserves_the_volume_it_was_given_exactly() -> None:
    """The rounding residual is assigned, not dropped — otherwise the total
    depends on how many bins the bars happened to straddle."""
    bars = [minute_bar(index, high="103", low="100", volume="1000") for index in range(7)]

    built = profile(*bars)

    assert built is not None
    assert built.total_volume == Decimal("7000")
    assert sum(cell.volume for cell in built.bins) == Decimal("7000")


def test_the_grid_is_anchored_to_the_width_not_to_the_windows_lowest_bar() -> None:
    """One more bar below must not move every boundary above it."""
    original = profile(minute_bar(0, high="100.5", low="100.5", volume="100"))
    extended = profile(
        minute_bar(0, high="100.5", low="100.5", volume="100"),
        minute_bar(1, high="97.25", low="97.25", volume="100"),
    )

    assert original is not None
    assert extended is not None
    assert original.bins[0].low == Decimal("100")
    assert extended.bins[-1].low == Decimal("100")  # the same cell, still


def test_bins_are_contiguous_across_a_price_the_market_skipped() -> None:
    """The empty cells inside the range are the whole point — they are the
    low-volume nodes."""
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="500"),
        minute_bar(1, high="105.5", low="105.5", volume="500"),
    )

    assert built is not None
    assert [cell.index for cell in built.bins] == list(
        range(built.bins[0].index, built.bins[0].index + 6)
    )
    assert built.bins[2].volume == Decimal(0)


def test_a_grid_too_fine_for_the_range_is_coarsened_by_a_whole_factor() -> None:
    config = ProfileConfig(bin_width=ZoneWidth.absolute("1"), max_bins=10)

    built = profile(
        minute_bar(0, high="100", low="100", volume="100"),
        minute_bar(1, high="180", low="180", volume="100"),
        config=config,
    )

    assert built is not None
    assert len(built.bins) <= 10
    # A whole factor keeps the coarse boundaries aligned with the fine ones.
    assert built.bin_width % Decimal("1") == 0


# -- what the profile says --------------------------------------------------


def test_the_point_of_control_is_the_price_the_most_traded_at() -> None:
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="100"),
        minute_bar(1, high="102.5", low="102.5", volume="900"),
        minute_bar(2, high="104.5", low="104.5", volume="100"),
    )

    assert built is not None
    assert built.poc.price == Decimal("102.5")
    assert built.poc.volume == Decimal("900")


def test_the_value_area_covers_the_configured_share_of_volume() -> None:
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="100"),
        minute_bar(1, high="101.5", low="101.5", volume="800"),
        minute_bar(2, high="102.5", low="102.5", volume="100"),
    )

    assert built is not None
    # The 800 in the middle bin is 80% on its own, so the area stops there.
    assert built.value_area == Zone(low=Decimal("101"), high=Decimal("102"))


def test_the_value_area_grows_toward_the_heavier_side() -> None:
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="100"),
        minute_bar(1, high="101.5", low="101.5", volume="500"),
        minute_bar(2, high="102.5", low="102.5", volume="400"),
    )

    assert built is not None
    assert built.value_area.high == Decimal("103")  # the 400 side, not the 100 side


def test_a_high_volume_shelf_is_one_node_not_three() -> None:
    """Reporting one node per bin would be reporting the grid (ADR 0025 D2)."""
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="1000"),
        minute_bar(1, high="110.5", low="110.5", volume="800"),
        minute_bar(2, high="111.5", low="111.5", volume="950"),
        minute_bar(3, high="112.5", low="112.5", volume="900"),
        minute_bar(4, high="120.5", low="120.5", volume="10"),
    )

    assert built is not None
    (node,) = built.high_nodes
    assert node.price == Decimal("111.5")  # the heaviest bin of the shelf
    assert node.zone == Zone(low=Decimal("110"), high=Decimal("113"))
    assert node.volume == Decimal("2650")


def test_the_shelf_around_the_point_of_control_is_not_reported_twice() -> None:
    """The bins beside the POC are the POC's own shelf. A high node one bin away
    from it would be the same evidence, drawn again."""
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="1000"),
        minute_bar(1, high="101.5", low="101.5", volume="950"),
        minute_bar(2, high="102.5", low="102.5", volume="900"),
    )

    assert built is not None
    assert built.poc.price == Decimal("100.5")
    assert built.high_nodes == ()


def test_a_low_volume_node_sits_in_the_middle_of_the_ground_price_crossed() -> None:
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="1000"),
        minute_bar(1, high="105.5", low="105.5", volume="1000"),
    )

    assert built is not None
    (node,) = built.low_nodes
    assert node.price == Decimal("103")  # the empty middle of 101–105
    assert node.volume == Decimal(0)


def test_the_tails_of_the_distribution_are_not_low_volume_nodes() -> None:
    """A quiet edge is where the profile ends, not a shelf price crossed."""
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="10"),
        minute_bar(1, high="101.5", low="101.5", volume="1000"),
        minute_bar(2, high="102.5", low="102.5", volume="10"),
    )

    assert built is not None
    assert built.low_nodes == ()


# -- volume context ---------------------------------------------------------


def test_context_is_highest_at_the_point_of_control() -> None:
    built = profile(
        minute_bar(0, high="100.5", low="100.5", volume="100"),
        minute_bar(1, high="102.5", low="102.5", volume="900"),
    )

    assert built is not None
    busy = built.context(Zone(low=Decimal("102"), high=Decimal("103")))
    quiet = built.context(Zone(low=Decimal("100"), high=Decimal("101")))

    assert busy == pytest.approx(1.0)
    assert quiet is not None
    assert quiet < busy


def test_context_of_an_exact_level_is_measured_over_at_least_one_bin() -> None:
    """A zero-width window would score every exact level zero (ADR 0025 D8)."""
    built = profile(minute_bar(0, high="102.5", low="102.5", volume="900"))

    assert built is not None
    exact = Zone(low=Decimal("102.5"), high=Decimal("102.5"))
    assert built.context(exact) == pytest.approx(1.0)


def test_context_stays_within_its_bounds() -> None:
    built = profile(
        minute_bar(0, high="103", low="100", volume="1000"),
        minute_bar(1, high="103", low="100", volume="1000"),
    )

    assert built is not None
    for low, high in ((100, 101), (101, 102), (102, 103), (99, 104)):
        value = built.context(Zone(low=Decimal(low), high=Decimal(high)))
        assert value is not None
        assert 0.0 <= value <= 1.0


# -- refusals ---------------------------------------------------------------


def test_there_is_no_profile_without_volume() -> None:
    assert profile(minute_bar(0, high="101", low="99", volume="0")) is None


def test_there_is_no_profile_from_an_unfinalized_bar() -> None:
    assert profile(minute_bar(0, high="101", low="99", is_final=False)) is None


def test_there_is_no_profile_when_the_bin_width_cannot_be_resolved() -> None:
    """ATR-based bins before ATR is warm: no profile, not a fabricated grid."""
    config = ProfileConfig(bin_width=ZoneWidth.atr("0.5"))

    assert build_volume_profile([minute_bar(0, high="101", low="99")], config=config) is None


def test_there_is_no_profile_from_no_bars() -> None:
    assert build_volume_profile([]) is None


def test_the_profile_is_the_same_however_often_it_is_built() -> None:
    bars = [
        minute_bar(0, high="103", low="100", volume="1000"),
        minute_bar(1, high="104", low="101", volume="1300"),
        minute_bar(2, high="102", low="99", volume="700"),
    ]

    assert build_volume_profile(bars, config=DOLLAR) == build_volume_profile(bars, config=DOLLAR)


@pytest.mark.parametrize(
    ("field", "value"),
    [("value_area", 0.0), ("value_area", 1.5), ("high_node_ratio", 0.0), ("max_bins", 0)],
)
def test_an_impossible_profile_configuration_is_refused(field: str, value: float) -> None:
    with pytest.raises(InvariantViolation, match=field):
        ProfileConfig(**{field: value})  # type: ignore[arg-type]


def test_a_bin_cannot_be_both_kinds_of_node() -> None:
    with pytest.raises(InvariantViolation, match="low_node_ratio"):
        ProfileConfig(high_node_ratio=0.3, low_node_ratio=0.5)
