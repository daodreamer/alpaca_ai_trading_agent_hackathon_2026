"""Structure invariants over generated price paths — specs/14-testing.md.

A random walk is a better adversary than a hand-drawn zig-zag for the two claims
that must never fail: nothing is known before it could be, and nothing already
said is taken back.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.bar import Bar
from alphagate.core.structure import EqualPricePolicy, StructureEngine, StructureLabel, SwingKind
from tests.core.structure.synthetic import bar_at

centres = st.lists(
    st.decimals(min_value=Decimal("10"), max_value=Decimal("200"), places=2),
    min_size=1,
    max_size=50,
)
windows = st.integers(min_value=1, max_value=4)


def walk(path: list[Decimal]) -> tuple[Bar, ...]:
    return tuple(
        bar_at(
            index,
            open_=str(centre),
            high=str(centre + 1),
            low=str(centre - 1),
            close=str(centre),
        )
        for index, centre in enumerate(path)
    )


@given(centres, windows, windows)
def test_confirmation_always_lags_the_pivot_by_the_right_window(
    path: list[Decimal], left: int, right: int
) -> None:
    engine = StructureEngine(left_bars=left, right_bars=right)
    for bar in walk(path):
        engine.update(bar)

    for swing in engine.swings:
        assert swing.bar_start_utc < swing.detected_at_utc


@given(centres, windows)
def test_structure_is_only_ever_appended_to(path: list[Decimal], right: int) -> None:
    engine = StructureEngine(left_bars=1, right_bars=right)
    previous: tuple[object, ...] = ()

    for bar in walk(path):
        engine.update(bar)
        assert engine.swings[: len(previous)] == previous
        previous = engine.swings


@given(centres)
def test_a_pivot_high_is_the_highest_bar_in_its_own_window(path: list[Decimal]) -> None:
    left, right = 2, 2
    bars = walk(path)
    engine = StructureEngine(left_bars=left, right_bars=right)
    for bar in bars:
        engine.update(bar)

    index_of = {bar.start_time_utc: position for position, bar in enumerate(bars)}
    for swing in engine.swings:
        if swing.kind is not SwingKind.HIGH:
            continue
        position = index_of[swing.bar_start_utc]
        window = bars[position - left : position + right + 1]
        assert swing.price == max(bar.high for bar in window)


@given(centres)
def test_labels_agree_with_the_prices_they_were_derived_from(path: list[Decimal]) -> None:
    engine = StructureEngine(left_bars=1, right_bars=1, equal_prices=EqualPricePolicy.STRICT)
    for bar in walk(path):
        engine.update(bar)

    for kind, higher, lower in (
        (SwingKind.HIGH, StructureLabel.HH, StructureLabel.LH),
        (SwingKind.LOW, StructureLabel.HL, StructureLabel.LL),
    ):
        same_kind = [swing for swing in engine.swings if swing.kind is kind]
        assert same_kind[:1] == [] or same_kind[0].label is None
        for previous, current in itertools.pairwise(same_kind):
            expected = higher if current.price > previous.price else lower
            assert current.label is expected


@given(centres, windows)
def test_replaying_the_same_path_twice_gives_the_same_answer(
    path: list[Decimal], right: int
) -> None:
    bars = walk(path)

    def structure() -> tuple[object, object]:
        engine = StructureEngine(left_bars=1, right_bars=right)
        for bar in bars:
            engine.update(bar)
        return engine.swings, engine.breaks

    assert structure() == structure()
