"""The capital sleeve — specs/03 D6.

One property carries this whole module, and every test below is a way of
stating it:

    A sleeve's equity is a function of its own allocation and its own P&L.
    The account it lives in does not appear in the arithmetic.

That is what makes two strategies sharing one broker account independent. The
equity book can lose a fifth of itself without moving the options sleeve's
drawdown by a cent, and the options sleeve can be wiped out without touching
the equity book's kill switch. Before this existed both gates scaled off
`account.equity` and each could latch the other's kill switch for a reason that
had nothing to do with it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.risk.sleeve import Sleeve, residual_sleeve


def sleeve(**overrides: object) -> Sleeve:
    base: dict[str, object] = {
        "name": "options",
        "allocation": Decimal(5000),
        "realised": Decimal(0),
        "unrealised": Decimal(0),
    }
    base.update(overrides)
    return Sleeve(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# equity
# ------------------------------------------------------------------ #


def test_equity_is_the_allocation_when_nothing_has_happened() -> None:
    assert sleeve().equity == Decimal(5000)


def test_equity_adds_realised_and_unrealised() -> None:
    book = sleeve(realised=Decimal(120), unrealised=Decimal("-45.50"))
    assert book.equity == Decimal("5074.50")


def test_equity_is_decimal_end_to_end() -> None:
    """Rule 4: money is `Decimal`. A float here would round a kill switch."""
    assert isinstance(sleeve(realised=Decimal("0.01")).equity, Decimal)


def test_a_loss_larger_than_the_allocation_floors_at_zero() -> None:
    """Equity is what the sleeve has, and it cannot have less than nothing.

    Defined-risk structures make this unreachable, but a limit that relies on a
    strategy invariant to stay sane is a limit that breaks the day the strategy
    changes. A negative equity would invert every percentage limit derived from
    it, which is worse than being wrong -- it would silently permit more.
    """
    assert sleeve(realised=Decimal(-9000)).equity == Decimal(0)


# ------------------------------------------------------------------ #
# isolation -- the reason this type exists
# ------------------------------------------------------------------ #


def test_the_account_does_not_appear_in_the_signature() -> None:
    """Structural, not behavioural.

    A future edit that "helpfully" passes account equity in to make a sleeve
    aware of its surroundings would re-couple the two kill switches, and it
    would do so without failing any arithmetic test. So the absence of the
    parameter is asserted directly.
    """
    import inspect

    fields = set(inspect.signature(Sleeve).parameters)
    assert fields == {"name", "allocation", "realised", "unrealised"}


def test_two_sleeves_with_the_same_allocation_are_independent() -> None:
    equities = sleeve(name="equity", allocation=Decimal(95000), unrealised=Decimal(-9000))
    options = sleeve(unrealised=Decimal(30))
    assert equities.equity == Decimal(86000)
    assert options.equity == Decimal(5030)
    assert options.drawdown(peak=Decimal(5000)) == Decimal(0)


# ------------------------------------------------------------------ #
# drawdown
# ------------------------------------------------------------------ #


def test_drawdown_is_zero_without_a_peak() -> None:
    """No history is not a drawdown of zero percent, it is no measurement.

    Zero is returned because the kill switch must not fire on a first run, and
    the caller is the one that carries the high-water mark across days.
    """
    assert sleeve().drawdown(peak=None) == Decimal(0)


def test_drawdown_is_zero_at_the_peak() -> None:
    assert sleeve().drawdown(peak=Decimal(5000)) == Decimal(0)


def test_drawdown_is_zero_above_the_peak() -> None:
    assert sleeve(realised=Decimal(500)).drawdown(peak=Decimal(5000)) == Decimal(0)


def test_drawdown_below_the_peak() -> None:
    """$5,000 peak, $1,000 lost, and the answer is a fifth -- not 1% of an account."""
    assert sleeve(realised=Decimal(-1000)).drawdown(peak=Decimal(5000)) == Decimal("0.2")


def test_drawdown_is_measured_against_the_sleeve_not_the_account() -> None:
    """The whole point, as a number.

    A $100k account holding a $95k equity book and a $5k options sleeve. The
    equity book loses $8,000 -- an 8% account drawdown, which under the old
    account-scaled rule tripped the options kill switch at its 5% threshold.
    The options sleeve, having lost nothing, must report zero.
    """
    options = sleeve(realised=Decimal(0), unrealised=Decimal(0))
    assert options.drawdown(peak=Decimal(5000)) == Decimal(0)


def test_a_nonpositive_peak_is_not_a_drawdown() -> None:
    assert sleeve().drawdown(peak=Decimal(0)) == Decimal(0)


# ------------------------------------------------------------------ #
# construction
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("bad", [Decimal(0), Decimal(-1)])
def test_allocation_must_be_positive(bad: Decimal) -> None:
    """A zero allocation is not a conservative sleeve, it is a disabled one.

    Same wording as `RiskLimits`, and for the same reason: every budget in
    `limits.py` is a fraction of this number, so a zero here silently disables
    every check that scales off it rather than tightening them.
    """
    with pytest.raises(InvariantViolation, match="allocation"):
        sleeve(allocation=bad)


@pytest.mark.parametrize("field", ["allocation", "realised", "unrealised"])
def test_money_must_be_decimal(field: str) -> None:
    with pytest.raises(InvariantViolation, match=field):
        sleeve(**{field: 5000.0})


def test_name_must_not_be_empty() -> None:
    """The name is what the journal and the dashboard call this pool of money."""
    with pytest.raises(InvariantViolation, match="name"):
        sleeve(name="")


def test_sleeve_is_frozen() -> None:
    """An allocation that can be reassigned is not an allocation."""
    book = sleeve()
    with pytest.raises(FrozenInstanceError):
        book.allocation = Decimal(1)  # type: ignore[misc]


# ------------------------------------------------------------------ #
# determinism -- Rule 7
# ------------------------------------------------------------------ #


def test_same_inputs_same_outputs() -> None:
    a = sleeve(realised=Decimal("12.34"), unrealised=Decimal("-5.67"))
    b = sleeve(realised=Decimal("12.34"), unrealised=Decimal("-5.67"))
    assert a == b
    assert a.equity == b.equity
    assert a.drawdown(peak=Decimal(5000)) == b.drawdown(peak=Decimal(5000))


# ------------------------------------------------------------------ #
# the residual -- the other half of the split
# ------------------------------------------------------------------ #


class TestResidual:
    """`account = residual + others`, held as an identity rather than hoped for."""

    def test_a_flat_account_splits_into_its_allocations(self) -> None:
        options = sleeve()
        book = residual_sleeve(
            "equity",
            allocation=Decimal(95000),
            account_equity=Decimal(100000),
            others=(options,),
        )
        assert book.equity == Decimal(95000)
        assert book.equity + options.equity == Decimal(100000)

    def test_an_options_loss_does_not_land_on_the_equity_book(self) -> None:
        """The identity doing its job.

        Options lose $1,000. The account falls to $99,000 *and* the options
        sleeve falls to $4,000, so the residual is still $95,000. Under the old
        account-scaled rule the equity book would have seen a 1% drawdown it did
        not take.
        """
        options = sleeve(realised=Decimal(-1000))
        book = residual_sleeve(
            "equity",
            allocation=Decimal(95000),
            account_equity=Decimal(99000),
            others=(options,),
        )
        assert options.equity == Decimal(4000)
        assert book.equity == Decimal(95000)
        assert book.drawdown(peak=Decimal(95000)) == Decimal(0)

    def test_an_equity_loss_lands_entirely_on_the_equity_book(self) -> None:
        """And the other direction, which is the one that was actually biting."""
        options = sleeve()
        book = residual_sleeve(
            "equity",
            allocation=Decimal(95000),
            account_equity=Decimal(92000),
            others=(options,),
        )
        assert book.equity == Decimal(87000)
        assert options.equity == Decimal(5000)
        assert options.drawdown(peak=Decimal(5000)) == Decimal(0)

    def test_the_residual_of_an_empty_account_floors_at_zero(self) -> None:
        book = residual_sleeve(
            "equity",
            allocation=Decimal(95000),
            account_equity=Decimal(1000),
            others=(sleeve(),),
        )
        assert book.equity == Decimal(0)

    def test_no_others_means_the_residual_is_the_whole_account(self) -> None:
        book = residual_sleeve(
            "equity",
            allocation=Decimal(95000),
            account_equity=Decimal(97000),
            others=(),
        )
        assert book.equity == Decimal(97000)
