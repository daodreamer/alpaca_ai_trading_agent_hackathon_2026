"""ADR 0005 — the exact numeric domain."""

from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.numeric import (
    PRICE_PLACES,
    format_exact,
    from_approximate,
    is_quantized,
    price,
    quantity,
)


class TestPrice:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("123.45", "123.45000000"),
            ("0", "0.00000000"),
            (7, "7.00000000"),
            (Decimal("1.000000005"), "1.00000000"),  # ROUND_HALF_EVEN -> down to even
            (Decimal("1.000000015"), "1.00000002"),  # ROUND_HALF_EVEN -> up to even
            ("-3.5", "-3.50000000"),
        ],
    )
    def test_quantizes_to_eight_places(self, raw: str | int | Decimal, expected: str) -> None:
        assert price(raw) == Decimal(expected)
        assert format_exact(price(raw)) == expected

    def test_rejects_float(self) -> None:
        with pytest.raises(TypeError, match="ADR 0005"):
            price(123.45)  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        # bool is a subclass of int; True must not silently become 1.00000000.
        with pytest.raises(TypeError):
            price(True)  # type: ignore[arg-type]

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "sNaN"])
    def test_rejects_non_finite(self, raw: str) -> None:
        with pytest.raises(InvariantViolation):
            price(raw)

    def test_rejects_unparseable_text(self) -> None:
        with pytest.raises(InvariantViolation):
            price("not a number")

    def test_is_idempotent(self) -> None:
        once = price("10.123456789")
        assert price(once) == once

    def test_preserves_significant_digits_beyond_four_places(self) -> None:
        # Sub-penny prints and provider VWAP routinely exceed 4dp; truncating on
        # ingest would destroy reproducibility of the provider's own aggregates.
        assert price("123.456789") == Decimal("123.45678900")


class TestQuantity:
    def test_allows_fractional_shares(self) -> None:
        assert quantity("0.00000001") == Decimal("0.00000001")

    def test_rejects_float(self) -> None:
        with pytest.raises(TypeError):
            quantity(1.5)  # type: ignore[arg-type]


class TestFromApproximate:
    def test_crosses_the_boundary_explicitly(self) -> None:
        assert from_approximate(0.1) == Decimal("0.10000000")

    def test_does_not_import_binary_error(self) -> None:
        # Decimal(0.1) would be 0.1000000000000000055511151231257827...
        assert from_approximate(0.1) != Decimal(0.1)  # noqa: RUF032 - the point of the test

    @pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, raw: float) -> None:
        with pytest.raises(InvariantViolation):
            from_approximate(raw)


class TestFormatExact:
    def test_zero_does_not_render_as_scientific_notation(self) -> None:
        # str(Decimal("0.00000000")) is "0E-8", which no JS client will display.
        assert format_exact(price("0")) == "0.00000000"

    def test_large_magnitudes_stay_plain(self) -> None:
        assert format_exact(price("1000000")) == "1000000.00000000"

    def test_round_trips_through_decimal(self) -> None:
        original = price("123.456789")
        assert Decimal(format_exact(original)) == original


class TestIsQuantized:
    def test_true_for_price_output(self) -> None:
        assert is_quantized(price("1.5"))

    def test_false_for_raw_decimal(self) -> None:
        assert not is_quantized(Decimal("1.5"))

    def test_places_constant_matches_adr(self) -> None:
        assert PRICE_PLACES == 8
