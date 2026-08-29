"""Symbol identity — specs/04-domain-model.md."""

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.symbol import AssetType, Symbol


class TestTicker:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("aapl", "AAPL"),
            (" msft ", "MSFT"),
            ("AAPL\n", "AAPL"),  # trailing newline is surrounding whitespace
            ("BRK.B", "BRK.B"),
            ("SPY", "SPY"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert ticker(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "AA PL", "AAPL!", "AA\nPL", "A" * 17])
    def test_rejects_malformed(self, raw: str) -> None:
        with pytest.raises(InvariantViolation):
            ticker(raw)


class TestSymbol:
    def test_builds_a_valid_symbol(self) -> None:
        symbol = Symbol(
            ticker=ticker("AAPL"),
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            exchange="XNAS",
            currency="USD",
        )
        assert symbol.ticker == "AAPL"
        assert symbol.active

    def test_rejects_non_usd_for_mvp(self) -> None:
        # ADR 0005 D5: currency is carried, not assumed, but MVP asserts USD on ingest.
        with pytest.raises(InvariantViolation, match="USD"):
            Symbol(
                ticker=ticker("AAPL"),
                name="Apple Inc.",
                asset_type=AssetType.STOCK,
                exchange="XNAS",
                currency="EUR",
            )

    def test_requires_an_exchange(self) -> None:
        with pytest.raises(InvariantViolation, match="exchange"):
            Symbol(
                ticker=ticker("AAPL"),
                name="Apple Inc.",
                asset_type=AssetType.STOCK,
                exchange="",
                currency="USD",
            )

    def test_provider_symbol_falls_back_to_the_ticker(self) -> None:
        symbol = Symbol(
            ticker=ticker("AAPL"),
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            exchange="XNAS",
            currency="USD",
        )
        assert symbol.provider_symbol("alpaca") == "AAPL"

    def test_provider_mapping_overrides_the_ticker(self) -> None:
        symbol = Symbol(
            ticker=ticker("BRK.B"),
            name="Berkshire Hathaway Inc. Class B",
            asset_type=AssetType.STOCK,
            exchange="XNYS",
            currency="USD",
            provider_mapping={"alpaca": "BRK.B", "ibkr": "BRK B"},
        )
        assert symbol.provider_symbol("ibkr") == "BRK B"
        assert symbol.provider_symbol("alpaca") == "BRK.B"

    def test_identity_is_the_ticker(self) -> None:
        def build(name: str) -> Symbol:
            return Symbol(
                ticker=ticker("AAPL"),
                name=name,
                asset_type=AssetType.STOCK,
                exchange="XNAS",
                currency="USD",
            )

        assert build("Apple Inc.") == build("Apple Incorporated")
        assert len({build("a"), build("b")}) == 1
