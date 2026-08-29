"""The strategy language: parsing, containment, validation, identity.

The DSL is the security boundary between "an LLM proposed something" and "code
ran". These tests are mostly about what the grammar *refuses*, because that is
the property the containment argument rests on.
"""

from __future__ import annotations

import pytest

from aqr.dsl.expr import (
    Binary,
    Call,
    Compare,
    Logic,
    Number,
    ParseError,
    Unary,
    evaluate,
    parse,
)
from aqr.dsl.loader import dumps, loads
from aqr.dsl.schema import StrategySpec, Universe
from aqr.dsl.validator import validate, validate_against
from aqr.features.engine import FeatureFrame
from tests.conftest import make_bars


class TestParsing:
    def test_bare_field_is_a_feature_call(self) -> None:
        assert parse("close") == Call("close", ())

    def test_comparison(self) -> None:
        node = parse("close > ema(200)")
        assert isinstance(node, Compare)
        assert node.op == ">"
        assert node.right == Call("ema", (200.0,))

    def test_arithmetic_precedence(self) -> None:
        # 1 + 2 * 3 must parse as 1 + (2 * 3)
        node = parse("close <= ema(20) * 1.01 + 1")
        assert isinstance(node, Compare)
        assert isinstance(node.right, Binary) and node.right.op == "+"
        assert isinstance(node.right.left, Binary) and node.right.left.op == "*"

    def test_and_binds_tighter_than_or(self) -> None:
        node = parse("close > 1 or close > 2 and close > 3")
        assert isinstance(node, Logic) and node.op == "or"
        assert isinstance(node.right, Logic) and node.right.op == "and"

    def test_parentheses_override_precedence(self) -> None:
        node = parse("(close > 1 or close > 2) and close > 3")
        assert isinstance(node, Logic) and node.op == "and"

    def test_negative_literals(self) -> None:
        """Unary minus in an expression stays a node; only feature arguments fold.

        Keeping the sign separate is what lets parameter perturbation rewrite the
        magnitude ``0.02`` without also having to reason about the sign.
        """
        node = parse("roc(1) > -0.02")
        assert isinstance(node.right, Unary)  # type: ignore[attr-defined]
        assert node.right.operand == Number(0.02)  # type: ignore[attr-defined]
        bars = make_bars(closes=[100.0, 90.0, 100.0, 100.0, 100.0, 100.0])
        fired = evaluate(node, FeatureFrame(bars))
        assert fired[1] == False  # noqa: E712 - a -10% bar is below -2%
        assert fired[3] == True  # noqa: E712 - a flat bar is above it

    def test_round_trips_through_str(self) -> None:
        source = "close <= ema(20) * 1.01"
        assert str(parse(source)) == source


class TestContainment:
    """What the grammar cannot express is the whole safety argument."""

    @pytest.mark.parametrize(
        "source",
        [
            "__import__('os').system('rm -rf /')",
            "os.system('x')",
            "close.__class__",
            "close[0]",
            "lambda x: x",
            "exec('1')",
            "eval('1')",
            "close = 5",
            "open('/etc/passwd')",
            "close ; drop table strategies",
        ],
    )
    def test_code_shaped_input_is_rejected(self, source: str) -> None:
        with pytest.raises(ParseError):
            parse(source)

    def test_unknown_feature_is_rejected_with_a_suggestion(self) -> None:
        with pytest.raises(ParseError, match="Did you mean"):
            parse("emaa(200) > 1")

    def test_too_many_arguments_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="at most"):
            parse("ema(20, 30, 40) > 1")

    def test_feature_argument_must_be_a_literal(self) -> None:
        """``ema(rsi(14))`` would make warm-up depend on the data itself."""
        with pytest.raises(ParseError):
            parse("ema(rsi(14)) > 1")

    def test_empty_expression_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            parse("   ")

    def test_unbalanced_parenthesis_is_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse("(close > 1")


class TestEvaluation:
    def test_nan_compares_false_in_both_directions(self) -> None:
        """A strategy must be silent during its own warm-up."""
        bars = make_bars(closes=[100.0] * 10)
        frame = FeatureFrame(bars)
        greater = evaluate(parse("ema(200) > 1"), frame)
        less = evaluate(parse("ema(200) < 1"), frame)
        assert not greater.any(), "an unwarmed indicator produced a signal"
        assert not less.any(), "NaN compared true in the other direction"

    def test_division_by_zero_yields_nan_not_an_exception(self) -> None:
        bars = make_bars(closes=[100.0] * 5)
        result = evaluate(parse("close / 0"), FeatureFrame(bars))
        assert all(x != x for x in result)  # NaN

    def test_a_number_cannot_be_used_as_a_condition(self) -> None:
        """``rsi(14) and x`` is a mistake, not "RSI is non-zero"."""
        bars = make_bars(closes=[100.0] * 5)
        with pytest.raises(ParseError, match="not a condition"):
            evaluate(parse("rsi(14) and close > 1"), FeatureFrame(bars))


class TestSpec:
    def test_yaml_round_trip_preserves_the_fingerprint(self, spec: StrategySpec) -> None:
        assert loads(dumps(spec)).fingerprint() == spec.fingerprint()

    def test_fingerprint_ignores_the_name(self, spec: StrategySpec) -> None:
        """Renaming a strategy must not make it look like a new discovery."""
        renamed = spec.with_params(name="something_else", hypothesis="reworded")
        assert renamed.fingerprint() == spec.fingerprint()

    def test_fingerprint_tracks_behaviour(self, spec: StrategySpec) -> None:
        changed = spec.with_params(entry="close <= ema(20) * 1.02 and rsi(14) > 40")
        assert changed.fingerprint() != spec.fingerprint()

    def test_fingerprint_ignores_symbol_order(self, spec: StrategySpec) -> None:
        a = spec.with_params(universe=Universe(symbols=("SPY", "QQQ")))
        b = spec.with_params(universe=Universe(symbols=("QQQ", "SPY")))
        assert a.fingerprint() == b.fingerprint()

    def test_unknown_field_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            loads(
                """
                strategy:
                  name: typo
                  entry: close > 1
                  universe: {symbols: [SPY]}
                  stop_los: {multiplier: 2}
                """
            )

    def test_misspelled_nested_field_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            loads(
                """
                strategy:
                  name: typo
                  entry: close > 1
                  universe: {symbols: [SPY]}
                  exit:
                    stop_loss: {type: atr, multiplyer: 2}
                """
            )

    def test_features_includes_the_stop_loss_atr(self, spec: StrategySpec) -> None:
        assert any(str(k) == "atr(14)" for k in spec.features())

    def test_out_of_range_risk_is_rejected(self, spec: StrategySpec) -> None:
        from aqr.dsl.schema import Sizing

        with pytest.raises(ValueError, match="risk_per_trade"):
            spec.with_params(sizing=Sizing(risk_per_trade=0.9))

    def test_empty_universe_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="symbols"):
            Universe(symbols=())


class TestValidator:
    def test_entry_must_be_a_condition(self, spec: StrategySpec) -> None:
        report = validate(spec.with_params(entry="ema(20)"))
        assert not report.ok
        assert "condition" in report.errors[0]

    def test_high_risk_warns_but_does_not_block(self, spec: StrategySpec) -> None:
        from aqr.dsl.schema import Sizing

        report = validate(spec.with_params(sizing=Sizing(risk_per_trade=0.05)))
        assert report.ok
        assert any("exceeds 2%" in w for w in report.warnings)

    def test_a_strategy_that_never_fires_is_rejected(self, spy) -> None:
        never = StrategySpec(
            name="impossible",
            entry="rsi(14) > 101",
            universe=Universe(symbols=("SPY",)),
        )
        report = validate_against(never, spy)
        assert not report.ok
        assert "never fires" in report.errors[0]

    def test_insufficient_history_is_rejected(self) -> None:
        short = make_bars(closes=[100.0] * 50, symbol="SPY")
        spec = StrategySpec(
            name="needs_history",
            entry="close > ema(200)",
            universe=Universe(symbols=("SPY",)),
        )
        report = validate_against(spec, short)
        assert not report.ok
        assert "warm up" in report.errors[0]

    def test_always_on_entry_warns_about_buy_and_hold(self, spy) -> None:
        always = StrategySpec(
            name="basically_long",
            entry="close > 0",
            universe=Universe(symbols=("SPY",)),
        )
        report = validate_against(always, spy)
        assert any("buy-and-hold" in w for w in report.warnings)
