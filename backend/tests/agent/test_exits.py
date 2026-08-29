"""05 D8 — exits are deterministic and are not the model's decision.

The asymmetry with `test_deepseek.py` is the point of this file. Entering gets a
model, a prompt and a menu; leaving gets three thresholds and a fixed order. A
model asked whether to close a losing position will find a reason to wait, every
time, and it will be articulate about it — so the defence is that there is no
prompt, and this file is where that is checked rather than asserted.

Numbers are hand-computed against a spread opened for a $60 credit on one
contract, which is the position the live account actually holds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from alphagate.agent.exits import (
    DEFAULT_EXIT_POLICY,
    ExitDecision,
    ExitPolicy,
    ExitRule,
    evaluate_exit,
)
from alphagate.core.errors import InvariantViolation
from alphagate.options import Greeks, StructureRisk
from alphagate.risk import Intent, OpenPosition
from tests.agent.conftest import NOW, contract, put_chain

ENTRY_CREDIT = Decimal(60)
FLAT = Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.20)


def position(quantity: int = 1) -> OpenPosition:
    from alphagate.options import Leg, OptionStructure, Side, StructureKind

    short, long = contract("752"), contract("747")
    return OpenPosition(
        structure=OptionStructure(
            StructureKind.VERTICAL_CREDIT, (Leg(short, Side.SELL), Leg(long, Side.BUY))
        ),
        quantity=quantity,
        max_loss=Decimal(440),
        net_greeks=FLAT,
        opened_at=NOW - timedelta(days=2),
    )


def mark(net_premium: str, *, dte: int = 9) -> StructureRisk:
    """What it would cost to be flat now, in the domain's convention."""
    return StructureRisk(
        net_premium=Decimal(net_premium),
        max_loss=Decimal(440),
        max_profit=Decimal(60),
        breakevens=(),
        net_greeks=FLAT,
        worst_spread_pct=Decimal("0.02"),
        days_to_expiry=dte,
        quote_age_seconds=0.0,
    )


def decide(
    net_premium: str, *, dte: int = 9, quantity: int = 1, **kwargs: Any
) -> ExitDecision:
    return evaluate_exit(
        position(quantity),
        mark(net_premium, dte=dte),
        ENTRY_CREDIT,
        as_of=NOW,
        policy=ExitPolicy(**kwargs) if kwargs else DEFAULT_EXIT_POLICY,
    )


class TestProfitTarget:
    def test_half_the_credit_earned_closes(self) -> None:
        """Sold for 60, buy it back for 30: half the premium is realised."""
        decision = decide("30")
        assert decision.rule is ExitRule.PROFIT_TARGET
        assert decision.should_close
        assert decision.unrealised == Decimal(30)
        assert decision.fraction_of_credit == Decimal("0.5")

    def test_just_short_of_the_target_holds(self) -> None:
        decision = decide("30.60")  # 49% earned
        assert decision.rule is ExitRule.HOLD

    def test_more_than_the_target_still_closes(self) -> None:
        assert decide("5").rule is ExitRule.PROFIT_TARGET

    def test_it_scales_with_quantity(self) -> None:
        """The fraction is per-unit; the cash is not."""
        decision = decide("30", quantity=3)
        assert decision.unrealised == Decimal(90)
        assert decision.fraction_of_credit == Decimal("0.5")

    def test_the_target_is_configurable_but_not_absent(self) -> None:
        assert decide("30", profit_target=Decimal("0.75")).rule is ExitRule.HOLD
        assert decide("15", profit_target=Decimal("0.75")).rule is ExitRule.PROFIT_TARGET


class TestStop:
    def test_twice_the_credit_lost_closes(self) -> None:
        """Sold for 60, now costs 180 to close: down 120, which is 2x."""
        decision = decide("180")
        assert decision.rule is ExitRule.STOP
        assert decision.unrealised == Decimal(-120)

    def test_just_short_of_the_stop_holds(self) -> None:
        assert decide("179").rule is ExitRule.HOLD

    def test_a_deeper_loss_still_stops(self) -> None:
        assert decide("400").rule is ExitRule.STOP

    def test_profit_wins_a_tie_with_stop(self) -> None:
        """A mark that somehow satisfies both — a crossed quote, a wild print —
        is read the safe way round. Taking a profit that is not there costs a
        commission; taking a loss that is not there costs the loss."""
        both = ExitPolicy(profit_target=Decimal("0.10"), stop_multiple=Decimal("0.05"))
        assert decide("54", profit_target=both.profit_target,
                      stop_multiple=both.stop_multiple).rule is ExitRule.PROFIT_TARGET


class TestDteClose:
    def test_it_closes_at_the_floor(self) -> None:
        """Gamma into expiry: a spread that behaved for two weeks can lose its
        whole width in an afternoon."""
        decision = decide("45", dte=2)
        assert decision.rule is ExitRule.DTE_CLOSE
        assert "gamma" in decision.detail

    def test_it_closes_past_the_floor(self) -> None:
        assert decide("45", dte=0).rule is ExitRule.DTE_CLOSE

    def test_it_holds_above_the_floor(self) -> None:
        assert decide("45", dte=3).rule is ExitRule.HOLD

    def test_profit_and_stop_are_checked_first(self) -> None:
        """A position at its target on its last day closes for the profit, and
        the journal says so — the reason a trade closed is data."""
        assert decide("30", dte=1).rule is ExitRule.PROFIT_TARGET
        assert decide("180", dte=1).rule is ExitRule.STOP


class TestHold:
    def test_a_quiet_position_is_left_alone(self) -> None:
        decision = decide("55", dte=9)
        assert decision.rule is ExitRule.HOLD
        assert not decision.should_close

    def test_holding_still_reports_the_numbers(self) -> None:
        """The dashboard shows a position approaching its target, not just the
        moment it crosses."""
        decision = decide("36", dte=9)
        assert decision.fraction_of_credit == Decimal("0.4")
        assert decision.days_to_expiry == 9


class TestDebitStructures:
    def test_a_debit_has_no_fraction_of_the_credit(self) -> None:
        """"A number that would read as a percentage of something that does not
        exist." """
        decision = evaluate_exit(
            position(), mark("-100"), Decimal(-236), as_of=NOW
        )
        assert decision.fraction_of_credit is None

    def test_the_rules_still_fire_on_the_debit_paid(self) -> None:
        """Bought for 236, now worth 380 to close: up 144, 61% of the debit."""
        decision = evaluate_exit(
            position(), mark("-380"), Decimal(-236), as_of=NOW
        )
        assert decision.rule is ExitRule.PROFIT_TARGET
        assert decision.unrealised == Decimal(144)


class TestTheModelHasNoVote:
    def test_the_signature_takes_no_proposer(self) -> None:
        """The structural claim. There is nowhere to pass a model."""
        import inspect

        parameters = set(inspect.signature(evaluate_exit).parameters)
        assert parameters == {"position", "current", "entry_premium", "as_of", "policy"}

    def test_the_module_imports_no_model(self) -> None:
        import ast
        from pathlib import Path

        source = Path(evaluate_exit.__code__.co_filename).read_text(encoding="utf-8")
        names = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not names & {"anthropic", "openai", "httpx"}
        assert "deepseek" not in source.lower()

    def test_an_exit_is_always_a_close(self) -> None:
        assert decide("30").intent is Intent.CLOSE

    def test_it_reads_no_clock(self) -> None:
        """`as_of` is an argument, so a replayed exit and a live one differ in
        one parameter and nothing else."""
        early = evaluate_exit(position(), mark("30"), ENTRY_CREDIT, as_of=NOW)
        late = evaluate_exit(
            position(), mark("30"), ENTRY_CREDIT, as_of=NOW + timedelta(days=30)
        )
        assert early == late


class TestPolicyInvariants:
    @pytest.mark.parametrize("bad", [Decimal(0), Decimal("-0.5"), Decimal("1.5")])
    def test_the_profit_target_is_a_fraction(self, bad: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="profit_target"):
            ExitPolicy(profit_target=bad)

    def test_the_stop_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="stop_multiple"):
            ExitPolicy(stop_multiple=Decimal(0))

    def test_the_dte_floor_must_not_be_negative(self) -> None:
        with pytest.raises(InvariantViolation, match="min_dte"):
            ExitPolicy(min_dte=-1)

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="tz-aware"):
            evaluate_exit(
                position(),
                mark("30"),
                ENTRY_CREDIT,
                as_of=datetime(2026, 8, 26, 14, 30),  # noqa: DTZ001
            )

    def test_the_defaults_are_the_documented_ones(self) -> None:
        assert DEFAULT_EXIT_POLICY.profit_target == Decimal("0.50")
        assert DEFAULT_EXIT_POLICY.stop_multiple == Decimal("2.0")
        assert DEFAULT_EXIT_POLICY.min_dte == 2


def test_the_chain_fixture_still_prices_the_position() -> None:
    """Anchor: the strikes this file names are the ones the suite trades."""
    assert contract("752") in put_chain()
    assert contract("747") in put_chain()


def test_utc_anchor() -> None:
    assert NOW.tzinfo is UTC
