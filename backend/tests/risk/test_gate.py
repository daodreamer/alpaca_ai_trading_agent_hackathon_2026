"""03 D1, D3 and D6 — the Gate as a whole.

The checks are pinned individually in `test_checks.py`. This file is about the
four properties that make the Gate a gate rather than a function that returns
opinions:

* it is total — every check runs, and the tape comes back whole;
* it is unforgeable — a `GatedOrder` exists only because `evaluate` made one;
* it never blocks an exit;
* it is deterministic.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphagate.core.errors import InvariantViolation
from alphagate.risk import (
    Approved,
    GatedOrder,
    Intent,
    RiskLimits,
    Vetoed,
    VetoReason,
    evaluate,
)
from alphagate.risk.checks import CHECKS
from tests.risk.conftest import (
    MSFT,
    NOW,
    make_risk,
    position,
    proposal,
    snapshot,
    spread_risk,
)

CHECK_COUNT = len(CHECKS)


class TestApproval:
    def test_a_clean_proposal_is_approved(self, limits: RiskLimits) -> None:
        verdict = evaluate(proposal(), snapshot(), limits, NOW)
        assert isinstance(verdict, Approved)
        assert all(check.passed for check in verdict.checks)

    def test_the_order_carries_the_domain_sign_convention(self, limits: RiskLimits) -> None:
        """A credit is positive here. The flip to Alpaca's convention happens in
        exactly one named function in `execution` — specs/04 D2."""
        verdict = evaluate(proposal(), snapshot(), limits, NOW)
        assert isinstance(verdict, Approved)
        assert verdict.order.limit_price == Decimal(150)
        assert verdict.order.limit_price > 0

    def test_the_order_carries_what_execution_needs(self, limits: RiskLimits) -> None:
        request = proposal(quantity=2, proposal_id="p-4242")
        verdict = evaluate(request, snapshot(), limits, NOW)
        assert isinstance(verdict, Approved)
        assert verdict.order.quantity == 2
        assert verdict.order.proposal_id == "p-4242"
        assert verdict.order.intent is Intent.OPEN
        assert verdict.order.structure is request.structure

    def test_approval_is_stamped_with_the_supplied_time(self, limits: RiskLimits) -> None:
        """`approved_at` is the argument, never a clock read."""
        later = NOW + timedelta(hours=3)
        verdict = evaluate(
            proposal(risk=spread_risk(as_of=later), risk_as_of=later), snapshot(), limits, later
        )
        assert isinstance(verdict, Approved)
        assert verdict.order.approved_at == later


class TestVeto:
    def test_a_veto_names_the_check(self, limits: RiskLimits) -> None:
        verdict = evaluate(proposal(quantity=3), snapshot(), limits, NOW)
        assert isinstance(verdict, Vetoed)
        assert [reason.check for reason in verdict.reasons] == ["per_trade_loss"]

    def test_every_check_runs_even_after_the_first_veto(self, limits: RiskLimits) -> None:
        """D3: the Gate does not short-circuit. The tape length is constant."""
        clean = evaluate(proposal(), snapshot(), limits, limits and NOW)
        broken = evaluate(
            proposal(risk=make_risk(days_to_expiry=0, worst_spread_pct=Decimal("0.40")),
                     quantity=9),
            snapshot(drawdown="0.20", fills_today=99),
            limits,
            NOW,
        )
        assert len(clean.checks) == CHECK_COUNT
        assert len(broken.checks) == CHECK_COUNT
        assert [c.name for c in clean.checks] == [c.name for c in broken.checks]

    def test_a_veto_with_five_reasons_says_five(self, limits: RiskLimits) -> None:
        """"A veto with one reason and a veto with five are different
        situations and the journal should show which." — D3."""
        verdict = evaluate(
            proposal(risk=make_risk(days_to_expiry=0, worst_spread_pct=Decimal("0.40")),
                     quantity=9),
            snapshot(drawdown="0.20", fills_today=99),
            limits,
            NOW,
        )
        assert isinstance(verdict, Vetoed)
        names = {reason.check for reason in verdict.reasons}
        assert names == {
            "per_trade_loss",
            "underlying_concentration",
            "liquidity",
            "expiry_window",
            "drawdown_killswitch",
            "daily_trade_cap",
        }

    def test_reasons_follow_the_declared_check_order(self, limits: RiskLimits) -> None:
        verdict = evaluate(
            proposal(risk=make_risk(days_to_expiry=0, worst_spread_pct=Decimal("0.40"))),
            snapshot(fills_today=99),
            limits,
            NOW,
        )
        assert isinstance(verdict, Vetoed)
        declared = [c.name for c in verdict.checks]
        positions = [declared.index(r.check) for r in verdict.reasons]
        assert positions == sorted(positions)

    def test_a_veto_carries_the_full_tape_not_just_the_failures(
        self, limits: RiskLimits
    ) -> None:
        verdict = evaluate(proposal(quantity=3), snapshot(), limits, NOW)
        assert isinstance(verdict, Vetoed)
        assert len(verdict.checks) == CHECK_COUNT
        assert sum(1 for c in verdict.checks if c.passed) == CHECK_COUNT - 1

    def test_an_empty_veto_is_not_constructible(self) -> None:
        with pytest.raises(InvariantViolation, match="no reason"):
            Vetoed(reasons=(), checks=())


class TestUnforgeability:
    """D3 — `execution` accepts a `GatedOrder`, and only the Gate makes one."""

    def test_a_gated_order_cannot_be_constructed_by_hand(self) -> None:
        with pytest.raises(InvariantViolation, match="only be constructed inside"):
            GatedOrder(
                structure=proposal().structure,
                quantity=1,
                intent=Intent.OPEN,
                limit_price=Decimal(150),
                approved_at=NOW,
                proposal_id="forged",
            )

    def test_approval_cannot_be_constructed_by_hand(self, limits: RiskLimits) -> None:
        verdict = evaluate(proposal(), snapshot(), limits, NOW)
        assert isinstance(verdict, Approved)
        with pytest.raises(InvariantViolation, match="only be constructed inside"):
            Approved(order=verdict.order, checks=())

    def test_an_approved_order_cannot_be_cloned_into_a_second_one(
        self, limits: RiskLimits
    ) -> None:
        """`replace` is the obvious way to turn one fill into two. It raises."""
        verdict = evaluate(proposal(), snapshot(), limits, NOW)
        assert isinstance(verdict, Approved)
        with pytest.raises(InvariantViolation, match="only be constructed inside"):
            replace(verdict.order, quantity=99)

    def test_there_is_no_override_on_the_public_surface(self) -> None:
        """No `force`, no bypass, no debug switch — specs/01 Rule 2."""
        import inspect

        import alphagate.risk as risk_package

        parameters = set(inspect.signature(evaluate).parameters)
        assert parameters == {"proposal", "portfolio", "limits", "as_of"}
        banned = {"force", "bypass", "override", "skip_checks", "unsafe"}
        assert not banned & {name.lower() for name in dir(risk_package)}


class TestKillSwitch:
    """D4 — trips at the threshold, blocks opens, permits closes, stays tripped."""

    def test_it_trips_at_the_threshold(self, limits: RiskLimits) -> None:
        verdict = evaluate(proposal(), snapshot(drawdown="0.05"), limits, NOW)
        assert isinstance(verdict, Vetoed)
        assert any(r.check == "drawdown_killswitch" for r in verdict.reasons)

    def test_it_blocks_a_roll_too(self, limits: RiskLimits) -> None:
        """A roll opens new risk, so it is an open for budgeting purposes."""
        verdict = evaluate(
            proposal(intent=Intent.ROLL), snapshot(drawdown="0.06"), limits, NOW
        )
        assert isinstance(verdict, Vetoed)

    def test_it_never_blocks_an_exit(self, limits: RiskLimits) -> None:
        tripped = snapshot(drawdown="0.40", killswitch_tripped=True, fills_today=99)
        verdict = evaluate(proposal(intent=Intent.CLOSE), tripped, limits, NOW)
        assert isinstance(verdict, Approved)

    def test_it_stays_tripped_after_a_recovery(self, limits: RiskLimits) -> None:
        recovered = snapshot(drawdown="0.001", killswitch_tripped=True)
        verdict = evaluate(proposal(), recovered, limits, NOW)
        assert isinstance(verdict, Vetoed)
        assert any(r.check == "drawdown_killswitch" for r in verdict.reasons)

    def test_re_arming_is_the_caller_clearing_the_latch(self, limits: RiskLimits) -> None:
        """The Gate is pure, so the latch is state the caller carries in.
        Re-arming is a human editing that state, not a method on the Gate."""
        assert isinstance(
            evaluate(proposal(), snapshot(drawdown="0.001"), limits, NOW), Approved
        )


class TestExitsAreNeverBlocked:
    """D4's closing sentence outranks every budget in the file."""

    @pytest.fixture
    def hostile(self) -> object:
        return snapshot(
            positions=tuple(position(max_loss="9000") for _ in range(9)),
            drawdown="0.60",
            fills_today=400,
            killswitch_tripped=True,
        )

    def test_a_close_survives_a_book_that_breaks_every_limit(
        self, limits: RiskLimits, hostile: object
    ) -> None:
        verdict = evaluate(
            proposal(
                risk=make_risk(
                    days_to_expiry=0,
                    worst_spread_pct=Decimal("0.90"),
                    quote_age_seconds=3600.0,
                    net_greeks=None,
                ),
                quantity=50,
                intent=Intent.CLOSE,
            ),
            hostile,  # type: ignore[arg-type]
            limits,
            NOW,
        )
        assert isinstance(verdict, Approved)

    def test_the_waiver_is_visible_in_the_tape(
        self, limits: RiskLimits, hostile: object
    ) -> None:
        """A waived check still shows what it measured. Silence would be worse
        than a veto: the dashboard has to be able to say the exit went out over
        a 90% spread."""
        verdict = evaluate(
            proposal(risk=make_risk(worst_spread_pct=Decimal("0.90")), intent=Intent.CLOSE),
            hostile,  # type: ignore[arg-type]
            limits,
            NOW,
        )
        liquidity = next(c for c in verdict.checks if c.name == "liquidity")
        assert liquidity.passed
        assert liquidity.observed == Decimal("0.90")
        assert "never blocks an exit" in liquidity.detail

    def test_an_open_on_the_same_book_is_refused(
        self, limits: RiskLimits, hostile: object
    ) -> None:
        """The waiver is scoped to exits, not switched on globally."""
        verdict = evaluate(proposal(), hostile, limits, NOW)  # type: ignore[arg-type]
        assert isinstance(verdict, Vetoed)


class TestDeterminism:
    """D6 — same inputs, same verdict, including the order of the tape."""

    def test_one_hundred_evaluations_agree(self, limits: RiskLimits) -> None:
        book = snapshot(positions=(position(max_loss="1200"),), fills_today=3)
        request = proposal(quantity=2)
        first = evaluate(request, book, limits, NOW)
        assert all(evaluate(request, book, limits, NOW) == first for _ in range(100))

    def test_a_veto_is_reproducible_reason_for_reason(self, limits: RiskLimits) -> None:
        request = proposal(quantity=9)
        book = snapshot(drawdown="0.20", fills_today=99)
        a = evaluate(request, book, limits, NOW)
        b = evaluate(request, book, limits, NOW)
        assert isinstance(a, Vetoed)
        assert isinstance(b, Vetoed)
        assert a.reasons == b.reasons
        assert a.checks == b.checks

    @settings(max_examples=200, deadline=None)
    @given(
        quantity=st.integers(min_value=1, max_value=12),
        fills=st.integers(min_value=0, max_value=30),
        drawdown=st.decimals(
            min_value=Decimal(0), max_value=Decimal("0.5"), places=4, allow_nan=False
        ),
        dte=st.integers(min_value=-5, max_value=60),
        open_positions=st.integers(min_value=0, max_value=10),
        intent=st.sampled_from(list(Intent)),
    )
    def test_evaluate_is_a_pure_function(
        self,
        quantity: int,
        fills: int,
        drawdown: Decimal,
        dte: int,
        open_positions: int,
        intent: Intent,
    ) -> None:
        limits_ = __import__(
            "alphagate.risk", fromlist=["DEFAULT_LIMITS"]
        ).DEFAULT_LIMITS
        request = proposal(
            risk=make_risk(days_to_expiry=dte), quantity=quantity, intent=intent
        )
        book = snapshot(
            positions=tuple(position(underlying=MSFT) for _ in range(open_positions)),
            drawdown=str(drawdown),
            fills_today=fills,
        )
        assert evaluate(request, book, limits_, NOW) == evaluate(request, book, limits_, NOW)

    def test_the_time_is_an_argument_and_it_matters(self, limits: RiskLimits) -> None:
        """No clock read anywhere: a later `as_of` is the only thing that can make
        the same proposal age out.

        This is the LLM-latency case. The agent perceives, computes risk, and then
        spends time in a model call before the Gate sees the result. The proposal
        is byte-identical in both evaluations below; only the hour differs.
        """
        request = proposal(risk=spread_risk(age=0))
        assert isinstance(evaluate(request, snapshot(), limits, NOW), Approved)
        late = evaluate(request, snapshot(), limits, NOW + timedelta(minutes=5))
        assert isinstance(late, Vetoed)
        assert [r.check for r in late.reasons] == ["fresh_quotes"]

    def test_a_naive_timestamp_is_refused(self, limits: RiskLimits) -> None:
        with pytest.raises(InvariantViolation, match="tz-aware"):
            evaluate(proposal(), snapshot(), limits, datetime(2026, 9, 1, 15, 30))  # noqa: DTZ001

    def test_utc_is_utc_whatever_offset_it_arrives_in(self, limits: RiskLimits) -> None:
        elsewhere = NOW.astimezone(__import__("datetime").timezone(timedelta(hours=9)))
        a = evaluate(proposal(), snapshot(), limits, NOW)
        b = evaluate(proposal(), snapshot(), limits, elsewhere)
        assert isinstance(a, Approved)
        assert isinstance(b, Approved)
        assert a.order.approved_at == b.order.approved_at


def test_veto_reasons_are_values() -> None:
    assert VetoReason("liquidity", "too wide") == VetoReason("liquidity", "too wide")


def test_the_gate_reads_no_clock() -> None:
    """Guard, not a unit test: `risk/` must not contain a `now()` call at all.

    A single `datetime.now()` in this layer would make backtest and live diverge
    while every test above still passed, because the tests supply a time the code
    would then ignore.
    """
    import ast
    from pathlib import Path

    risk_root = Path(evaluate.__code__.co_filename).parent
    offenders: list[str] = []
    for path in sorted(risk_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"now", "utcnow", "today"}:
                offenders.append(f"{path.name}:{node.lineno} reads a clock")
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None)
                if name in {"time", "monotonic"}:
                    offenders.append(f"{path.name}:{node.lineno} reads a clock")
    assert not offenders, "\n".join(offenders)


def test_the_gate_never_calls_a_model() -> None:
    """Rule 2 of CLAUDE.md, restated where it is easiest to violate.

    `tests/test_boundaries.py` checks the whole tree; this is the copy that fails
    with the Gate's own name on it.
    """
    import ast
    from pathlib import Path

    risk_root = Path(evaluate.__code__.co_filename).parent
    banned = {"anthropic", "openai", "httpx", "requests"}
    offenders: list[str] = []
    for path in sorted(risk_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.name} imports {a.name}"
                    for a in node.names
                    if a.name.split(".")[0] in banned
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in banned
            ):
                offenders.append(f"{path.name} imports {node.module}")
    assert not offenders, "a Gate that calls a model is not a Gate:\n" + "\n".join(offenders)


@pytest.mark.parametrize("intent", list(Intent))
def test_the_verdict_is_one_of_exactly_two_things(
    limits: RiskLimits, intent: Intent
) -> None:
    verdict = evaluate(proposal(intent=intent), snapshot(), limits, NOW)
    assert isinstance(verdict, Approved | Vetoed)
    assert verdict.is_approved is isinstance(verdict, Approved)


def test_utc_fixture_is_utc() -> None:
    assert NOW.tzinfo is UTC
