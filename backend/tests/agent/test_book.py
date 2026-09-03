"""The open book — `agent/book.py`.

The module exists because Alpaca has no concept of a structure and the Gate
budgets in nothing else. Everything worth testing here is about the seam between
those two facts, and the failure mode is always the same shape: a book that
looks tidy and is false.

The case that matters most is the one at the bottom. A short leg at the broker
that no journalled fill accounts for must **not** be quietly absorbed into the
risk model — an agent that did that would be reporting a defined risk it had not
defined, which is the exact claim this whole project is built on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent.book import (
    contract_from,
    fills_on,
    open_positions,
    read_book,
    structure_from,
    working_closes,
)
from alphagate.core.errors import InvariantViolation
from alphagate.equity.policy import EQUITY_SLEEVE_ALLOCATION
from alphagate.execution import AccountRead, LegPosition
from alphagate.journal import Journal, outcome_from
from alphagate.options import OptionContract, Right, Side, StructureKind
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION, SLEEVE_LIMITS
from tests.agent.conftest import EXPIRY, SPY
from tests.journal.conftest import submission

NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)


def account(equity: str = "100000") -> AccountRead:
    return AccountRead(
        equity=Decimal(equity),
        last_equity=Decimal(equity),
        buying_power=Decimal(equity),
        options_buying_power=Decimal(equity),
        options_level=3,
        cash=Decimal(equity),
        multiplier=1,
        is_blocked=False,
        envelope=None,
        observed_at=NOW,
    )


def closed_for(realised: Decimal) -> dict[str, object]:
    """A journal record for a round-trip that closed at `realised`.

    Only the outcome amendment matters here: `realised_pl` is the same field
    `interface/read.py` renders, so the dashboard's number and the kill switch's
    number have one source.
    """
    return {"cycle_id": "2026-08-31-000", "outcome": {"realised_pl": str(realised)}}


def contract(strike: str, right: Right = Right.PUT) -> OptionContract:
    return OptionContract(SPY, EXPIRY, Decimal(strike), right)


def leg_at(strike: str, quantity: int, right: Right = Right.PUT) -> LegPosition:
    return LegPosition(
        contract=contract(strike, right),
        quantity=quantity,
        average_price=Decimal("1.50"),
        market_value=Decimal("150"),
        unrealised=Decimal("10"),
    )


def structure_payload(
    short: str = "752", long: str = "747", right: str = "P"
) -> dict[str, object]:
    """A put credit spread as the journal writes it — specs/06 D2."""
    return {
        "kind": "vertical_credit",
        "cover": None,
        "legs": [
            {
                "contract": {
                    "underlying": "SPY",
                    "expiry": EXPIRY.isoformat(),
                    "strike": f"{long}.000",
                    "right": right,
                    "multiplier": 100,
                },
                "side": "buy",
                "quantity": 1,
            },
            {
                "contract": {
                    "underlying": "SPY",
                    "expiry": EXPIRY.isoformat(),
                    "strike": f"{short}.000",
                    "right": right,
                    "multiplier": 100,
                },
                "side": "sell",
                "quantity": 1,
            },
        ],
    }


def filled(
    cycle_id: str = "2026-08-26-SPY-000",
    *,
    quantity: int = 1,
    max_loss: str = "393.00",
    structure: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "as_of": NOW.isoformat(),
        "stage": "filled",
        "proposal": {
            "quantity": quantity,
            "structure": structure if structure is not None else structure_payload(),
            "risk": {
                "max_loss": max_loss,
                "net_greeks": {
                    "delta": 9.0,
                    "gamma": -0.45,
                    "theta": 1.2,
                    "vega": -3.0,
                    "rho": 0.1,
                    "iv": 0.19,
                },
            },
        },
    }


OPEN_LEGS = (leg_at("747", 1), leg_at("752", -1))


class TestDecodingAJournalledStructure:
    def test_it_round_trips_through_the_domain_type(self) -> None:
        structure = structure_from(structure_payload())
        assert structure.kind is StructureKind.VERTICAL_CREDIT
        assert structure.underlying == SPY
        assert {leg.side for leg in structure.legs} == {Side.BUY, Side.SELL}

    def test_strikes_come_back_as_decimal(self) -> None:
        assert contract_from(
            {
                "underlying": "SPY",
                "expiry": EXPIRY.isoformat(),
                "strike": "752.500",
                "right": "P",
            }
        ).strike == Decimal("752.500")

    def test_it_goes_back_through_the_invariants(self) -> None:
        """A journal line from before a rule tightened is not a licence to hold
        a position the rule now forbids."""
        naked = {"kind": "vertical_credit", "cover": None, "legs": [
            structure_payload()["legs"][1]  # type: ignore[index]
        ]}
        with pytest.raises(InvariantViolation):
            structure_from(naked)


class TestMatchingLegsToStructures:
    def test_a_journalled_fill_still_at_the_broker_is_an_open_position(self) -> None:
        positions, unexplained, closed = open_positions(OPEN_LEGS, [filled()])
        assert len(positions) == 1
        assert positions[0].position.underlying == SPY
        assert unexplained == ()
        assert closed == ()

    def test_max_loss_is_scaled_by_quantity(self) -> None:
        """`StructureRisk.max_loss` is per unit. Forgetting the multiply
        understates the book by exactly the position size."""
        positions, _, _ = open_positions(OPEN_LEGS, [filled(quantity=3)])
        assert positions[0].position.max_loss == Decimal("1179.00")

    def test_greeks_are_scaled_too(self) -> None:
        positions, _, _ = open_positions(OPEN_LEGS, [filled(quantity=2)])
        greeks = positions[0].position.net_greeks
        assert greeks is not None
        assert greeks.delta == pytest.approx(18.0)

    def test_a_fill_whose_legs_are_gone_is_closed_not_open(self) -> None:
        positions, unexplained, closed = open_positions((), [filled()])
        assert positions == ()
        assert unexplained == ()
        assert closed == ("2026-08-26-SPY-000",)

    def test_a_half_closed_structure_does_not_count_as_open(self) -> None:
        """One leg left is not the structure. Counting it as one would budget
        against a defined risk that no longer exists."""
        positions, unexplained, closed = open_positions((leg_at("752", -1),), [filled()])
        assert positions == ()
        assert closed == ("2026-08-26-SPY-000",)
        assert [leg.contract.strike for leg in unexplained] == [Decimal("752")]

    def test_a_leg_on_the_wrong_side_does_not_match(self) -> None:
        """A short put we believe we hold and the broker reports as long is not
        our position, and netting risk against it points the wrong way."""
        flipped = (leg_at("747", 1), leg_at("752", 1))
        positions, unexplained, _ = open_positions(flipped, [filled()])
        assert positions == ()
        assert len(unexplained) == 2

    def test_only_filled_cycles_are_considered(self) -> None:
        """A `VETOED` or `DRY_RUN` cycle never reached the broker, so its
        structure cannot be holding these legs."""
        for stage in ("vetoed", "dry_run", "declined", "submitted"):
            record = {**filled(), "stage": stage}
            positions, unexplained, _ = open_positions(OPEN_LEGS, [record])
            assert positions == (), stage
            assert len(unexplained) == 2, stage

    def test_each_leg_is_claimed_once(self) -> None:
        """Two cycles that opened the same structure, and one contract at the
        broker: they must not both claim it. That would double the book's risk
        against one real position. The broker's quantity is the arbiter — see
        `TestTheBrokersQuantityDecidesHowManyFit`."""
        positions, unexplained, closed = open_positions(
            OPEN_LEGS, [filled("2026-08-26-SPY-000"), filled("2026-08-26-SPY-001")]
        )
        assert len(positions) == 1
        assert closed == ("2026-08-26-SPY-000",), "the older one reads as closed"
        assert unexplained == ()

    def test_the_newest_fill_claims_the_legs(self) -> None:
        positions, _, _ = open_positions(
            OPEN_LEGS,
            [filled("2026-08-26-SPY-000", max_loss="100.00"),
             filled("2026-08-26-SPY-001", max_loss="393.00")],
        )
        assert positions[0].position.max_loss == Decimal("393.00")

    def test_an_undecodable_structure_does_not_stop_the_rest(self) -> None:
        broken = {**filled("2026-08-26-SPY-000"), "proposal": {"structure": {"kind": "nope"}}}
        positions, unexplained, _ = open_positions(
            OPEN_LEGS, [broken, filled("2026-08-26-SPY-001")]
        )
        assert len(positions) == 1
        assert unexplained == ()


class TestTheBrokersQuantityDecidesHowManyFit:
    """Two journalled 1-lots and two contracts at the broker are two positions.

    The rule this class pins down is the one the first version got wrong in the
    safe-looking direction. Legs used to be claimed whole: the newest fill took
    the contract and every older fill that wanted it was reported closed. With
    one contract at the broker that is right. With two — the same rule firing
    twice in a session, which is what a rule that trades a single underlying
    does — it read a two-spread book as one spread, so the Gate budgeted half
    the risk that was actually on and nothing anywhere said so.

    The asymmetry below is deliberate. A journalled size *larger* than the
    broker can cover still reads as open, because dropping a live position out
    of the book would take it out of the exit policy too; a broker quantity
    larger than the journal explains is reported as unexplained, because that
    is the direction where the surplus is risk nobody decided on.
    """

    def test_two_identical_spreads_are_both_open_when_the_broker_holds_two(self) -> None:
        legs = (leg_at("747", 2), leg_at("752", -2))
        positions, unexplained, closed = open_positions(
            legs, [filled("2026-08-26-SPY-000"), filled("2026-08-26-SPY-001")]
        )
        assert [item.cycle_id for item in positions] == [
            "2026-08-26-SPY-000",
            "2026-08-26-SPY-001",
        ]
        assert unexplained == ()
        assert closed == ()

    def test_the_gate_budgets_for_both(self) -> None:
        """The reason the count matters: `open_risk` is what the portfolio-loss
        check measures, and half of it is a licence to open a third."""
        legs = (leg_at("747", 2), leg_at("752", -2))
        book = read_book(
            account(), legs, [filled("2026-08-26-SPY-000"), filled("2026-08-26-SPY-001")]
        )
        assert book.snapshot.open_structures == 2
        assert book.snapshot.open_risk == Decimal("786.00")
        assert book.is_clean

    def test_the_older_copy_is_closed_when_the_broker_holds_only_one(self) -> None:
        """Unchanged from before: one contract cannot be two positions."""
        positions, unexplained, closed = open_positions(
            OPEN_LEGS, [filled("2026-08-26-SPY-000"), filled("2026-08-26-SPY-001")]
        )
        assert [item.cycle_id for item in positions] == ["2026-08-26-SPY-001"]
        assert closed == ("2026-08-26-SPY-000",)
        assert unexplained == ()

    def test_the_surplus_the_journal_cannot_explain_is_reported(self) -> None:
        """Three contracts at the broker, one 1-lot in the journal. The spread
        we opened is open; the other two are somebody else's doing and are
        reported rather than absorbed."""
        legs = (leg_at("747", 3), leg_at("752", -3))
        positions, unexplained, closed = open_positions(legs, [filled()])
        assert len(positions) == 1
        assert closed == ()
        assert [(leg.contract.strike, leg.quantity) for leg in unexplained] == [
            (Decimal("747"), 2),
            (Decimal("752"), -2),
        ]

    def test_the_surplus_carries_only_its_share_of_the_money(self) -> None:
        """A residual reported at the whole line's market value would overstate
        what is unaccounted for by exactly the part we *can* account for."""
        legs = (leg_at("747", 3), leg_at("752", -3))
        _, unexplained, _ = open_positions(legs, [filled()])
        assert unexplained[0].quantity == 2, "one of the three is accounted for"
        assert unexplained[0].market_value == Decimal("100.00")
        assert unexplained[0].unrealised == Decimal("6.67")
        assert unexplained[0].average_price == Decimal("1.50"), "a per-contract price"

    def test_an_untouched_leg_is_reported_exactly_as_it_arrived(self) -> None:
        """Nothing claimed, nothing scaled: the common case must not go through
        the arithmetic at all."""
        legs = (leg_at("747", 3), leg_at("752", -3))
        _, unexplained, _ = open_positions(legs, [])
        assert unexplained == legs

    def test_a_journalled_size_the_broker_cannot_cover_still_reads_as_open(self) -> None:
        """The other direction, and the deliberate asymmetry. A 3-lot in the
        journal against one contract at the broker is over-modelled — and that
        is the safe way to be wrong: the position stays in the Gate's risk
        model and stays under the exit policy, where dropping it would leave a
        live short leg nobody was managing."""
        positions, unexplained, closed = open_positions(OPEN_LEGS, [filled(quantity=3)])
        assert len(positions) == 1
        assert positions[0].position.quantity == 3
        assert unexplained == (), "claiming never leaves a negative remainder"
        assert closed == ()

    def test_a_second_copy_gets_nothing_once_the_broker_is_exhausted(self) -> None:
        positions, unexplained, closed = open_positions(
            OPEN_LEGS,
            [filled("2026-08-26-SPY-000"), filled("2026-08-26-SPY-001", quantity=3)],
        )
        assert [item.cycle_id for item in positions] == ["2026-08-26-SPY-001"]
        assert closed == ("2026-08-26-SPY-000",)
        assert unexplained == ()


class TestAFillThatArrivedAsAnAmendment:
    """The seam where a real spread went missing — the bug this pairing fixes.

    The cycle is journalled the moment the order is placed, when the broker
    still says `pending_new`, so its line reads `submitted`. The fill lands
    later and arrives as an amendment (specs/06 D3). Everything here depends on
    the two halves agreeing about what that means: `Journal.read` settles the
    stage from the outcome, and `open_positions` claims the legs of a cycle
    that reads as filled. Assert them together, because separately they were
    both "right" while a live two-legged spread sat at the broker labelled a
    leg the journal could not explain.
    """

    def journal_with_a_late_fill(self, tmp_path: Path) -> Journal:
        journal = Journal(directory=tmp_path / "journal")
        journal.append({**filled(), "stage": "submitted"})
        journal.record_outcome(
            outcome_from(
                submission("order_filled"),
                cycle_id="2026-08-26-SPY-000",
                observed_at=NOW,
            ),
            day=NOW.date(),
        )
        return journal

    def test_the_spread_is_in_the_book(self, tmp_path: Path) -> None:
        journal = self.journal_with_a_late_fill(tmp_path)
        positions, unexplained, _ = open_positions(
            OPEN_LEGS, journal.read_through(NOW.date())
        )
        assert len(positions) == 1
        assert unexplained == (), "not a leg the journal cannot explain"

    def test_the_gate_budgets_against_it(self, tmp_path: Path) -> None:
        journal = self.journal_with_a_late_fill(tmp_path)
        book = read_book(account(), OPEN_LEGS, journal.read_through(NOW.date()))
        assert book.snapshot.open_structures == 1
        assert book.snapshot.open_risk == Decimal("393.00")
        assert book.is_clean

    def test_the_exit_policy_can_see_it(self, tmp_path: Path) -> None:
        """`book.held` is what the runner iterates to re-price and exit. An
        empty tuple here is a position nobody is managing."""
        journal = self.journal_with_a_late_fill(tmp_path)
        book = read_book(account(), OPEN_LEGS, journal.read_through(NOW.date()))
        assert [item.cycle_id for item in book.held] == ["2026-08-26-SPY-000"]


def closing(
    cycle_id: str = "2026-08-26-SPY-500",
    *,
    stage: str = "submitted",
    outcome: dict[str, object] | None = None,
    structure: dict[str, object] | None = None,
) -> dict[str, object]:
    """An exit cycle as `run_exit_cycle` journals one: the *held* structure,
    `intent: close`, and whatever the broker has said about it so far."""
    record: dict[str, object] = {
        "cycle_id": cycle_id,
        "as_of": NOW.isoformat(),
        "stage": stage,
        "proposal": {
            "intent": "close",
            "quantity": 1,
            "structure": structure if structure is not None else structure_payload(),
            "risk": {"max_loss": "393.00"},
        },
    }
    if outcome is not None:
        record["outcome"] = outcome
    return record


class TestAClosingOrderAlreadyWorking:
    """One decision, sent once. The hole this closes was opened by fixing the
    exit path at all.

    The exit policy is evaluated every slot against the broker's positions, and
    a position with a working close order is still a position: the legs are
    there until the close fills. So the slot after a close is submitted proposes
    the same close again, mints a *different* `client_order_id` (the key derives
    from the proposal, and every slot is a new proposal), and the broker accepts
    it. Two working closes on a one-lot spread that both fill do not flatten it
    twice — they flatten it and then open the mirror position, which no journal
    line asked for.

    Until 2026-09-03 this was unreachable: every close was refused at the door
    for a malformed position intent. Fixing that made it reachable, so this is
    part of the same fix.
    """

    def test_a_submitted_close_is_reported_as_working(self) -> None:
        working = working_closes([filled(), closing()])
        assert [leg.contract.strike for leg in working[0].legs] == [
            Decimal("747"),
            Decimal("752"),
        ]

    def test_the_book_names_the_position_being_closed(self) -> None:
        book = read_book(account(), OPEN_LEGS, [filled(), closing()])
        assert book.closing == ("2026-08-26-SPY-000",)
        assert [item.cycle_id for item in book.held] == ["2026-08-26-SPY-000"], (
            "still held -- a working close is not a closed position"
        )

    def test_a_filled_close_is_not_working(self) -> None:
        """`Journal.read` settles the stage from the outcome, so a close that
        filled reads `filled` here. Its legs are gone from the broker anyway."""
        book = read_book(
            account(), OPEN_LEGS, [filled(), closing(stage="filled", outcome={"status": "filled"})]
        )
        assert book.closing == ()

    def test_a_cancelled_close_is_not_working(self) -> None:
        """The case a naive `stage == submitted` test gets wrong. A cancelled
        order is terminal and has no `Stage` of its own, so the line still reads
        `submitted` -- and the position must be closable again today, not left
        open until tomorrow."""
        book = read_book(
            account(),
            OPEN_LEGS,
            [filled(), closing(outcome={"status": "canceled"})],
        )
        assert book.closing == ()

    def test_a_live_outcome_is_still_working(self) -> None:
        book = read_book(
            account(), OPEN_LEGS, [filled(), closing(outcome={"status": "new"})]
        )
        assert book.closing == ("2026-08-26-SPY-000",)

    def test_an_opening_cycle_is_not_a_close(self) -> None:
        """`intent` is what separates them, not the structure: an exit carries
        the same structure as the position it closes."""
        book = read_book(account(), OPEN_LEGS, [filled()])
        assert book.closing == ()

    def test_a_close_for_a_different_position_does_not_block_this_one(self) -> None:
        other = structure_payload(short="762", long="757")
        book = read_book(
            account(), OPEN_LEGS, [filled(), closing(structure=other)]
        )
        assert book.closing == ()


class TestTheDailyFillCapCanSeeTheDaysFills:
    """`daily_trade_cap` counts `PortfolioSnapshot.fills_today`, and live that
    number was always zero: `gather_for` threads a counter nobody increments.
    So specs/03 D5's ceiling on how fast a bad day can compound never bound, and
    the dashboard's "fills today" tile read 0 on a day with two fills.

    Counted off the journal instead, which is the same place the day's fills are
    read from everywhere else -- and which only became reliable once
    `Journal.read` started settling `stage` from the outcome, because a fill
    that landed as an amendment used to read `submitted`.
    """

    def test_todays_fills_are_counted(self) -> None:
        day = NOW.date()
        assert fills_on([filled(), filled("2026-08-26-SPY-001")], day) == 2

    def test_another_days_fills_are_not(self) -> None:
        yesterday = {**filled("2026-08-25-SPY-000"), "as_of": "2026-08-25T14:30:00+00:00"}
        assert fills_on([yesterday, filled()], NOW.date()) == 1

    def test_only_fills_count(self) -> None:
        """Not proposals, and not orders still working -- a Gate that counted
        its own vetoes could talk itself out of trading."""
        records = [
            filled(),
            {**filled("2026-08-26-SPY-001"), "stage": "submitted"},
            {**filled("2026-08-26-SPY-002"), "stage": "vetoed"},
        ]
        assert fills_on(records, NOW.date()) == 1

    def test_an_exit_that_filled_counts_too(self) -> None:
        """specs/03's own wording is "fills today", without a carve-out. A close
        is a fill: it crosses a spread and it is a trade the day did."""
        assert fills_on([filled(), closing(stage="filled")], NOW.date()) == 2

    def test_an_equity_pass_is_not_an_option_fill(self) -> None:
        equity = {
            "cycle_id": "2026-08-26-EQ-000",
            "as_of": NOW.isoformat(),
            "kind": "equity",
            "stage": "submitted",
        }
        assert fills_on([equity, filled()], NOW.date()) == 1

    def test_a_record_with_no_day_is_not_counted_as_todays(self) -> None:
        assert fills_on([{"cycle_id": "x", "stage": "filled"}], NOW.date()) == 0


class TestUnexplainedLegsAreNeverModelled:
    """The one that must not regress."""

    def test_a_leg_no_fill_accounts_for_is_reported(self) -> None:
        positions, unexplained, _ = open_positions(OPEN_LEGS, [])
        assert positions == ()
        assert len(unexplained) == 2

    def test_it_is_kept_out_of_the_snapshot_the_gate_sees(self) -> None:
        """An agent that absorbed an unknown short leg into its risk model would
        be reporting a defined risk it had not defined."""
        book = read_book(account(), OPEN_LEGS, [])
        assert book.snapshot.open_structures == 0
        assert book.snapshot.open_risk == Decimal(0)
        assert not book.is_clean
        assert "UNEXPLAINED" in book.summary()

    def test_a_clean_book_says_so(self) -> None:
        book = read_book(account(), OPEN_LEGS, [filled()])
        assert book.is_clean
        assert book.snapshot.open_structures == 1
        assert "UNEXPLAINED" not in book.summary()


class TestDrawdown:
    """Measured against the sleeve, never against the account — specs/03 D6."""

    def test_it_is_measured_against_the_high_water_mark(self) -> None:
        """The whole sleeve allocated, 5% of it lost on closed trades.

        Written against `OPTIONS_SLEEVE_ALLOCATION` rather than against the
        figure it happens to hold: the invariant is "drawdown is a fraction of
        the sleeve", and a test that spelled the number out would fail on the
        day the operator re-splits the account, which is a configuration change
        and not a regression.
        """
        loss = OPTIONS_SLEEVE_ALLOCATION / 20
        book = read_book(
            account(),
            (),
            [closed_for(-loss)],
            peak_equity=OPTIONS_SLEEVE_ALLOCATION,
        )
        assert book.snapshot.equity == OPTIONS_SLEEVE_ALLOCATION - loss
        assert book.snapshot.drawdown_pct == Decimal("0.05")

    def test_a_new_high_is_not_a_drawdown(self) -> None:
        book = read_book(
            account(),
            (),
            [closed_for(Decimal("250"))],
            peak_equity=OPTIONS_SLEEVE_ALLOCATION,
        )
        assert book.snapshot.drawdown_pct == Decimal(0)

    def test_the_equity_book_cannot_trip_the_options_kill_switch(self) -> None:
        """The bug this sleeve exists to remove.

        The account falls from $100,000 to $92,000 — an 8% fall, all of it the
        equity book's mark-to-market. Under the old account-scaled rule that was
        a 5% drawdown against a 5% threshold, and the options agent latched shut
        having lost nothing. The options sleeve traded nothing, so it is flat.
        """
        book = read_book(
            account("92000"), (), [], peak_equity=OPTIONS_SLEEVE_ALLOCATION
        )
        assert book.snapshot.equity == OPTIONS_SLEEVE_ALLOCATION
        assert book.snapshot.drawdown_pct == Decimal(0)

    def test_the_gate_budgets_against_the_sleeve_not_the_account(self) -> None:
        """A $100,000 account does not buy a $100,000 options budget."""
        book = read_book(account("100000"), (), [])
        assert book.snapshot.equity == OPTIONS_SLEEVE_ALLOCATION

    def test_the_two_sleeves_sum_to_the_account_and_neither_may_grow_alone(
        self,
    ) -> None:
        """Alpaca holds one pool of buying power and has never heard of sleeves,
        so the split is only meaningful while it adds up. This is the check that
        catches a re-split done on one side and forgotten on the other."""
        assert Decimal(100_000) == EQUITY_SLEEVE_ALLOCATION + OPTIONS_SLEEVE_ALLOCATION

    def test_the_options_sleeve_funds_a_contract_of_the_researched_rule(self) -> None:
        """The arithmetic that forced the 90/10 split, kept as a test.

        specs/07 D1's structure risked $1,389 a contract when it was priced
        against the real chain on 2026-08-28, and `agent/sizing.py` floors the
        quantity — so a per-trade budget under that figure buys nothing at all
        and the rule looks like a market with no setups. At $5,000 the budget
        was $1,000 and this failed silently every cycle.
        """
        budget = SLEEVE_LIMITS.max_trade_loss(OPTIONS_SLEEVE_ALLOCATION)
        assert budget >= Decimal("1389"), (
            f"per-trade budget {budget} cannot fund one contract of the rule this "
            "sleeve exists to trade"
        )
        # And it is the size the sealed run measured: 2% of $100,000, specs/10 D8a.
        assert budget == Decimal("2000")

    def test_no_history_means_no_drawdown_and_has_to_be_asked_for(self) -> None:
        """specs/03 D4's kill switch watches this number. A drawdown that resets
        every morning is a kill switch that cannot latch across the days that
        matter — so `peak_equity` is a parameter, not a default."""
        assert read_book(account(), (), []).snapshot.drawdown_pct == Decimal(0)

    def test_the_latch_is_carried_in_rather_than_recomputed(self) -> None:
        """specs/03 D4: the Gate is pure, so the latch rides in on the snapshot."""
        book = read_book(account(), (), [], killswitch_tripped=True)
        assert book.snapshot.killswitch_tripped
