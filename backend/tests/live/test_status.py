"""The status snapshot — `live/status.py` and `interface/status.py`.

The journal says what was decided; this says what is true now. The two failures
that matter are both about honesty rather than features:

1. **A stopped agent must look stopped.** The file stops being rewritten and the
   age grows. A dashboard that cannot tell you it has lost contact is worse than
   no dashboard, because it shows yesterday's book with today's confidence.
2. **An unpriced position must not read as a free one.** A missing mark
   defaulted to zero would render as a 100% profit and an imminent close, which
   is the most alarming possible way to say "the chain request failed".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent.book import BookRead, HeldPosition, read_book
from alphagate.agent.exits import DEFAULT_EXIT_POLICY, ExitPolicy
from alphagate.execution import AccountRead, LegPosition
from alphagate.interface.status import STALE_AFTER, _age_of
from alphagate.interface.status import read_status as read_from_interface
from alphagate.live.status import StatusSnapshot, build_status, read_status, write_status
from alphagate.options import OptionContract, Right, compute_risk
from alphagate.risk import DEFAULT_LIMITS, OpenPosition, PortfolioSnapshot
from tests.agent.conftest import EXPIRY, SPY
from tests.agent.test_exit_cycle import held, quotes, spread

NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)


def account(equity: str = "100000", **overrides: object) -> AccountRead:
    fields: dict[str, object] = {
        "equity": Decimal(equity),
        "last_equity": Decimal(equity),
        "buying_power": Decimal(equity),
        "options_buying_power": Decimal(equity),
        "options_level": 3,
        "cash": Decimal(equity),
        "multiplier": 1,
        "is_blocked": False,
        "envelope": None,
        "observed_at": NOW,
    }
    fields.update(overrides)
    return AccountRead(**fields)  # type: ignore[arg-type]


def book(*positions: HeldPosition, drawdown: str = "0", latched: bool = False) -> BookRead:
    return BookRead(
        snapshot=PortfolioSnapshot(
            equity=Decimal(100000),
            positions=tuple(item.position for item in positions),
            drawdown_pct=Decimal(drawdown),
            fills_today=0,
            killswitch_tripped=latched,
        ),
        held=positions,
    )


def snapshot(
    *positions: HeldPosition,
    marks: dict[str, object] | None = None,
    policy: ExitPolicy = DEFAULT_EXIT_POLICY,
    **book_kwargs: object,
) -> StatusSnapshot:
    return build_status(
        account=account(),
        book=book(*positions, **book_kwargs),  # type: ignore[arg-type]
        limits=DEFAULT_LIMITS,
        policy=policy,
        marks=marks or {},  # type: ignore[arg-type]
        as_of=NOW,
        next_slot=NOW + timedelta(minutes=15),
        slot_sequence=3,
        universe=("SPY", "QQQ"),
        peak_equity=Decimal(105000),
        stage_counts={"dry_run": 2},
    )


class TestItSaysWhatIsTrueNow:
    def test_the_account_reaches_the_snapshot(self) -> None:
        status = snapshot()
        assert status.equity == Decimal(100000)
        assert status.options_level == 3
        assert status.can_trade_spreads

    def test_limits_are_rendered_in_currency_not_percentages(self) -> None:
        """So the page shows used-against-limit rather than a fraction the
        reader has to un-multiply by an equity figure three tiles away."""
        assert snapshot().max_portfolio_risk == DEFAULT_LIMITS.max_portfolio_loss(
            Decimal(100000)
        )

    def test_the_exit_policy_travels(self) -> None:
        """`ExitPolicy` says of itself: one place, logged at startup, shown in
        the dashboard. This is the third of those."""
        status = snapshot()
        assert status.profit_target == DEFAULT_EXIT_POLICY.profit_target
        assert status.stop_multiple == DEFAULT_EXIT_POLICY.stop_multiple
        assert status.min_dte == DEFAULT_EXIT_POLICY.min_dte

    def test_a_flat_book_is_healthy(self) -> None:
        assert snapshot().is_healthy

    def test_a_latched_killswitch_is_not_healthy(self) -> None:
        assert not snapshot(latched=True).is_healthy

    def test_a_blocked_account_is_not_healthy(self) -> None:
        status = build_status(
            account=account(is_blocked=True),
            book=book(),
            limits=DEFAULT_LIMITS,
            policy=DEFAULT_EXIT_POLICY,
            marks={},
            as_of=NOW,
            next_slot=None,
            slot_sequence=0,
            universe=(),
            peak_equity=None,
            stage_counts={},
        )
        assert not status.is_healthy


class TestPositionsSayHowCloseTheyAreToClosing:
    def test_a_priced_position_reports_its_distance_to_both_rules(self) -> None:
        """"Up 34%" is a fact about the past. "16 points from the target" is
        what says what the agent is about to do."""
        item = held()
        # 0.50 net against a 0.60 credit: 17% earned, short of the 50% target
        # and nowhere near the stop, so both distances are meaningful.
        mark = compute_risk(item.position.structure, quotes("0.60", "0.10"), NOW)
        status = snapshot(item, marks={item.cycle_id: mark})

        position = status.positions[0]
        assert position.rule == "hold"
        assert position.fraction_of_credit is not None
        assert position.to_target is not None
        assert position.to_stop is not None
        assert position.to_target > 0, "still short of the profit target"

    def test_a_position_at_its_target_says_it_will_close(self) -> None:
        item = held()
        mark = compute_risk(item.position.structure, quotes("0.30", "0.10"), NOW)
        status = snapshot(item, marks={item.cycle_id: mark})
        assert status.positions[0].should_close
        assert status.positions[0].rule == "profit_target"

    def test_an_unpriced_position_says_unpriced_not_zero(self) -> None:
        """The failure this whole field exists for. A mark defaulted to zero
        renders as a 100% profit and an imminent close."""
        item = held()
        status = snapshot(item, marks={})
        position = status.positions[0]
        assert position.rule == "unpriced"
        assert position.fraction_of_credit is None
        assert position.to_target is None
        assert not position.should_close

    def test_the_position_carries_the_cycle_that_opened_it(self) -> None:
        """So a reader can go from the live row to the decision behind it."""
        item = held()
        assert snapshot(item).positions[0].cycle_id == item.cycle_id


class TestUnexplainedLegsAreLoud:
    def test_they_are_listed_and_the_book_is_unhealthy(self) -> None:
        leg = LegPosition(
            contract=OptionContract(SPY, EXPIRY, Decimal("752"), Right.PUT),
            quantity=-1,
            average_price=Decimal("1.50"),
            market_value=Decimal("-150"),
            unrealised=Decimal(0),
        )
        read = read_book(account(), (leg,), [])
        status = build_status(
            account=account(),
            book=read,
            limits=DEFAULT_LIMITS,
            policy=DEFAULT_EXIT_POLICY,
            marks={},
            as_of=NOW,
            next_slot=None,
            slot_sequence=0,
            universe=(),
            peak_equity=None,
            stage_counts={},
        )
        assert status.unexplained == ("SPY260904P00752000",)
        assert not status.is_healthy


class TestTheFileOnDisk:
    def test_it_round_trips(self, tmp_path: Path) -> None:
        write_status(snapshot(), directory=tmp_path)
        loaded = read_status(tmp_path)
        assert loaded is not None
        assert loaded["equity"] == "100000"

    def test_money_stays_a_string(self, tmp_path: Path) -> None:
        """Same discipline as the journal: a float here would disagree with the
        Decimal the Gate budgeted against."""
        path = write_status(snapshot(), directory=tmp_path)
        raw = path.read_text(encoding="utf-8")
        assert '"equity": "100000"' in raw
        assert '"equity": 100000' not in raw

    def test_it_is_written_atomically(self, tmp_path: Path) -> None:
        """A poller that caught a half-written file would show a parse error
        every few seconds on a healthy system."""
        write_status(snapshot(), directory=tmp_path)
        assert not list(tmp_path.glob("*.tmp")), "no temporary file left behind"
        assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    def test_rewriting_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        write_status(snapshot(), directory=tmp_path)
        write_status(snapshot(), directory=tmp_path)
        assert len(list(tmp_path.glob("status*"))) == 1

    def test_an_absent_file_is_none_not_an_error(self, tmp_path: Path) -> None:
        assert read_status(tmp_path) is None
        assert read_from_interface(tmp_path) is None

    def test_a_corrupt_file_is_none_not_an_error(self, tmp_path: Path) -> None:
        """The dashboard's job then is to say it lost contact, not to 500."""
        (tmp_path / "status.json").write_text("{not json", encoding="utf-8")
        assert read_status(tmp_path) is None
        assert read_from_interface(tmp_path) is None

    def test_no_credentials_reach_it(self, tmp_path: Path) -> None:
        """specs/06 D4. This file is on screen during the demo for the same
        reasons the journal is."""
        path = write_status(snapshot(), directory=tmp_path)
        raw = path.read_text(encoding="utf-8")
        for forbidden in ("account_number", "api_key", "secret", "PKTEST"):
            assert forbidden not in raw


class TestStaleness:
    def test_a_fresh_snapshot_is_young(self) -> None:
        age = _age_of(datetime.now(UTC).isoformat())
        assert age is not None
        assert age < 5

    def test_an_old_snapshot_ages(self) -> None:
        """The agent stopping is not an event the file can report — the absence
        of a rewrite is the signal, and only the reader can see it."""
        old = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        age = _age_of(old)
        assert age is not None
        assert age > STALE_AFTER

    def test_a_naive_timestamp_is_read_as_utc(self) -> None:
        """Everything in this system is tz-aware, so a naive stamp means
        something upstream lost one — read it as UTC rather than crashing the
        page over it."""
        assert _age_of("2026-08-26T14:30:00") is not None

    @pytest.mark.parametrize("value", [None, 12, "not a timestamp", ""])
    def test_an_unreadable_timestamp_has_no_age(self, value: object) -> None:
        assert _age_of(value) is None


def test_the_stale_threshold_allows_one_slow_slot() -> None:
    """Cycles are fifteen minutes apart. Tighter than the cadence would flag
    every slow slot; much looser and a crashed agent looks alive through
    lunch."""
    assert 900 < STALE_AFTER < 2400


def test_the_snapshot_covers_the_position_fields_the_page_reads() -> None:
    """A field the page renders and the snapshot does not carry is a blank cell
    nobody notices until the demo."""
    item = held()
    mark = compute_risk(item.position.structure, quotes("0.60", "0.10"), NOW)
    position = snapshot(item, marks={item.cycle_id: mark}).positions[0]
    for field in (
        "cycle_id",
        "underlying",
        "structure",
        "quantity",
        "expiry",
        "days_to_expiry",
        "entry_premium",
        "mark",
        "unrealised",
        "rule",
        "detail",
        "max_loss",
    ):
        assert getattr(position, field) is not None, field


def test_the_structure_label_is_readable() -> None:
    item = held()
    assert snapshot(item).positions[0].structure == "vertical credit 747/752"


def test_open_position_is_reused_not_mirrored() -> None:
    """`HeldPosition` wraps the risk-layer type rather than copying it, so the
    Gate and the page cannot drift into two different books."""
    item = held()
    assert isinstance(item.position, OpenPosition)
    assert item.position.structure == spread()
