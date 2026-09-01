"""Both sleeves, measured apart, for the dashboard — specs/03 D6.

Every test builds the two snapshots by hand rather than through the live
agents, because `interface.sleeves` is a pure function of whatever the two
status files last published (`interface/sleeves.py`'s own module docstring
explains why that is deliberately not the same thing the live kill switch
enforces). The one property worth testing everywhere is the one the whole
module exists for: the options sleeve's arithmetic never reads the equity
snapshot, and a loss on one side never appears on the other's ledger except
through the residual identity `risk.sleeve.residual_sleeve` already owns.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.equity import EQUITY_SLEEVE_ALLOCATION
from alphagate.interface.sleeves import (
    build_sleeve_overview,
    equity_sleeve_view,
    options_sleeve_view,
)
from alphagate.journal import Journal
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION

DAY = date(2026, 8, 31)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(directory=tmp_path / "journal")


def options_status(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_day": DAY.isoformat(),
        "positions": [],
        "drawdown_pct": "0",
        "max_drawdown_pct": "0.20",
        "killswitch_tripped": False,
        "fills_today": 0,
    }
    base.update(overrides)
    return base


def equity_status(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "equity": "100000",
        "drawdown_pct": "0",
        "max_drawdown_pct": "0.10",
        "killswitch_tripped": False,
        "positions_held": 40,
        "orders_today": 0,
    }
    base.update(overrides)
    return base


class TestOptionsSleeve:
    def test_no_snapshot_is_not_running(self, journal: Journal) -> None:
        view = options_sleeve_view(None, journal=journal)
        assert not view.running
        assert view.allocation == OPTIONS_SLEEVE_ALLOCATION
        assert view.equity is None

    def test_a_flat_sleeve_is_worth_its_allocation(self, journal: Journal) -> None:
        view = options_sleeve_view(options_status(), journal=journal)
        assert view.running
        assert view.equity == OPTIONS_SLEEVE_ALLOCATION
        assert view.realised == Decimal(0)
        assert view.unrealised == Decimal(0)

    def test_unrealised_sums_the_open_positions(self, journal: Journal) -> None:
        status = options_status(
            positions=[{"unrealised": "40.50"}, {"unrealised": "-15.00"}]
        )
        view = options_sleeve_view(status, journal=journal)
        assert view.unrealised == Decimal("25.50")
        assert view.equity == OPTIONS_SLEEVE_ALLOCATION + Decimal("25.50")
        assert view.open_positions == 2

    def test_realised_pl_is_read_from_the_journal_cumulatively(
        self, journal: Journal
    ) -> None:
        """The same field `interface/read.py` already renders for one cycle,
        summed the way `agent.book.realised_pl` sums it for the live Gate."""
        journal.append(
            {
                "cycle_id": "2026-08-20-SPY-000",
                "as_of": "2026-08-20T14:00:00+00:00",
                "outcome": {"realised_pl": "-42.50"},
            },
            day=date(2026, 8, 20),
        )
        journal.append(
            {
                "cycle_id": "2026-08-31-SPY-000",
                "as_of": "2026-08-31T14:00:00+00:00",
                "outcome": {"realised_pl": "10.00"},
            },
            day=DAY,
        )
        view = options_sleeve_view(options_status(), journal=journal)
        assert view.realised == Decimal("-32.50")

    def test_a_future_day_is_not_read(self, journal: Journal) -> None:
        """Look-ahead is look-ahead even when the thing leaking backwards is our
        own realised P&L — the same bound `Journal.read_through` documents."""
        journal.append(
            {
                "cycle_id": "2026-09-05-SPY-000",
                "as_of": "2026-09-05T14:00:00+00:00",
                "outcome": {"realised_pl": "999"},
            },
            day=date(2026, 9, 5),
        )
        view = options_sleeve_view(options_status(), journal=journal)
        assert view.realised == Decimal(0)

    def test_drawdown_and_threshold_travel_from_the_snapshot(
        self, journal: Journal
    ) -> None:
        status = options_status(drawdown_pct="0.05", max_drawdown_pct="0.20")
        view = options_sleeve_view(status, journal=journal)
        assert view.drawdown_pct == Decimal("0.05")
        assert view.max_drawdown_pct == Decimal("0.20")


class TestEquitySleeve:
    def test_no_snapshot_is_not_running(self) -> None:
        from alphagate.interface.sleeves import SleeveSummary

        options = SleeveSummary(name="options", allocation=OPTIONS_SLEEVE_ALLOCATION, running=False)
        view = equity_sleeve_view(None, options=options)
        assert not view.running
        assert view.allocation == EQUITY_SLEEVE_ALLOCATION

    def test_the_residual_is_the_account_less_the_options_sleeve(
        self, journal: Journal
    ) -> None:
        options = options_sleeve_view(
            options_status(positions=[{"unrealised": "500"}]), journal=journal
        )
        account_equity = OPTIONS_SLEEVE_ALLOCATION + Decimal(500) + EQUITY_SLEEVE_ALLOCATION
        view = equity_sleeve_view(equity_status(equity=str(account_equity)), options=options)
        assert view.equity == EQUITY_SLEEVE_ALLOCATION

    def test_an_options_loss_does_not_touch_the_equity_residual(
        self, journal: Journal
    ) -> None:
        """specs/03 D6's whole point, restated as arithmetic: if the account
        fell by exactly what the options sleeve lost, the equity book is
        untouched."""
        options = options_sleeve_view(
            options_status(positions=[{"unrealised": "-1000"}]), journal=journal
        )
        account_equity = OPTIONS_SLEEVE_ALLOCATION - Decimal(1000) + EQUITY_SLEEVE_ALLOCATION
        view = equity_sleeve_view(equity_status(equity=str(account_equity)), options=options)
        assert view.equity == EQUITY_SLEEVE_ALLOCATION

    def test_an_unmeasured_options_sleeve_leaves_the_residual_unavailable(self) -> None:
        from alphagate.interface.sleeves import SleeveSummary

        options = SleeveSummary(name="options", allocation=OPTIONS_SLEEVE_ALLOCATION, running=False)
        view = equity_sleeve_view(equity_status(), options=options)
        assert view.equity is None
        assert view.note

    def test_drawdown_and_threshold_are_the_equity_sleeves_own(
        self, journal: Journal
    ) -> None:
        options = options_sleeve_view(options_status(), journal=journal)
        status = equity_status(drawdown_pct="0.03", max_drawdown_pct="0.10")
        view = equity_sleeve_view(status, options=options)
        assert view.drawdown_pct == Decimal("0.03")
        assert view.max_drawdown_pct == Decimal("0.10")

    def test_the_two_thresholds_differ(self, journal: Journal) -> None:
        """The property the whole panel exists to make obvious: each sleeve is
        measured against its own kill switch, and the two are not the same
        number."""
        options = options_sleeve_view(options_status(), journal=journal)
        equity = equity_sleeve_view(equity_status(), options=options)
        assert options.max_drawdown_pct != equity.max_drawdown_pct


class TestBuildSleeveOverview:
    def test_both_sides_come_back_as_plain_json(self, journal: Journal) -> None:
        body = build_sleeve_overview(options_status(), equity_status(), journal=journal)
        assert body["options"]["running"] is True
        assert body["equity"]["running"] is True
        assert isinstance(body["options"]["allocation"], str)

    def test_neither_side_running_still_reports_the_fixed_allocation(
        self, journal: Journal
    ) -> None:
        body = build_sleeve_overview(None, None, journal=journal)
        assert body["options"]["running"] is False
        assert body["options"]["allocation"] == format(OPTIONS_SLEEVE_ALLOCATION, "f")
        assert body["equity"]["allocation"] == format(EQUITY_SLEEVE_ALLOCATION, "f")

    def test_no_blended_number_is_shared_between_the_two(self, journal: Journal) -> None:
        """The requirement this module exists to satisfy: never show a single
        figure both sleeves share as though it meant the same thing twice."""
        body = build_sleeve_overview(
            options_status(positions=[{"unrealised": "200"}]),
            equity_status(),
            journal=journal,
        )
        assert body["options"]["equity"] != body["equity"]["equity"]
