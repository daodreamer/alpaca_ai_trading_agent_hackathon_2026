"""Shaping journal lines for a page — `interface/read.py`.

Pure, so it is tested without a web server. Everything here is a consequence of
one sentence in specs/06: a judge opens a fill and reads the reasoning that
produced it, *including the checks that nearly stopped it*.

The load-bearing piece is `headroom`. "Passed" and "passed with 4% of the budget
left" are very different facts about a risk system, and only the second is
evidence that the Gate is doing anything. If that number is wrong the dashboard
tells a flattering story, which is worse than telling none.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent import Stage
from alphagate.interface.read import available_days, day_view, to_cycle
from alphagate.journal import Journal, outcome_from
from tests.journal.conftest import DAY, at_stage, cycle, submission

FILLED_AT = "2026-08-26T18:00:00+00:00"


@pytest.fixture
def loaded(journal: Journal) -> Journal:
    journal.append(cycle(with_setup=False))
    journal.append(at_stage(Stage.VETOED, sequence=1))
    journal.append(at_stage(Stage.FILLED, sequence=2))
    return journal


class TestTheDay:
    def test_every_cycle_is_present_including_the_quiet_ones(self, loaded: Journal) -> None:
        """specs/06 D2: a dashboard that lists only fills answers the easy
        question."""
        view = day_view(loaded, DAY)
        assert len(view.cycles) == 3
        assert view.quiet == 1
        assert view.fills == 1
        assert view.vetoes == 1

    def test_stage_counts_add_up(self, loaded: Journal) -> None:
        view = day_view(loaded, DAY)
        assert sum(view.stage_counts.values()) == len(view.cycles)

    def test_an_absent_day_is_empty_not_an_error(self, journal: Journal) -> None:
        view = day_view(journal, date(2020, 1, 1))
        assert view.cycles == ()
        assert not view.has_warnings

    def test_available_days_are_newest_first(self, journal: Journal) -> None:
        journal.append(cycle())
        journal.append({"cycle_id": "2026-09-01-SPY-000"}, day=date(2026, 9, 1))
        assert available_days(journal) == (date(2026, 9, 1), date(2026, 8, 26))

    def test_a_directory_that_does_not_exist_yields_nothing(self, tmp_path: Path) -> None:
        assert available_days(Journal(directory=tmp_path / "absent")) == ()

    def test_warnings_surface_duplicates_and_orphans(self, journal: Journal) -> None:
        """The dashboard is where an operator would notice. Hiding a collapsed
        decision behind a tidy table is how it stays unnoticed."""
        journal.append(at_stage(Stage.VETOED, sequence=3))
        journal.append(at_stage(Stage.DRY_RUN, sequence=3))
        journal.amend("2026-08-26-QQQ-009", outcome={"status": "filled"})
        view = day_view(journal, DAY)
        assert view.has_warnings
        assert view.duplicates == {"2026-08-26-SPY-003": 2}
        assert view.orphans == ("2026-08-26-QQQ-009",)


class TestRealisedPnl:
    def test_only_closed_positions_count(self, journal: Journal) -> None:
        """specs/07 D7. An open position's mark is not a result, and adding it
        in produces a number that flatters."""
        record = at_stage(Stage.FILLED)
        journal.append(record)
        assert day_view(journal, DAY).realised == Decimal(0)

    def test_a_close_amendment_lands_in_the_total(self, journal: Journal) -> None:
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        fill = outcome_from(
            submission("order_filled"),
            cycle_id=record.cycle_id,
            observed_at=record.as_of,
        )
        from alphagate.journal import realised

        journal.record_outcome(realised(fill, Decimal("-42.50"), closed_at=record.as_of))
        assert day_view(journal, DAY).realised == Decimal("-42.50")


class TestOneCycle:
    def test_the_market_read_survives_into_the_view(self, journal: Journal) -> None:
        journal.append(cycle())
        view = day_view(journal, DAY).cycles[0]
        assert view.underlying == "SPY"
        assert view.spot == "766.00"
        assert view.iv_rank == "62"

    def test_an_unmeasured_field_says_so_rather_than_showing_zero(self) -> None:
        """A `None` iv_rank is nobody's measurement, not a middling one."""
        assert to_cycle({"read": {"iv_rank": None}}).iv_rank == "unmeasured"

    def test_the_whole_menu_is_counted(self, journal: Journal) -> None:
        journal.append(cycle())
        assert day_view(journal, DAY).cycles[0].candidate_count > 1

    def test_the_rationale_and_its_provenance_come_together(self, journal: Journal) -> None:
        """"A rationale without the model id and prompt version behind it is a
        quote with no source." """
        journal.append(cycle())
        view = day_view(journal, DAY).cycles[0]
        assert view.rationale
        assert view.model == "fixture-model"
        assert view.prompt_version == "test"

    def test_confidence_is_shown_and_nothing_reads_it(self, journal: Journal) -> None:
        """specs/05 D5. Rendering it is fine; the view does no arithmetic on it."""
        journal.append(cycle())
        assert day_view(journal, DAY).cycles[0].confidence == "0.55"

    def test_a_quiet_cycle_is_marked_quiet(self, journal: Journal) -> None:
        journal.append(cycle(with_setup=False))
        assert day_view(journal, DAY).cycles[0].is_quiet

    def test_the_trust_line_travels(self, journal: Journal) -> None:
        journal.append(at_stage(Stage.FILLED))
        view = day_view(journal, DAY).cycles[0]
        assert "untrusted" in view.trust
        assert "submission.raw" in view.untrusted_paths


class TestNearMisses:
    """The reason this module exists."""

    def test_headroom_is_the_fraction_of_the_budget_left(self) -> None:
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "max_loss", "passed": True, "observed": "20", "limit": "100"}
            ]}}
        )
        assert view.checks[0].headroom == pytest.approx(0.8)

    def test_a_failed_check_is_zero_not_negative(self) -> None:
        """A scale running below zero would sort a badly failed check as if it
        had nearly passed."""
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "max_loss", "passed": False, "observed": "400", "limit": "100"}
            ]}}
        )
        assert view.checks[0].headroom == 0.0

    def test_a_check_with_no_magnitude_has_no_headroom(self) -> None:
        """`defined_risk` is a yes or a no. A percentage of it means nothing."""
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "defined_risk", "passed": True, "observed": None, "limit": None}
            ]}}
        )
        assert view.checks[0].headroom is None
        assert not view.checks[0].is_near_miss

    def test_a_near_miss_is_a_pass_that_nearly_was_not(self) -> None:
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "tight", "passed": True, "observed": "96", "limit": "100"},
                {"name": "loose", "passed": True, "observed": "10", "limit": "100"},
            ]}}
        )
        assert [check.name for check in view.near_misses] == ["tight"]

    def test_near_misses_come_back_tightest_first(self) -> None:
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "b", "passed": True, "observed": "90", "limit": "100"},
                {"name": "a", "passed": True, "observed": "99", "limit": "100"},
            ]}}
        )
        assert [check.name for check in view.near_misses] == ["a", "b"]

    def test_every_check_is_kept_not_only_the_failures(self, journal: Journal) -> None:
        """specs/06 D2: "Every check, passed and failed." A gate that only shows
        what it refused looks arbitrary."""
        journal.append(cycle())
        checks = day_view(journal, DAY).cycles[0].checks
        assert len(checks) == 13
        assert any(check.passed for check in checks)

    def test_a_cycle_that_never_reached_the_gate_has_no_checks(
        self, journal: Journal
    ) -> None:
        journal.append(cycle(with_setup=False))
        assert day_view(journal, DAY).cycles[0].checks == ()

    def test_a_zero_limit_does_not_divide_by_zero(self) -> None:
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "x", "passed": True, "observed": "0", "limit": "0"}
            ]}}
        )
        assert view.checks[0].headroom is None


class TestMalformedLinesDoNotBreakThePage:
    """A dashboard that 500s on one odd line is a dashboard nobody trusts."""

    @pytest.mark.parametrize(
        "record",
        [
            {},
            {"read": "not a mapping"},
            {"verdict": {"checks": "not a list"}},
            {"candidates": None},
            {"proposal": {"structure": {"legs": [{"contract": {}}]}}},
            {"as_of": "not-a-timestamp"},
        ],
    )
    def test_it_renders_something(self, record: dict[str, object]) -> None:
        view = to_cycle(record)
        assert isinstance(view.stage, str)
        assert isinstance(view.structure, str)
