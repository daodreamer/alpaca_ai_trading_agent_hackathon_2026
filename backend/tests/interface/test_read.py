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
from alphagate.interface.read import (
    available_days,
    day_records_with_category,
    day_view,
    is_equity_record,
    stage_tally,
    to_cycle,
)
from alphagate.journal import Journal, outcome_from
from tests.journal.conftest import DAY, at_stage, cycle, submission

FILLED_AT = "2026-08-26T18:00:00+00:00"


def equity_pass() -> dict[str, object]:
    """One rebalance pass as `live/equity.py` journals it — specs/09 D10.

    Two orders, because the interesting counts are the ones that differ: one
    reached the broker and one the equity Gate refused. Both carry their check
    tape, which is what the options renderer was throwing away.
    """
    return {
        "cycle_id": "2026-08-26-EQ-000",
        "kind": "equity",
        "as_of": "2026-08-26T13:45:12+00:00",
        "stage": "submitted",
        "equity": "90010.46",
        "band_pct": "0.20",
        "turnover": "274.84",
        "note": "2 of 2 intents submitted",
        "orders": [
            {
                "symbol": "HD",
                "side": "buy",
                "shares": "0.5956",
                "notional": "137.42",
                "outcome": "pending_new",
                "verdict": {"checks": [{"name": "buying_power", "passed": True}]},
                "submission": {"status": "pending_new"},
            },
            {
                "symbol": "LOW",
                "side": "buy",
                "shares": "0.4",
                "notional": "137.42",
                "outcome": "vetoed",
                "verdict": {
                    "checks": [{"name": "position_cap", "passed": False}],
                    "reasons": [{"check": "position_cap", "detail": "9.1% > 8%"}],
                },
                "submission": None,
            },
        ],
        "skipped": [
            {"symbol": "SPY", "reason": "inside_band", "detail": "0.1% < 0.2%", "drift": "0.001"}
        ],
    }


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

    def test_a_floor_a_passing_check_sits_above_is_not_ranked(self) -> None:
        """`order_is_material` passes when the notional is *above* $25, so a
        $137 order is as far from failing as it gets. `CheckResult` carries no
        direction and the fraction assumes a ceiling, which would rank this at
        zero room — the tightest thing on the page, and a "near" badge on the
        safest check in the pass. Unrankable is the honest answer."""
        view = to_cycle(
            {"verdict": {"checks": [
                {"name": "order_is_material", "passed": True,
                 "observed": "137.407704", "limit": "25"}
            ]}}
        )
        assert view.checks[0].headroom is None
        assert not view.checks[0].is_near_miss
        assert view.near_misses == ()

    def test_a_failed_check_above_its_limit_is_still_zero(self) -> None:
        """The floor rule reads `passed`, not the ratio. A ceiling that was
        actually breached is a failure with no room left, not an unrankable
        one."""
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


class TestDayRecordsWithCategory:
    """The serialized form `/api/day/{day}` now ships — the single classifier.

    Before this existed the React app had no `category` field to read and
    re-derived the same judgement in TypeScript, which is two implementations
    of one classification with nothing keeping them in agreement. This is the
    one place that adds the fields the frontend now trusts, so the test that
    matters most here is the one that says every original key survives.
    """

    def test_every_original_key_survives(self) -> None:
        record = {"cycle_id": "x", "stage": "no_setup", "read": {"iv_rank": None}}
        (enriched,) = day_records_with_category([record])
        assert enriched["cycle_id"] == "x"
        assert enriched["stage"] == "no_setup"
        assert enriched["read"] == {"iv_rank": None}

    def test_the_three_fields_match_to_cycles_own_answer(self) -> None:
        record = {"stage": "vetoed"}
        (enriched,) = day_records_with_category([record])
        view = to_cycle(record)
        assert enriched["category"] == view.category
        assert enriched["category_label"] == view.category_label
        assert enriched["category_detail"] == view.category_detail

    def test_an_equity_record_is_passed_through_unstamped(self) -> None:
        """`category` is options' own decline taxonomy. Stamping it onto an
        equity cycle's `no_trades` / `submitted` stages would not classify
        anything -- it would mislabel it under a taxonomy that was never about
        that sleeve, so equity records are left exactly as they arrived."""
        record = {"cycle_id": "eq", "kind": "equity", "stage": "no_trades"}
        (enriched,) = day_records_with_category([record])
        assert "category" not in enriched
        assert "category_label" not in enriched
        assert "category_detail" not in enriched

    def test_the_input_records_are_not_mutated(self) -> None:
        record = {"stage": "declined"}
        day_records_with_category([record])
        assert "category" not in record

    def test_todays_real_fixture_tells_no_setup_and_no_candidates_apart(self) -> None:
        """The coordinator's own example: a `no_setup` cycle whose `iv_rank`
        was measured and did not clear the bar, next to a `no_candidates`
        cycle where the entry fired but nothing survived the menu screens.
        The measured value here is a stand-in, not a claim about what the
        correct reading was that day."""
        records = [
            {
                "cycle_id": "a",
                "stage": "no_setup",
                "read": {"iv_rank": "12.3"},
                "note": "iv_rank()=12.3 not < 15",
            },
            {
                "cycle_id": "b",
                "stage": "no_candidates",
                "read": {"iv_rank": "12.3"},
                "note": "no structure survived pricing, freshness, spread, DTE and sizing",
            },
        ]
        by_id = {r["category"]: r for r in day_records_with_category(records)}
        assert by_id["no_setup"]["cycle_id"] == "a"
        assert by_id["no_candidates"]["cycle_id"] == "b"


class TestTheTwoSleevesAreToldApart:
    """One journal, two agents — and one taxonomy that only fits one of them.

    Both agents write to the same daily file (`live/equity.py`'s `CYCLE_KIND`
    explains why: the question "what happened on 2026-09-02" is about the
    account, not about which process asked). Everything in this module is the
    options sleeve's vocabulary — `Stage`, `iv_rank`, a menu of candidates, a
    structure — so an equity pass read through `to_cycle` does not come out
    misclassified, it comes out *fabricated*: no spot, `iv_rank` "unmeasured",
    a menu of zero, and no checks, which every renderer here words as "the
    cycle never reached the Gate".

    That last one is the reason this is a correctness test and not a cosmetic
    one. The equity pass in the fixture below carries a full twelve-check
    verdict per order. Saying it never reached the Gate is the single most
    misleading sentence this dashboard could print about a risk system whose
    whole claim is that the Gate is load-bearing.
    """

    def test_an_equity_pass_is_not_one_of_the_options_cycles(
        self, journal: Journal
    ) -> None:
        journal.append(at_stage(Stage.FILLED))
        journal.append(equity_pass())
        view = day_view(journal, DAY)
        assert [cycle.cycle_id for cycle in view.cycles] == ["2026-08-26-SPY-000"]

    def test_it_is_carried_beside_them_rather_than_dropped(
        self, journal: Journal
    ) -> None:
        """Filtered out of the options list, not out of the day. A record that
        exists and is not shown is its own kind of dishonest."""
        journal.append(at_stage(Stage.FILLED))
        journal.append(equity_pass())
        (pass_view,) = day_view(journal, DAY).equity_passes
        assert pass_view.cycle_id == "2026-08-26-EQ-000"
        assert pass_view.stage == "submitted"
        assert pass_view.orders == 2
        assert pass_view.submitted == 1
        assert pass_view.vetoed == 1
        assert pass_view.skipped == 1
        assert pass_view.note == "2 of 2 intents submitted"

    def test_the_stage_counts_are_the_options_sleeves_own(
        self, journal: Journal
    ) -> None:
        """`submitted` on an equity pass and `submitted` on an options cycle are
        two different agents' words. Adding them up produces a number that is
        about neither sleeve — and it is the number the Live page prints."""
        journal.append(at_stage(Stage.SUBMITTED))
        journal.append(equity_pass())
        view = day_view(journal, DAY)
        assert view.stage_counts == {"submitted": 1}
        assert sum(view.stage_counts.values()) == len(view.cycles)

    def test_the_tally_helper_is_the_one_the_live_page_uses(self) -> None:
        """One implementation. `live/wiring.py` publishes `stage_counts` into
        `status.json` and this shapes the same field for the journal page; two
        copies of "count the stages, options only" would be one copy too many."""
        assert stage_tally(
            [
                {"stage": "declined"},
                {"stage": "declined"},
                {"kind": "equity", "stage": "submitted"},
                {"stage": ""},
            ]
        ) == {"declined": 2}

    def test_an_equity_pass_is_never_asked_for_a_category(self) -> None:
        """The same rule `day_records_with_category` already applies, restated
        where the view types are built."""
        assert is_equity_record({"kind": "equity"})
        assert not is_equity_record({"stage": "no_setup"})


class TestCategory:
    """specs/06 D2's three-way distinction, made renderable — requirement 3.

    A cycle where the entry rule could not be decided must read differently
    from one where the market simply did not qualify, and both differently
    from a Risk Gate veto. `stage` alone cannot make the first distinction:
    `no_setup` covers both an unmeasured feature and a measured one that failed
    the rule, so `category` adds the split from `iv_rank` instead.
    """

    def test_traded_covers_filled_and_submitted(self) -> None:
        assert to_cycle({"stage": "filled"}).category == "traded"
        assert to_cycle({"stage": "submitted"}).category == "traded"

    def test_dry_run_is_approved_not_sent(self) -> None:
        assert to_cycle({"stage": "dry_run"}).category == "approved_not_sent"

    def test_vetoed_and_breached_are_both_a_gate_veto(self) -> None:
        assert to_cycle({"stage": "vetoed"}).category == "gate_veto"
        assert to_cycle({"stage": "breached"}).category == "gate_veto"

    def test_rejected_is_the_brokers_doing_not_the_gates(self) -> None:
        assert to_cycle({"stage": "rejected"}).category == "broker_rejected"

    def test_declined_is_the_model_saying_no_to_a_menu(self) -> None:
        assert to_cycle({"stage": "declined"}).category == "model_declined"

    def test_no_setup_with_unmeasured_iv_rank_is_undecidable(self) -> None:
        """The label and detail both render on the dashboard.

        This bucket is the one a reader is most likely to misread as "the
        market was quiet", because from the outside the two look identical. So
        the label has to name the missing input rather than the internal field,
        and the detail has to say what to run — a reader who has never opened
        this repository cannot act on the word `iv_rank`.
        """
        view = to_cycle({"stage": "no_setup", "read": {"iv_rank": None}})
        assert view.category == "undecidable"
        assert "volatility history" in view.category_label
        assert "pipeline.py iv-seed" in view.category_detail

    def test_no_candidates_is_its_own_category_whether_or_not_iv_rank_is_measured(
        self,
    ) -> None:
        """A third fact, not a rerun of the `no_setup` split.

        `no_candidates` means the entry rule already fired — a `Setup` exists —
        and the menu-building screens (pricing, freshness, spread, DTE, sizing)
        found nothing to propose. That is a menu problem, not an entry problem,
        and it is true whether or not `iv_rank` happened to be measured that
        cycle — so this category must not collapse into `undecidable` just
        because the read looks the same as a `no_setup` one would.

        Modelled on today's real fixture: a `no_candidates` cycle noting "no
        structure survived pricing, freshness, spread, DTE and sizing" — the
        number below is an arbitrary stand-in for "measured", not a value this
        test claims is correct behaviour.
        """
        measured = to_cycle({"stage": "no_candidates", "read": {"iv_rank": "12.3"}})
        unmeasured = to_cycle({"stage": "no_candidates", "read": {"iv_rank": None}})
        assert measured.category == "no_candidates"
        assert unmeasured.category == "no_candidates"

    def test_no_candidates_reads_differently_from_no_setup(self) -> None:
        """The coordinator's own fixture: a `no_setup` cycle where `iv_rank`
        was measured and simply did not clear the bar is a different fact from
        a `no_candidates` cycle where the entry fired but nothing was
        priceable — both quiet, neither the same story."""
        no_setup = to_cycle({"stage": "no_setup", "read": {"iv_rank": "12.3"}})
        no_candidates = to_cycle({"stage": "no_candidates", "read": {"iv_rank": "12.3"}})
        assert no_setup.category == "no_setup"
        assert no_candidates.category == "no_candidates"
        assert no_setup.category != no_candidates.category

    def test_no_setup_with_a_measured_iv_rank_is_a_plain_no_setup(self) -> None:
        """A market that was read and simply did not qualify — not the same
        fact as one the agent could not decide about at all."""
        view = to_cycle({"stage": "no_setup", "read": {"iv_rank": "62"}})
        assert view.category == "no_setup"
        assert view.category != "undecidable"

    def test_an_unrecognised_stage_is_other_not_a_crash(self) -> None:
        assert to_cycle({"stage": "some_future_stage"}).category == "other"

    @pytest.mark.parametrize(
        "stage",
        [
            "filled",
            "submitted",
            "dry_run",
            "vetoed",
            "breached",
            "rejected",
            "declined",
            "no_setup",
            "no_candidates",
        ],
    )
    def test_every_real_stage_has_a_non_empty_label_and_detail(self, stage: str) -> None:
        view = to_cycle({"stage": stage})
        assert view.category_label
        assert view.category_detail

    def test_a_gate_veto_and_a_market_decline_are_different_categories(self) -> None:
        """The Risk Gate stopping an order and the market never producing one
        are different facts, and a dashboard that badged them the same would
        erase the thing specs/03 exists to demonstrate."""
        vetoed = to_cycle({"stage": "vetoed"})
        quiet = to_cycle({"stage": "no_setup", "read": {"iv_rank": "10"}})
        assert vetoed.category != quiet.category

    def test_real_pipeline_cycles_classify_the_same_way(self, journal: Journal) -> None:
        """Integration with the actual pipeline, not just hand-built dicts: a
        genuine `NO_SETUP` cycle and a genuine `VETOED` one land in different
        buckets once they have been through `run_cycle`, `Journal.append` and
        `Journal.read`."""
        journal.append(cycle(with_setup=False))
        journal.append(at_stage(Stage.VETOED, sequence=1))
        journal.append(at_stage(Stage.FILLED, sequence=2))
        views = day_view(journal, DAY).cycles
        by_stage = {view.stage: view.category for view in views}
        assert by_stage["no_setup"] in ("undecidable", "no_setup")
        assert by_stage["vetoed"] == "gate_veto"
        assert by_stage["filled"] == "traded"


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
