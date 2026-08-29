"""The L0 keyword screen.

Two properties matter and they pull in opposite directions. The screen must not
drop a real event — a missed FDA approval is the failure the whole feature
exists to prevent. And it must reject the conference-presentation releases that
share the same vocabulary, because those are the bulk of a biotech wire and
every one of them costs a model call and a place in the queue.
"""

from __future__ import annotations

import pytest

from alphagate.core.news.screening import NewsCategory, screen


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        (
            "FDA Approves LUMAKRAS for Previously Treated Patients",
            NewsCategory.FDA_DECISION,
        ),
        (
            "Sarepta Receives Complete Response Letter from FDA for SRP-9003",
            NewsCategory.FDA_DECISION,
        ),
        (
            "Moderna and Merck Announce mRNA-4157 Met Primary Endpoint in Phase 3",
            NewsCategory.CLINICAL_RESULT,
        ),
        (
            "Company Reports Topline Results from Pivotal Trial",
            NewsCategory.CLINICAL_RESULT,
        ),
        (
            "Acme Corp Enters Definitive Agreement to Acquire Beta Inc for $4.2B",
            NewsCategory.MERGER_ACQUISITION,
        ),
        (
            "Biotech Announces Pricing of $75 Million Public Offering",
            NewsCategory.OFFERING,
        ),
    ],
)
def test_the_four_watched_event_kinds_pass(headline: str, expected: NewsCategory) -> None:
    hit = screen(headline)
    assert hit is not None
    assert expected in hit.categories


@pytest.mark.parametrize(
    "headline",
    [
        "Acme Appoints Jane Doe to Board of Directors",
        "Biotech to Present Phase 3 Data at ASCO 2026 Annual Meeting",
        "Company Will Present at the Jefferies Healthcare Conference",
        "Notice of Annual Meeting of Stockholders",
        "Acme Announces Third Quarter Conference Call to Discuss Results",
        "Pixelgen Technologies Closes $15.5M Oversubscribed Series B",
        "Duolingo Appoints Sallie Krawcheck to Board of Directors",
    ],
)
def test_vocabulary_without_an_event_is_rejected(headline: str) -> None:
    # "to Present Phase 3 Data" is the expensive one: it matches CLINICAL_RESULT
    # on keywords alone and is announced constantly.
    assert screen(headline) is None


def test_a_story_can_be_two_kinds_at_once() -> None:
    headline = (
        "Acme Enters Definitive Agreement to Acquire Beta, Whose Lead Asset Met "
        "Its Primary Endpoint in Phase 3"
    )
    hit = screen(headline)
    assert hit is not None
    assert hit.categories == {
        NewsCategory.MERGER_ACQUISITION,
        NewsCategory.CLINICAL_RESULT,
    }


def test_the_screen_reports_what_matched() -> None:
    # An alert a person did not expect has to be explainable without rerunning
    # the pipeline, so the matched text travels with the hit.
    hit = screen("FDA Approves New Therapy")
    assert hit is not None
    assert any("FDA" in m for m in hit.matched)


def test_direction_is_never_inferred_from_keywords() -> None:
    # Both of these are CLINICAL_RESULT and nothing more. A screen that called
    # one bullish would eventually call a failed trial a success, because the
    # words are nearly identical.
    good = screen("Trial Met Its Primary Endpoint")
    bad = screen("Trial Did Not Meet Its Primary Endpoint")
    assert good is not None
    assert bad is not None
    assert good.categories == bad.categories == {NewsCategory.CLINICAL_RESULT}


# ----------------------------------------------------- categories for tech --
#
# The first four categories were chosen for biotech, where the material events
# are regulatory and clinical. Technology issuers move on a different set, and
# adding their feeds without adding their events would mean polling a hundred
# semiconductor releases a day and only ever seeing the acquisitions.


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        (
            "Acme Raises Full Year 2026 Revenue Guidance",
            NewsCategory.GUIDANCE,
        ),
        (
            "Chipmaker Lowers Q3 Outlook on Weaker Data Center Demand",
            NewsCategory.GUIDANCE,
        ),
        (
            "Acme Announces Preliminary Third Quarter Results Above Prior Outlook",
            NewsCategory.GUIDANCE,
        ),
        (
            "Aerospace Corp Awarded $1.2 Billion U.S. Air Force Contract",
            NewsCategory.MAJOR_CONTRACT,
        ),
        (
            "Acme Receives $450 Million Order for AI Accelerator Systems",
            NewsCategory.MAJOR_CONTRACT,
        ),
        (
            "Storage Vendor Files for Chapter 11 Bankruptcy Protection",
            NewsCategory.BANKRUPTCY,
        ),
        (
            "Acme Receives Nasdaq Notice of Non-Compliance With Listing Rule",
            NewsCategory.BANKRUPTCY,
        ),
        (
            "Acme Corp to Join S&P 500 Effective September 22",
            NewsCategory.INDEX_CHANGE,
        ),
    ],
)
def test_the_events_that_move_a_tech_stock_in_a_day(headline: str, expected: NewsCategory) -> None:
    hit = screen(headline)
    assert hit is not None
    assert expected in hit.categories


@pytest.mark.parametrize(
    "headline",
    [
        # Product and partnership announcements are the bulk of a tech wire and
        # almost never move a stock. Letting them through would drown the
        # signal the other categories exist to surface.
        "Acme Launches Next-Generation AI Inference Platform",
        "Acme Announces Strategic Partnership With Cloud Provider",
        "Acme Named a Leader in the 2026 Analyst Magic Quadrant",
        "Acme Unveils New Storage Array at Industry Summit",
        "Acme Expands Data Center Footprint in Texas",
        "Acme Wins Best Innovation Award for Its Chip Design",
        # Guidance vocabulary without a change to guidance.
        "Acme to Report Fourth Quarter Results on February 10",
    ],
)
def test_routine_tech_public_relations_is_rejected(headline: str) -> None:
    assert screen(headline) is None


def test_guidance_direction_is_still_not_inferred() -> None:
    up = screen("Acme Raises Full Year Revenue Guidance")
    down = screen("Acme Lowers Full Year Revenue Guidance")
    assert up is not None
    assert down is not None
    assert up.categories == down.categories == {NewsCategory.GUIDANCE}


@pytest.mark.parametrize(
    "headline",
    [
        # Found live: this one reached a phone. Trial *operations* share every
        # keyword with trial *results* and announce nothing about whether the
        # drug works, which is the only thing that moves the stock.
        "Revelation Biosciences Selects Avance Clinical as CRO for Phase 2/3 TITAN Study",
        "Acme Doses First Patient in Phase 3 Trial of ACME-101",
        "Acme Completes Enrollment in Pivotal Phase 3 Study",
        "Acme Initiates Phase 2 Trial in Solid Tumors",
        "Acme Announces First Patient Enrolled in Phase 1 Study",
        "Acme Expands Phase 3 Trial to Additional Sites in Europe",
        "Acme Receives IRB Approval to Begin Phase 2 Study",
    ],
)
def test_trial_operations_are_not_trial_results(headline: str) -> None:
    assert screen(headline) is None


def test_an_actual_readout_still_passes() -> None:
    # The guard above must not swallow the event it sits next to.
    hit = screen("Acme Announces Positive Topline Results From Phase 3 Trial")
    assert hit is not None
    assert NewsCategory.CLINICAL_RESULT in hit.categories
