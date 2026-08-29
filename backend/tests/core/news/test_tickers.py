"""Ticker attribution from press-release prose.

The cases here are real wire sentences, not invented ones. What is being
defended is mostly the *negative* behaviour: a foreign co-listing must not
become the alert's symbol, and prose that merely contains a colon must not
produce a ticker at all. A wrong symbol on a confident alert is worse than no
alert, because a person acts on it.
"""

from __future__ import annotations

import pytest

from alphagate.core.news.tickers import extract_tickers, primary_ticker


def test_reads_the_issuer_from_a_business_wire_lede() -> None:
    text = (
        "DUBAI, United Arab Emirates--(BUSINESS WIRE)--Uber Technologies, Inc "
        "(NYSE: UBER) today announced that ..."
    )
    assert primary_ticker(text) == "UBER"


def test_keeps_the_us_listing_and_drops_the_foreign_co_listing() -> None:
    # The failure this prevents: alerting on "9888", which nobody can trade in
    # a US account, because it happened to be mentioned second.
    text = "Baidu, Inc.'s (NASDAQ: BIDU and HKEX: 9888) fully driverless vehicles"
    assert [m.symbol for m in extract_tickers(text)] == ["BIDU"]


def test_ranks_the_issuer_ahead_of_a_partner_named_later() -> None:
    text = (
        "Moderna, Inc. (NASDAQ: MRNA) and Merck & Co., Inc. (NYSE: MRK) today "
        "announced that the Phase 3 trial met its primary endpoint."
    )
    assert [m.symbol for m in extract_tickers(text)] == ["MRNA", "MRK"]
    assert primary_ticker(text) == "MRNA"


def test_a_company_named_twice_is_ranked_by_its_first_mention() -> None:
    text = (
        "Merck & Co., Inc. (NYSE: MRK) reported. About Moderna (NASDAQ: MRNA). "
        "About Merck (NYSE: MRK)."
    )
    assert [m.symbol for m in extract_tickers(text)] == ["MRK", "MRNA"]


@pytest.mark.parametrize(
    "text",
    [
        "Pixelgen Technologies Closes $15.5M Oversubscribed Series B",
        "CONTACT: Jane Doe, Investor Relations: 555-0100",
        "The study ran at 40 sites: 12 in the EU and 28 in the US.",
        "Results at 10:30 a.m. ET",
    ],
)
def test_prose_without_a_listing_yields_nothing(text: str) -> None:
    # This is the filter doing its second job: no ticker means not a listed
    # company, which means this monitor has nothing to say about it.
    assert extract_tickers(text) == []
    assert primary_ticker(text) is None


def test_foreign_only_listings_are_dropped_entirely() -> None:
    text = "Danske Bank A/S (CPH: DANSKE) announced transactions by managers."
    assert primary_ticker(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Company (NASDAQ:ABCD) said", "ABCD"),
        ("Company (Nasdaq: abcd) said", None),  # lowercase is prose, not a symbol
        ("Company (NYSE American: XYZ) said", "XYZ"),
        ("Company (OTCQB: ABCDE) said", "ABCDE"),
        ("Berkshire (NYSE: BRK.B) said", "BRK.B"),
    ],
)
def test_venue_and_symbol_spellings(text: str, expected: str | None) -> None:
    assert primary_ticker(text) == expected


def test_a_slash_joined_dual_listing_keeps_the_us_symbol() -> None:
    # Found on a live GlobeNewswire release. The symbol belongs to the Nasdaq
    # listing; reading the venue as "TASE" and dropping it lost a real, tradeable
    # offering announcement.
    text = (
        "Odysight.ai Inc. (the “Company”) (Nasdaq/TASE: ODYS) today announced "
        "the pricing of its public offering of 3,437,500 shares."
    )
    assert primary_ticker(text) == "ODYS"


def test_a_slash_joined_listing_with_no_us_venue_is_still_dropped() -> None:
    assert primary_ticker("Foo Ltd (TASE/LSE: FOO) announced") is None
