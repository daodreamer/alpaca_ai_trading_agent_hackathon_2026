"""Contract selection — specs/10-options-research.md D5, test plan 1–3.

Selection is the part of an options backtest that quietly decides the answer.
"the 16-delta put at 28 DTE" names one contract on a good session and nothing at
all on a thin one, and a selector that returns *the nearest thing available*
instead of refusing turns a rule about 16-delta puts into a rule about whatever
was lying around. So the tests here are mostly about what selection must
**refuse** to do.

The other half is determinism. Two runs of the same spec on the same cache must
produce the same trades byte for byte, which means every tie in the ladder needs
a stated winner rather than whichever row the vendor happened to write first.
"""

from __future__ import annotations

from datetime import date

import pytest

from aqr.options.chain import ChainIndex, NoSuchContract, Quote, SessionChain

SESSION = date(2023, 8, 30)


def quote(
    expiration: str,
    strike: float,
    right: str,
    *,
    delta: float,
    bid: float = 1.00,
    ask: float = 1.10,
    iv: float = 0.15,
) -> Quote:
    return Quote(
        expiration=date.fromisoformat(expiration),
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        iv=iv,
        delta=delta,
        gamma=0.001,
        theta=-0.05,
        vega=0.09,
        rho=0.02,
    )


def chain(*quotes: Quote, session: date = SESSION) -> SessionChain:
    return SessionChain(session=session, quotes=tuple(quotes))


# --------------------------------------------------------------------------- #
# It resolves to exactly one contract, or it refuses
# --------------------------------------------------------------------------- #


def test_selection_returns_the_contract_nearest_the_delta_target() -> None:
    book = chain(
        quote("2023-09-29", 420.0, "put", delta=-0.09),
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 445.0, "put", delta=-0.27),
    )
    got = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert got.strike == 433.0


def test_a_put_is_matched_on_the_magnitude_of_its_delta() -> None:
    """The vendor writes puts with a negative delta and calls with a positive one.

    A rule says "the 16-delta put" and means the magnitude. Comparing the signed
    number would make every put miss by 0.32 and every tolerance a no-op.
    """
    book = chain(quote("2023-09-29", 433.0, "put", delta=-0.16))
    assert book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)


def test_a_signed_delta_target_is_an_authoring_error_not_a_thin_ladder() -> None:
    """``delta_target: -0.16`` is a spec that spelled the sign wrong.

    Answering it with ``NoSuchContract`` would be true and useless -- the author
    would read "the ladder does not have it" and go looking at the data. The
    request itself is malformed, so it raises as one and says which.
    """
    book = chain(quote("2023-09-29", 433.0, "put", delta=-0.16))
    with pytest.raises(ValueError, match="magnitude"):
        book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=-0.16)


def test_a_delta_outside_tolerance_refuses_rather_than_returning_the_nearest() -> None:
    """The whole point. A thin ladder must break the rule, not bend it."""
    book = chain(
        quote("2023-09-29", 420.0, "put", delta=-0.05),
        quote("2023-09-29", 445.0, "put", delta=-0.30),
    )
    with pytest.raises(NoSuchContract) as excinfo:
        book.select(
            right="put", dte_target=30, dte_tolerance=10, delta_target=0.16, delta_tolerance=0.06
        )
    assert "0.16" in str(excinfo.value)


def test_a_dte_outside_tolerance_refuses() -> None:
    book = chain(quote("2023-12-15", 433.0, "put", delta=-0.16))
    with pytest.raises(NoSuchContract) as excinfo:
        book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert "30" in str(excinfo.value)


def test_the_right_is_honoured() -> None:
    book = chain(quote("2023-09-29", 433.0, "call", delta=0.16))
    with pytest.raises(NoSuchContract):
        book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)


# --------------------------------------------------------------------------- #
# Ties have a stated winner
# --------------------------------------------------------------------------- #


def test_an_equal_delta_distance_is_broken_by_the_lower_strike() -> None:
    book = chain(
        quote("2023-09-29", 445.0, "put", delta=-0.20),
        quote("2023-09-29", 433.0, "put", delta=-0.12),
    )
    got = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert got.strike == 433.0


def test_an_equal_dte_distance_is_broken_by_the_earlier_expiry() -> None:
    """25 and 35 DTE are both five days from a 30-day target."""
    book = chain(
        quote("2023-10-04", 433.0, "put", delta=-0.16),  # 35 DTE
        quote("2023-09-24", 430.0, "put", delta=-0.16),  # 25 DTE
    )
    got = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert got.expiration == date(2023, 9, 24)


def test_the_expiry_is_chosen_before_the_leg() -> None:
    """Two steps, in a fixed order: expiry by DTE, then strike by delta.

    Choosing jointly would let a far better delta on a worse expiry win, which
    makes ``dte`` advisory. A rule that says 28 days means 28 days.
    """
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.20),  # 30 DTE, delta 4 off
        quote("2023-10-20", 420.0, "put", delta=-0.16),  # 51 DTE, delta exact
    )
    got = book.select(
        right="put", dte_target=30, dte_tolerance=25, delta_target=0.16, delta_tolerance=0.06
    )
    assert got.expiration == date(2023, 9, 29)


# --------------------------------------------------------------------------- #
# A quote that cannot be sold is not a short leg
# --------------------------------------------------------------------------- #


def test_a_zero_bid_cannot_be_sold() -> None:
    """4.7% of the cache has one. Selling into it is a fill at zero, not a trade."""
    book = chain(quote("2023-09-29", 433.0, "put", delta=-0.16, bid=0.0, ask=0.05))
    assert book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    with pytest.raises(NoSuchContract) as excinfo:
        book.select(
            right="put", dte_target=30, dte_tolerance=10, delta_target=0.16, sellable=True
        )
    assert "bid" in str(excinfo.value)


def test_an_unsellable_quote_does_not_shadow_a_sellable_one() -> None:
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16, bid=0.0, ask=0.05),
        quote("2023-09-29", 440.0, "put", delta=-0.19, bid=0.90, ask=1.00),
    )
    got = book.select(
        right="put", dte_target=30, dte_tolerance=10, delta_target=0.16, sellable=True
    )
    assert got.strike == 440.0


# --------------------------------------------------------------------------- #
# Selecting the far leg of a spread
# --------------------------------------------------------------------------- #


def test_the_wing_is_selected_by_points_from_the_short_strike() -> None:
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 423.0, "put", delta=-0.09),
        quote("2023-09-29", 413.0, "put", delta=-0.05),
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    wing = book.select_wing(short, width_points=10.0)
    assert wing.strike == 423.0
    assert wing.expiration == short.expiration


def test_a_wing_that_is_not_listed_refuses() -> None:
    """A vendor ladder spaced 8–15 points does not contain every width a rule
    might name, and silently widening the spread changes its maximum loss."""
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 423.0, "put", delta=-0.09),
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    with pytest.raises(NoSuchContract):
        book.select_wing(short, width_points=5.0)


def test_the_wing_is_on_the_protective_side_of_the_short_strike() -> None:
    """Below for a put, above for a call. A 'wing' on the wrong side is a
    different structure with a different maximum loss."""
    book = chain(
        quote("2023-09-29", 433.0, "call", delta=0.16),
        quote("2023-09-29", 443.0, "call", delta=0.09),
        quote("2023-09-29", 423.0, "call", delta=0.30),
    )
    short = book.select(right="call", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert book.select_wing(short, width_points=10.0).strike == 443.0


# --------------------------------------------------------------------------- #
# Quote arithmetic
# --------------------------------------------------------------------------- #


def test_a_quote_reports_its_mid_and_its_relative_spread() -> None:
    q = quote("2023-09-29", 433.0, "put", delta=-0.16, bid=1.00, ask=1.10)
    assert q.mid == pytest.approx(1.05)
    assert q.relative_spread == pytest.approx(0.10 / 1.05)


def test_a_quote_with_no_market_has_no_relative_spread() -> None:
    q = quote("2023-09-29", 433.0, "put", delta=-0.16, bid=0.0, ask=0.0)
    assert q.relative_spread is None


def test_a_crossed_quote_does_not_construct() -> None:
    with pytest.raises(ValueError, match="ask"):
        quote("2023-09-29", 433.0, "put", delta=-0.16, bid=1.20, ask=1.10)


# --------------------------------------------------------------------------- #
# The index over sessions
# --------------------------------------------------------------------------- #


def test_the_index_is_built_from_the_cache_rows_and_keeps_every_session() -> None:
    rows = [
        ("2023-08-30", "2023-09-29", "433.00", "Put", "1.00", "1.10", "0.15", "-0.16"),
        ("2023-08-30", "2023-09-29", "423.00", "Put", "0.50", "0.55", "0.16", "-0.09"),
        ("2023-09-01", "2023-09-29", "433.00", "Put", "0.90", "1.00", "0.15", "-0.14"),
    ]
    index = ChainIndex.from_rows(_as_dicts(rows))
    assert index.sessions == (date(2023, 8, 30), date(2023, 9, 1))
    assert len(index[date(2023, 8, 30)].quotes) == 2


def test_an_unknown_session_refuses_rather_than_returning_an_empty_chain() -> None:
    """An empty chain reads as 'nothing qualified today' and skips the entry.
    A missing session is a different fact and must not be spelled the same way."""
    index = ChainIndex.from_rows(
        _as_dicts([("2023-08-30", "2023-09-29", "433.00", "Put", "1.00", "1.10", "0.15", "-0.16")])
    )
    with pytest.raises(KeyError):
        index[date(2023, 8, 31)]


def test_the_index_is_ordered_regardless_of_the_order_rows_arrive_in() -> None:
    rows = [
        ("2023-09-01", "2023-09-29", "433.00", "Put", "0.90", "1.00", "0.15", "-0.14"),
        ("2023-08-30", "2023-09-29", "433.00", "Put", "1.00", "1.10", "0.15", "-0.16"),
    ]
    assert ChainIndex.from_rows(_as_dicts(rows)).sessions == (
        date(2023, 8, 30),
        date(2023, 9, 1),
    )


def test_sessions_on_or_after_a_boundary_can_be_excluded_at_index_time() -> None:
    """The engine needs to refuse entries whose *expiry* crosses the embargo
    (D3); this is the cruder cousin — not indexing what must not be read at all."""
    rows = [
        ("2023-08-30", "2023-09-29", "433.00", "Put", "1.00", "1.10", "0.15", "-0.16"),
        ("2024-09-03", "2024-10-04", "550.00", "Put", "1.00", "1.10", "0.15", "-0.16"),
    ]
    index = ChainIndex.from_rows(_as_dicts(rows), before=date(2024, 9, 1))
    assert index.sessions == (date(2023, 8, 30),)


def _as_dicts(rows: list[tuple[str, ...]]) -> list[dict[str, str]]:
    columns = ("date", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta")
    return [
        dict(zip(columns, row, strict=True))
        | {"act_symbol": "SPY", "gamma": "0.001", "theta": "-0.05", "vega": "0.09", "rho": "0.02"}
        for row in rows
    ]


# --------------------------------------------------------------------------- #
# The wing named by its own delta — the form this cache actually supports
# --------------------------------------------------------------------------- #


def test_the_wing_can_be_selected_by_its_own_delta() -> None:
    """Measured on the SPY window against a 16-delta short put: a delta wing
    resolves on 98% of sessions, an exact 10-point wing on 23%. The listed
    widths below a 16-delta strike are 8, 9, 10, 18, 25, 35 and 45 points
    depending on the session, so "ten points wide" is not a rule this data can
    express."""
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 421.0, "put", delta=-0.06),
        quote("2023-09-29", 409.0, "put", delta=-0.02),
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert book.select_wing_by_delta(short, delta_target=0.06).strike == 421.0


def test_a_delta_wing_stays_on_the_protective_side() -> None:
    """A 6-delta contract exists above the short put strike too -- it is a call's
    delta mirrored. Buying above the short put is not a spread, it is a
    different position with a different maximum loss."""
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 445.0, "put", delta=-0.06),  # above: wrong side
        quote("2023-09-29", 421.0, "put", delta=-0.07),  # below: protective
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert book.select_wing_by_delta(short, delta_target=0.06).strike == 421.0


def test_a_delta_wing_outside_tolerance_refuses() -> None:
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 421.0, "put", delta=-0.14),
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    with pytest.raises(NoSuchContract):
        book.select_wing_by_delta(short, delta_target=0.02, delta_tolerance=0.03)


def test_a_delta_wing_tie_goes_to_the_narrower_spread() -> None:
    """Two strikes equally close to the target delta: take the one nearer the
    short leg. The narrower spread is the smaller position, and a tie resolved
    by row order would make the result depend on the vendor's file layout."""
    book = chain(
        quote("2023-09-29", 433.0, "put", delta=-0.16),
        quote("2023-09-29", 425.0, "put", delta=-0.08),
        quote("2023-09-29", 415.0, "put", delta=-0.04),
    )
    short = book.select(right="put", dte_target=30, dte_tolerance=10, delta_target=0.16)
    assert book.select_wing_by_delta(short, delta_target=0.06, delta_tolerance=0.02).strike == 425.0
