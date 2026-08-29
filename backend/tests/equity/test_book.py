"""Loading a target book — specs/09 D1, test plan items 1 and 2.

The load is the only place an unvalidated strategy could get into this system,
so every one of the seven refusals gets its own test and the accumulation gets
one of its own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from alphagate.core.identifiers import ticker
from alphagate.equity import UnusableBook, load_target_book
from tests.equity.conftest import FINGERPRINT

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_BOOKS = REPO_ROOT / "ai_quant_researcher" / "runs" / "target_books"


def load(payload: Mapping[str, Any], fingerprint: str = FINGERPRINT) -> Any:
    return load_target_book(payload, pinned_fingerprint=fingerprint, digest="d1")


# --------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------- #


def test_a_valid_book_loads_with_its_provenance_intact(book_payload: dict[str, Any]) -> None:
    book = load(book_payload)
    assert book.fingerprint == FINGERPRINT
    assert book.as_of == date(2026, 8, 27)
    assert book.status == "PAPER"
    assert book.distinct_hypotheses == 324
    assert book.sealed.looks == 1
    assert book.sealed.refuted is False
    # The sealed window can refute and cannot confirm. The loader must carry that
    # distinction rather than flattening it into "passed".
    assert book.sealed.can_confirm is False


def test_weights_become_decimal_without_going_through_a_float(
    book_payload: dict[str, Any],
) -> None:
    """The next thing that happens to a weight is multiplication by equity.

    0.06 is the one that catches the mistake: `Decimal(0.06)` is
    0.059999999999999997779553950749686919152736663818359375, and multiplying
    that by an equity figure produces a target notional with a tail on it.
    """
    book = load(book_payload)
    assert book.weights[ticker("BBB")] == Decimal("0.06")
    assert all(isinstance(w, Decimal) for w in book.weights.values())


def test_the_sleeve_is_derived_rather_than_read(book_payload: dict[str, Any]) -> None:
    """`weights` and `core_weights` are the source; the sleeve is the difference.

    Reading all three from the artefact would let them disagree, and a book whose
    halves do not sum to its whole is one nobody would notice was wrong.
    """
    book = load(book_payload)
    assert book.sleeve_weights[ticker("AAA")] == Decimal("0.02")
    assert book.sleeve_weights[ticker("CCC")] == Decimal("0.04")


def test_gross_is_the_sum_of_the_weights(book_payload: dict[str, Any]) -> None:
    assert load(book_payload).gross == Decimal("0.20")


def test_age_is_measured_against_an_argument_not_a_clock(book_payload: dict[str, Any]) -> None:
    assert load(book_payload).age_days(date(2026, 8, 31)) == 4


# --------------------------------------------------------------------- #
# The seven refusals
# --------------------------------------------------------------------- #


def test_an_unknown_schema_version_is_refused(book_payload: dict[str, Any]) -> None:
    book_payload["schema_version"] = 2
    with pytest.raises(UnusableBook, match="schema_version"):
        load(book_payload)


def test_a_book_for_another_strategy_is_refused(book_payload: dict[str, Any]) -> None:
    """Test plan item 2. Otherwise valid, and refused by name.

    This is the check that makes "only the strategy the researcher validated"
    true. Without the pin, swapping the file swaps the strategy.
    """
    with pytest.raises(UnusableBook, match="pinned strategy"):
        load(book_payload, fingerprint="0000000000000000")


def test_a_candidate_has_not_earned_a_paper_position(book_payload: dict[str, Any]) -> None:
    book_payload["provenance"]["status"] = "CANDIDATE"
    with pytest.raises(UnusableBook, match="registry status"):
        load(book_payload)


def test_an_unspent_seal_is_refused(book_payload: dict[str, Any]) -> None:
    """A rule that still owes an out-of-sample verdict must not be executed."""
    book_payload["provenance"]["sealed_look"] = 0
    with pytest.raises(UnusableBook, match="seal is unspent"):
        load(book_payload)


def test_a_refuted_strategy_is_refused(book_payload: dict[str, Any]) -> None:
    book_payload["provenance"]["sealed_measurement"]["refuted"] = True
    with pytest.raises(UnusableBook, match="refuted"):
        load(book_payload)


def test_a_missing_sealed_measurement_is_refused(book_payload: dict[str, Any]) -> None:
    """Absent is not the same as clean. Nothing recorded what the seal bought."""
    del book_payload["provenance"]["sealed_measurement"]
    with pytest.raises(UnusableBook, match="no sealed_measurement"):
        load(book_payload)


def test_a_negative_weight_is_refused(book_payload: dict[str, Any]) -> None:
    """specs/09 D6. A short leg needs a locate and has unbounded loss."""
    book_payload["weights"]["BBB"] = -0.3
    with pytest.raises(UnusableBook, match="negative weights"):
        load(book_payload)


def test_leverage_is_refused(book_payload: dict[str, Any]) -> None:
    book_payload["weights"]["AAA"] = 1.5  # gross 1.6, unambiguously borrowed
    with pytest.raises(UnusableBook, match="gross exposure"):
        load(book_payload)


def test_float_noise_around_one_is_not_leverage(book_payload: dict[str, Any]) -> None:
    """A hundred weights summed as floats routinely land a shade over 1.0.

    The tolerance exists for that and for nothing else — it is three orders of
    magnitude below any leverage anybody would take on purpose.
    """
    book_payload["weights"]["AAA"] = 0.9000000000000002
    assert load(book_payload).gross > Decimal(1)


# --------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------- #


def test_every_fault_is_reported_at_once(book_payload: dict[str, Any]) -> None:
    """A book wrong in three ways should be fixed once, not discovered three
    mornings running."""
    book_payload["schema_version"] = 9
    book_payload["provenance"]["status"] = "CANDIDATE"
    book_payload["provenance"]["sealed_look"] = 0
    with pytest.raises(UnusableBook) as caught:
        load(book_payload)
    message = str(caught.value)
    assert "schema_version" in message
    assert "registry status" in message
    assert "seal is unspent" in message


def test_an_empty_book_is_not_read_as_hold_nothing(book_payload: dict[str, Any]) -> None:
    """A book with no weights is a build that failed, not an instruction to be flat.

    Reading it as "sell everything" would liquidate the account on the morning a
    data pull came back short.
    """
    book_payload["weights"] = {}
    book_payload["core_weights"] = {}
    with pytest.raises(Exception, match="failed to build"):
        load(book_payload)


def test_the_digest_is_the_callers_not_recomputed(book_payload: dict[str, Any]) -> None:
    """Hashing a re-serialised mapping would hash this process's JSON formatting
    rather than the file that was executed."""
    book = load_target_book(book_payload, pinned_fingerprint=FINGERPRINT, digest="abc123")
    assert book.digest == "abc123"


# --------------------------------------------------------------------- #
# The seam itself
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not REAL_BOOKS.is_dir(), reason="no target books generated yet")
def test_the_fixture_matches_the_real_artefact_shape() -> None:
    """The loader is tested against a fixture; the fixture is tested against reality.

    specs/09 D0 makes the file the whole interface between the two projects,
    which means a field renamed upstream is a silent break. This is the test that
    makes it a loud one — it reads whatever `aqr target-book` last wrote and
    asserts the loader can still find everything it depends on.
    """
    books = sorted(REAL_BOOKS.glob("*.json"))
    if not books:
        pytest.skip("no target books generated yet")
    payload = json.loads(books[-1].read_text(encoding="utf-8"))

    required = {
        "schema_version", "spec_fingerprint", "spec_name", "spec_version",
        "as_of", "generated_at", "dataset_version", "universe", "timeframe",
        "weights", "core_weights", "symbols_loaded", "symbols_declared",
        "provenance",
    }
    assert required <= set(payload), f"missing {sorted(required - set(payload))}"

    provenance = payload["provenance"]
    assert {"status", "sealed_look", "sealed_measurement", "distinct_hypotheses"} <= set(
        provenance
    )
    assert {"refuted", "residual"} <= set(provenance["sealed_measurement"])
    assert {"alpha", "beta", "t_alpha"} <= set(
        provenance["sealed_measurement"]["residual"]
    )

    # And it loads, against its own fingerprint.
    book = load_target_book(
        payload, pinned_fingerprint=payload["spec_fingerprint"], digest="real"
    )
    assert book.weights
    assert book.gross <= Decimal("1.005")
