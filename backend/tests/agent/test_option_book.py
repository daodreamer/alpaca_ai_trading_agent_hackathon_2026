"""Loading an option book — specs/07 D1.

The load is the only place an unvalidated option rule could get into this
system, so every refusal gets its own test and the accumulation gets one of its
own. The same discipline as `tests/equity/test_book.py`, applied to a rule
rather than to a vector of weights.

The payload fixture is shaped like the real artefact `aqr option-book` writes,
not like whatever the loader happens to need, and
`test_the_fixture_matches_the_real_artefact_shape` checks it against the
committed book so a schema change upstream fails this suite rather than a
morning's trading.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from alphagate.agent.option_book import (
    MEASURABLE_FEATURES,
    EntryRule,
    UnusableOptionBook,
    entry_refusal,
    load_option_book,
    measurable_read,
)
from alphagate.core.identifiers import ticker

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_BOOKS = REPO_ROOT / "ai_quant_researcher" / "runs" / "option_books"

FINGERPRINT = "cc197008e0deb097"


@pytest.fixture
def payload() -> dict[str, Any]:
    """The real rule, trimmed to the fields the loader reads.

    `iv_rank() < 15` is the entry the LLM search actually produced and the
    sealed run actually measured, so the happy path here is the rule the system
    is meant to trade rather than an invented one.
    """
    return {
        "schema_version": 1,
        "spec_fingerprint": FINGERPRINT,
        "spec_name": "iv_rank_low_sticky_put_credit_spread_v1",
        "spec_version": 1,
        "underlying": "SPY",
        "as_of": "2024-08-30",
        "generated_at": "2026-09-01T12:49:50.808065+00:00",
        "dataset_version": "csv:data-options-underlying:1D+chain:data-options:753sessions",
        "exit_convention": "Held to expiry.",
        "rule": {
            "structure": "put_credit_spread",
            "entry": "iv_rank() < 15",
            "dte": {"target": 14, "tolerance": 10},
            "anchor": {"delta": 0.16, "tolerance": 0.06},
            "width_delta": 0.08,
            "width_points": None,
            "call_anchor_delta": None,
            "call_width_delta": None,
            "cadence": {"min_sessions_between_entries": 1},
            "sizing": {"risk_per_trade": 0.02, "max_concurrent": 3},
        },
        "provenance": {
            "status": "PAPER",
            "hypothesis": "Sticky demand for downside protection in calm regimes.",
            "campaign_hypotheses": 41,
            "distinct_option_hypotheses": 172,
            "sealed_look": 1,
            "sealed_looks_total": 1,
            "preregistration": {"selection_rule": "the only ACCEPT among 40 hypotheses"},
            "sealed_measurement": {
                "strategy_return": 0.0814,
                "strategy_sharpe": 1.1457,
                "benchmark_sharpe": 1.0166,
                "max_drawdown": -0.0205,
                "trades": 85,
                "observations": 500,
                "refuted": False,
                "can_confirm": False,
                "significance_bar": 1.96,
                "first_session": "2024-09-03T04:00:00+00:00",
                "last_session": "2026-08-31T04:00:00+00:00",
                "note": "this window can refute and cannot confirm",
                "residual": {
                    "alpha": 0.0252,
                    "beta": 0.088,
                    "t_alpha": 1.1138,
                    "is_significant": False,
                },
            },
        },
    }


def load(book: Mapping[str, Any], fingerprint: str = FINGERPRINT) -> Any:
    return load_option_book(book, pinned_fingerprint=fingerprint, digest="d1")


def broken(book: dict[str, Any], **rule: Any) -> dict[str, Any]:
    """The payload with one thing changed inside `rule`."""
    out = deepcopy(book)
    out["rule"].update(rule)
    return out


# --------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------- #


def test_a_valid_book_loads_with_its_rule_and_provenance_intact(
    payload: dict[str, Any],
) -> None:
    book = load(payload)
    assert book.fingerprint == FINGERPRINT
    assert book.as_of == date(2024, 8, 30)
    assert book.status == "PAPER"
    assert book.underlying == ticker("SPY")
    assert book.rule.structure == "put_credit_spread"
    assert book.rule.dte_target == 14
    assert book.rule.anchor_delta == pytest.approx(0.16)
    assert book.rule.width_delta == pytest.approx(0.08)
    assert book.rule.max_concurrent == 3
    # Money is Decimal, and exactly the written value -- Decimal(0.02) via float
    # would be 0.0200000000000000004163336342344337.
    assert book.rule.risk_per_trade == Decimal("0.02")


def test_the_sealed_run_is_carried_without_being_promoted_to_a_pass(
    payload: dict[str, Any],
) -> None:
    """"Was not refuted" and "was confirmed" are different claims, and the option
    window (about 25 independent cycles, specs/10 D8) only supports the first.
    The loader must carry both flags rather than flatten them into a verdict."""
    sealed = load(payload).sealed
    assert sealed.refuted is False
    assert sealed.can_confirm is False
    assert sealed.is_significant is False
    assert sealed.t_alpha == pytest.approx(1.1138)
    assert sealed.t_alpha < sealed.significance_bar


def test_the_dte_window_never_reaches_zero_days(payload: dict[str, Any]) -> None:
    """14 +/- 10 is 4..24, and a tolerance that swallowed the target would
    otherwise re-admit 0DTE, which specs/03 excludes."""
    assert load(payload).rule.dte_window() == (4, 24)
    wide = broken(payload, dte={"target": 5, "tolerance": 30})
    assert load(wide).rule.dte_window()[0] == 1


# --------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------- #


def test_an_unknown_schema_version_is_refused(payload: dict[str, Any]) -> None:
    payload["schema_version"] = 2
    with pytest.raises(UnusableOptionBook, match="schema_version"):
        load(payload)


def test_a_book_for_another_fingerprint_is_refused(payload: dict[str, Any]) -> None:
    """The pin is the only checkable meaning of "it trades the rule the research
    validated". Believing the book's own name would let a swapped file swap the
    strategy with nothing noticing."""
    with pytest.raises(UnusableOptionBook, match="pinned rule"):
        load(payload, fingerprint="0000000000000000")


@pytest.mark.parametrize("status", ["CANDIDATE", "REJECTED", "", "RETIRED"])
def test_only_a_promoted_rule_may_be_executed(payload: dict[str, Any], status: str) -> None:
    payload["provenance"]["status"] = status
    with pytest.raises(UnusableOptionBook, match="registry status"):
        load(payload)


def test_an_unspent_seal_is_refused(payload: dict[str, Any]) -> None:
    payload["provenance"]["sealed_look"] = 0
    with pytest.raises(UnusableOptionBook, match="seal is unspent"):
        load(payload)


def test_a_refuted_rule_is_refused(payload: dict[str, Any]) -> None:
    payload["provenance"]["sealed_measurement"]["refuted"] = True
    with pytest.raises(UnusableOptionBook, match="refuted this rule"):
        load(payload)


def test_a_structure_this_executor_cannot_build_is_refused_at_load(
    payload: dict[str, Any],
) -> None:
    """Refused here rather than later. A book asking for a long put would
    otherwise produce an empty menu every cycle and be indistinguishable from a
    quiet market in the journal."""
    with pytest.raises(UnusableOptionBook, match="not one this executor builds"):
        load(broken(payload, structure="long_put"))


def test_a_protective_leg_at_or_inside_the_anchor_is_refused(
    payload: dict[str, Any],
) -> None:
    """This single check is the whole of "defined risk" at this boundary: a wing
    that is not further out of the money does not cap the loss, whatever the
    book's `structure` field claims (CLAUDE.md rule 6)."""
    with pytest.raises(UnusableOptionBook, match="strictly less than anchor delta"):
        load(broken(payload, width_delta=0.16))
    with pytest.raises(UnusableOptionBook, match="strictly less than anchor delta"):
        load(broken(payload, width_delta=0.30))


def test_a_points_width_is_refused(payload: dict[str, Any]) -> None:
    with pytest.raises(UnusableOptionBook, match="width_points"):
        load(broken(payload, width_points=10.0))


@pytest.mark.parametrize("risk", [0.0, -0.02, 0.5, 0.06])
def test_a_size_the_research_never_ran_at_is_refused(
    payload: dict[str, Any], risk: float
) -> None:
    """specs/10 D8a: the same rule produced 21 independent cycles at 1% of equity
    and 57 at 2%. Sizing is part of the experiment, so an out-of-range fraction
    is refused rather than clamped."""
    with pytest.raises(UnusableOptionBook, match="risk_per_trade"):
        load(broken(payload, sizing={"risk_per_trade": risk, "max_concurrent": 3}))


def test_a_naive_timestamp_is_refused(payload: dict[str, Any]) -> None:
    payload["generated_at"] = "2026-09-01T12:49:50"
    with pytest.raises(UnusableOptionBook, match="no timezone"):
        load(payload)


def test_every_fault_is_reported_at_once(payload: dict[str, Any]) -> None:
    """A book wrong in four ways should be fixed once, not discovered four
    mornings running."""
    payload["schema_version"] = 99
    payload["provenance"]["status"] = "CANDIDATE"
    payload["provenance"]["sealed_look"] = 0
    payload["rule"]["width_delta"] = 0.9
    with pytest.raises(UnusableOptionBook) as caught:
        load(payload)
    message = str(caught.value)
    for expected in ("schema_version", "registry status", "seal is unspent", "anchor delta"):
        assert expected in message


# --------------------------------------------------------------------- #
# The entry rule, which is where this executor's reach is narrower
# --------------------------------------------------------------------- #


def test_an_entry_naming_an_unmeasurable_feature_is_refused(
    payload: dict[str, Any],
) -> None:
    """The researcher had a seven-year vendor volatility history and could
    condition on `term_slope()` and `skew_25d()`. This account cannot compute
    either. Executing such a rule against a substitute would run a different
    strategy under a fingerprint certifying this one."""
    with pytest.raises(UnusableOptionBook, match="cannot measure"):
        load(broken(payload, entry="term_slope() > 0.01"))
    with pytest.raises(UnusableOptionBook, match="cannot measure"):
        load(broken(payload, entry="iv_rank() < 15 and skew_25d() > 0.05"))


def test_an_expression_outside_the_supported_grammar_is_refused(
    payload: dict[str, Any],
) -> None:
    """Not a second implementation of the researcher's DSL. What it cannot
    faithfully evaluate, it refuses."""
    for expression in ("iv_rank() < 15 or hv_rank() > 50", "not iv_rank() < 15"):
        with pytest.raises(UnusableOptionBook, match="mis-associating a disjunction"):
            load(broken(payload, entry=expression))
    with pytest.raises(UnusableOptionBook, match=re.escape("feature() OP number")):
        load(broken(payload, entry="iv_rank() < iv_percentile()"))


def test_an_empty_entry_is_refused(payload: dict[str, Any]) -> None:
    with pytest.raises(UnusableOptionBook, match="no entry condition"):
        load(broken(payload, entry=""))


def test_the_entry_fires_only_where_the_research_said_it_would(
    payload: dict[str, Any],
) -> None:
    rule = load(payload).rule.entry
    assert rule.features() == {"iv_rank"}

    fires, why = rule.decide({"iv_rank": Decimal("12")})
    assert fires is True
    assert "iv_rank() < 15" in why

    fires, why = rule.decide({"iv_rank": Decimal("15")})
    assert fires is False
    assert "not < 15" in why

    fires, why = rule.decide({"iv_rank": Decimal("47.1")})
    assert fires is False


def test_an_unmeasured_feature_declines_and_says_so(payload: dict[str, Any]) -> None:
    """The one that decides whether this rule can trade at all this week.

    `iv_rank` is `None` until the IV store holds MIN_HISTORY sessions, and an
    undecidable entry is not a false one. It must decline *and* be legible in
    the journal as "could not decide", never as "the market was quiet".
    """
    rule = load(payload).rule.entry
    fires, why = rule.decide({"iv_rank": None})
    assert fires is False
    assert "unmeasured" in why
    assert "guessing" in why


def test_a_multi_clause_entry_requires_every_clause(payload: dict[str, Any]) -> None:
    rule = load(broken(payload, entry="iv_rank() < 15 and hv_rank() >= 20")).rule.entry
    assert rule.features() == {"iv_rank", "hv_rank"}
    assert rule.decide({"iv_rank": Decimal("10"), "hv_rank": Decimal("25")})[0] is True
    assert rule.decide({"iv_rank": Decimal("10"), "hv_rank": Decimal("5")})[0] is False
    # One measured and one not is still undecidable, not false-because-measured.
    fires, why = rule.decide({"iv_rank": Decimal("10"), "hv_rank": None})
    assert fires is False
    assert "unmeasured" in why


def test_measurable_read_reports_absent_and_unmeasured_identically() -> None:
    """A rule treats both as undecidable, so the projection must not let a
    consumer tell them apart and act on the difference."""

    class Read:
        iv_rank = None
        hv_rank = Decimal("30")

    projected = measurable_read(Read())
    assert set(projected) == set(MEASURABLE_FEATURES)
    assert projected["iv_rank"] is None
    assert projected["iv_vs_hv"] is None  # absent attribute
    assert projected["hv_rank"] == Decimal("30")


# --------------------------------------------------------------------- #
# The seam is a file, and this is the check that it still fits
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not REAL_BOOKS.exists(), reason="no option book written yet")
def test_the_fixture_matches_the_real_artefact_shape(payload: dict[str, Any]) -> None:
    """The real book, loaded by the real loader.

    `ai_quant_researcher` and `alphagate` never import each other, so a field
    renamed upstream is invisible until something reads the file. This is that
    something, and it fails here rather than at 09:30.
    """
    books = sorted(REAL_BOOKS.glob("*.json"))
    if not books:
        pytest.skip("no option book written yet")
    raw = json.loads(books[-1].read_text(encoding="utf-8"))

    book = load_option_book(
        raw, pinned_fingerprint=str(raw["spec_fingerprint"]), digest="real"
    )
    assert book.rule.structure in {"put_credit_spread", "call_credit_spread", "iron_condor"}
    assert book.rule.risk_per_trade > 0
    assert book.sealed.can_confirm is False

    # Every key the fixture claims the artefact has, the artefact has.
    for key in payload:
        assert key in raw, f"fixture invents {key!r}, which the real book does not carry"
    for key in payload["rule"]:
        assert key in raw["rule"], f"fixture invents rule.{key}"
    for key in payload["provenance"]:
        assert key in raw["provenance"], f"fixture invents provenance.{key}"


def test_an_entry_rule_is_frozen() -> None:
    """Determinism: nothing downstream may retune the condition it was handed."""
    rule = EntryRule(expression="iv_rank() < 15", clauses=(("iv_rank", "<", 15.0),))
    with pytest.raises((AttributeError, TypeError)):
        rule.expression = "iv_rank() < 90"  # type: ignore[misc]


# --------------------------------------------------------------------- #
# Seeding the input the rule conditions on
# --------------------------------------------------------------------- #


def test_a_vendor_history_seeds_the_rank_the_research_meant(tmp_path: Path) -> None:
    """The trailing-year window is the whole point, not a detail.

    `options/volatility.py` ranks against every observation it holds, and the
    researched rule meant the vendor's own one-year range. Seed seven years and
    `iv_rank` ranks against a window containing March 2020 — a different number
    under the same name, which is exactly the substitution specs/07 D3 refuses
    for a feature it cannot measure and must equally refuse for one it can.
    """
    from datetime import date as _date
    from datetime import timedelta

    from alphagate.agent.iv_store import IvHistoryStore
    from alphagate.options.volatility import iv_rank

    store = IvHistoryStore(directory=tmp_path)
    spy = ticker("SPY")

    last = _date(2026, 8, 28)
    # A year of quiet, one panic spike two years back, and today at the bottom.
    rows = [{"date": (last - timedelta(days=800)).isoformat(), "iv_current": "0.85"}]
    rows += [
        {"date": (last - timedelta(days=n)).isoformat(), "iv_current": f"{0.11 + n * 0.0004:.4f}"}
        for n in range(360, 0, -1)
    ]

    added = store.seed_from_vendor_history(spy, rows, since=last - timedelta(days=365))
    assert added == 360, "the 800-day-old spike must be outside the window"

    history = store.observations(spy)
    ranked = iv_rank(0.1158, history)
    assert ranked is not None
    assert ranked < 0.15, "current IV near the year's low must rank low"

    # The same current value against the unwindowed history ranks differently,
    # which is the mistake this window exists to prevent.
    unwindowed = IvHistoryStore(directory=tmp_path / "all")
    unwindowed.seed_from_vendor_history(spy, rows)
    wide = iv_rank(0.1158, unwindowed.observations(spy))
    assert wide is not None
    assert wide < ranked, "a spike in the window compresses every later rank"


def test_seeding_is_idempotent_and_skips_vendor_blanks(tmp_path: Path) -> None:
    """The agent runs many cycles a session and a re-pull re-reads the file. A
    blank is a missing observation, not a zero: recording 0.0 would put an
    impossible low into the range and flatter every rank after it."""
    from alphagate.agent.iv_store import IvHistoryStore

    store = IvHistoryStore(directory=tmp_path)
    spy = ticker("SPY")
    rows = [
        {"date": "2026-08-26", "iv_current": "0.1136"},
        {"date": "2026-08-27", "iv_current": ""},
        {"date": "2026-08-28", "iv_current": "not-a-number"},
        {"date": "2026-08-31", "iv_current": "0.1158"},
    ]
    assert store.seed_from_vendor_history(spy, rows) == 2
    assert store.seed_from_vendor_history(spy, rows) == 0
    assert store.observations(spy) == [0.1136, 0.1158]


class TestTheRulesOwnCaps:
    """`max_concurrent` and `min_sessions_between_entries` were parsed, printed
    on two pages, and enforced by nothing.

    They are the researched rule's own limits, not the account's. The Gate's
    caps are about what this account can survive (specs/03); these are about
    what was actually measured — a rule validated at three concurrent positions
    and one entry a session is not the same rule at five and three, whatever the
    Gate allows. On 2026-09-02 the agent opened two spreads in one session while
    the dashboard printed "cadence: at most one entry per session" beside them.

    Refusing is `NO_SETUP` with a reason, not a veto: nothing was proposed, so
    there is nothing for the Gate to refuse, and a judge reading the day needs
    "the rule's cadence said no" to look different from "the market did not
    qualify".
    """

    def rule(self, payload: dict[str, Any], **cadence: int) -> Any:
        """The real rule, with its own caps or a changed spacing."""
        book = deepcopy(payload)
        if cadence:
            book["rule"]["cadence"] = dict(book["rule"]["cadence"], **cadence)
        return load(book).rule

    def test_room_under_both_caps_is_no_refusal(self, payload: dict[str, Any]) -> None:
        assert entry_refusal(self.rule(payload), open_structures=2, sessions_since_entry=1) == ""

    def test_the_concurrency_cap_is_the_rules_not_the_gates(
        self, payload: dict[str, Any]
    ) -> None:
        """Three concurrent is what the sealed run measured. The Gate would
        allow eight, and its book-heat budget stops at five — neither is the
        number this rule was validated at."""
        refusal = entry_refusal(
            self.rule(payload), open_structures=3, sessions_since_entry=9
        )
        assert "3 concurrent" in refusal
        assert entry_refusal(
            self.rule(payload), open_structures=4, sessions_since_entry=9
        ) != ""

    def test_a_second_entry_in_one_session_is_refused(
        self, payload: dict[str, Any]
    ) -> None:
        """The one that already happened. `min_sessions_between_entries` is 1,
        so an entry today is the session's entry."""
        refusal = entry_refusal(
            self.rule(payload), open_structures=0, sessions_since_entry=0
        )
        assert "session" in refusal

    def test_the_next_session_is_allowed(self, payload: dict[str, Any]) -> None:
        assert entry_refusal(
            self.rule(payload), open_structures=0, sessions_since_entry=1
        ) == ""

    def test_a_wider_spacing_holds_for_longer(self, payload: dict[str, Any]) -> None:
        wide = self.rule(payload, min_sessions_between_entries=3)
        assert entry_refusal(wide, open_structures=0, sessions_since_entry=2) != ""
        assert entry_refusal(wide, open_structures=0, sessions_since_entry=3) == ""

    def test_never_having_entered_is_not_a_spacing_problem(
        self, payload: dict[str, Any]
    ) -> None:
        """`None` is "no entry on record", which is not zero sessions ago."""
        assert entry_refusal(
            self.rule(payload), open_structures=0, sessions_since_entry=None
        ) == ""

    def test_the_concurrency_cap_is_checked_first(
        self, payload: dict[str, Any]
    ) -> None:
        """Both refuse; the one that is about risk on the book is the one worth
        printing."""
        refusal = entry_refusal(
            self.rule(payload), open_structures=3, sessions_since_entry=0
        )
        assert "concurrent" in refusal

