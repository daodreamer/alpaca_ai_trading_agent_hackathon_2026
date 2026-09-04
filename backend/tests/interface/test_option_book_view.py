"""Resolving the pinned option book for the dashboard — specs/07 D1, D8.

Hermetic: every book here is written to `tmp_path`, so the suite says nothing
about whether `ai_quant_researcher/runs/option_books` happens to hold a file
today. The one thing worth testing without a fixture is `default_option_books_dir`
pointing at the right sibling path, and that gets its own narrow test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import alphagate.interface.option_book_view as option_book_view
from alphagate.interface.option_book_view import (
    OPTION_FINGERPRINT_VAR,
    default_option_books_dir,
    find_latest_option_book,
    option_book_to_json,
    resolve_pinned_option_book,
)
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION, SLEEVE_LIMITS

FINGERPRINT = "cc197008e0deb097"


def payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
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
            "sealed_window_can_refute_not_confirm": (
                "about 25 independent 28-DTE cycles: entitled to say this stopped "
                "working, never entitled to say it works (specs/10 D8)"
            ),
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
                "note": "32 independent cycles settled inside the window",
                "residual": {
                    "alpha": 0.0252,
                    "beta": 0.088,
                    "t_alpha": 1.1138,
                    "is_significant": False,
                },
            },
        },
    }
    base.update(overrides)
    return base


def write_book(directory: Path, name: str, document: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class TestFindLatestOptionBook:
    def test_picks_the_lexicographically_latest_file(self, tmp_path: Path) -> None:
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-01.json", payload())
        newest = write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        assert find_latest_option_book(tmp_path, FINGERPRINT) == newest

    def test_ignores_a_different_fingerprint(self, tmp_path: Path) -> None:
        write_book(tmp_path, "rule-deadbeef00000000-2024-08-30.json", payload())
        assert find_latest_option_book(tmp_path, FINGERPRINT) is None

    def test_a_missing_directory_is_none_not_an_error(self, tmp_path: Path) -> None:
        assert find_latest_option_book(tmp_path / "absent", FINGERPRINT) is None

    def test_an_empty_fingerprint_never_matches(self, tmp_path: Path) -> None:
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        assert find_latest_option_book(tmp_path, "") is None


class TestResolvePinnedOptionBook:
    def test_no_fingerprint_is_unavailable_with_a_reason(self, tmp_path: Path) -> None:
        """The reasons render on the dashboard, so they have to be actionable.

        Asserted against the joined text rather than `reasons[0]`: what matters
        is that a reader is told the consequence *and* the variable to set, not
        which sentence carries which.
        """
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint="")
        assert not view.available
        joined = " ".join(view.reasons)
        assert OPTION_FINGERPRINT_VAR in joined
        assert "will not trade options" in joined

    def test_no_file_for_the_pin_is_unavailable(self, tmp_path: Path) -> None:
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert not view.available
        joined = " ".join(view.reasons)
        assert FINGERPRINT in joined
        # Naming the fingerprint is not enough on its own -- the reader also
        # needs to know what to run about it.
        assert "pipeline.py option-book" in joined

    def test_a_valid_book_loads(self, tmp_path: Path) -> None:
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert view.available
        assert view.book is not None
        assert view.book.fingerprint == FINGERPRINT
        assert view.book.rule.entry.expression == "iv_rank() < 15"
        assert "specs/10 D8" in view.can_refute_not_confirm

    def test_a_refuted_book_is_refused_and_the_reason_says_so(self, tmp_path: Path) -> None:
        """specs/07 D8: the refusal is the interesting fact, and this module
        must not hide it behind an empty panel."""
        document = deepcopy(payload())
        document["provenance"]["sealed_measurement"]["refuted"] = True
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", document)
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert not view.available
        assert any("refuted" in reason for reason in view.reasons)
        # The sentence still travels even though the typed book could not be built.
        assert "specs/10 D8" in view.can_refute_not_confirm

    def test_not_yet_a_paper_position_is_refused(self, tmp_path: Path) -> None:
        document = deepcopy(payload())
        document["provenance"]["status"] = "CANDIDATE"
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", document)
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert not view.available

    def test_malformed_json_is_unavailable_not_a_crash(self, tmp_path: Path) -> None:
        path = tmp_path / f"rule-{FINGERPRINT}-2024-08-30.json"
        path.write_text("{not json", encoding="utf-8")
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert not view.available
        assert "could not be read" in view.reasons[0]

    def test_a_json_array_is_unavailable_not_a_crash(self, tmp_path: Path) -> None:
        path = tmp_path / f"rule-{FINGERPRINT}-2024-08-30.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        assert not view.available

    def test_defaults_come_from_the_environment_when_not_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        monkeypatch.setenv(OPTION_FINGERPRINT_VAR, FINGERPRINT)
        view = resolve_pinned_option_book(books_dir=tmp_path)
        assert view.available


class TestDefaultOptionBooksDir:
    def test_points_at_the_researchers_output_directory(self) -> None:
        path = default_option_books_dir()
        assert path.parts[-3:] == ("ai_quant_researcher", "runs", "option_books")


class TestOptionBookToJson:
    def test_an_unavailable_view_reports_why_and_nothing_else(self, tmp_path: Path) -> None:
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        body = option_book_to_json(view)
        assert body["available"] is False
        assert body["reasons"]

    def test_a_loaded_book_never_says_confirmed_or_validated(self, tmp_path: Path) -> None:
        """The honesty requirement, at the boundary this module owns: the wire
        shape carries `refuted` and `can_confirm` as the two separate booleans
        the domain type gives them, never a flattened verdict word."""
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        body = option_book_to_json(view)
        blob = json.dumps(body)
        assert "confirmed" not in blob.lower()
        assert "validated" not in blob.lower()
        assert body["sealed"]["refuted"] is False
        assert body["sealed"]["can_confirm"] is False
        assert body["sealed"]["is_significant"] is False
        assert body["sealed"]["t_alpha"] == pytest.approx(1.1138)
        assert body["sealed"]["significance_bar"] == pytest.approx(1.96)
        assert body["rule"]["entry_expression"] == "iv_rank() < 15"
        assert body["rule"]["risk_per_trade"] == "0.02"


class TestLiveSizingIsDistinctFromTheResearchedFraction:
    """`rule.risk_per_trade` never sizes a live trade — `agent/sizing.py` reads
    `SLEEVE_LIMITS.max_trade_loss(OPTIONS_SLEEVE_ALLOCATION)` instead, and never
    the book's own fraction. The wire shape must carry both facts rather than
    let a reader mistake one for the other.
    """

    def test_the_live_budget_comes_from_the_sleeve_limits_not_the_book(
        self, tmp_path: Path
    ) -> None:
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        rule = option_book_to_json(view)["rule"]
        # What the research ran at — a fact about a different account.
        assert rule["risk_per_trade"] == "0.02"
        # What actually binds live — sourced from the risk config, not the book.
        assert Decimal(rule["sleeve_allocation"]) == OPTIONS_SLEEVE_ALLOCATION
        assert Decimal(rule["live_trade_budget_pct"]) == SLEEVE_LIMITS.max_trade_loss_pct
        assert Decimal(rule["live_trade_budget"]) == SLEEVE_LIMITS.max_trade_loss(
            OPTIONS_SLEEVE_ALLOCATION
        )
        # The two were deliberately made to agree in dollars — specs/10 D8a.
        assert Decimal(rule["live_trade_budget"]) == Decimal("2000")

    def test_a_re_split_of_the_sleeve_moves_the_live_figure_not_the_researched_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason this is read from `risk.limits` rather than hardcoded: if
        the sleeve is ever re-split, this wire shape moves with it, and the
        researched fraction — a fact about the sealed run's own account — does
        not move at all."""
        monkeypatch.setattr(option_book_view, "OPTIONS_SLEEVE_ALLOCATION", Decimal(20_000))
        write_book(tmp_path, f"rule-{FINGERPRINT}-2024-08-30.json", payload())
        view = resolve_pinned_option_book(books_dir=tmp_path, fingerprint=FINGERPRINT)
        rule = option_book_to_json(view)["rule"]
        assert rule["sleeve_allocation"] == "20000"
        assert Decimal(rule["live_trade_budget"]) == SLEEVE_LIMITS.max_trade_loss(Decimal(20_000))
        assert rule["risk_per_trade"] == "0.02"
