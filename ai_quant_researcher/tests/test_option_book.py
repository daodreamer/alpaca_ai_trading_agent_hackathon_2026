"""The options handoff — specs/10-options-research.md D5, D10 and specs/09 D0.

The artefact is the whole interface between this project and whatever places an
order, so what is asserted here is mostly what the file is *not allowed* to
contain:

* no strike, no expiry, no contract count, no price — the rule travels and the
  executor resolves it against a live chain, because a strike named from
  yesterday's close is wrong by today's open (D5),
* no structure whose maximum loss is unbounded, checked at the file boundary as
  well as in the type, because this validator also reads books it did not write,
* and no wording that lets the sealed window claim more than it can: 25
  independent cycles can refute and cannot confirm (D8).

The refusals around it are the pre-registration protocol, and their *order* is
the protocol: registered, then sealed, then not refuted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from aqr.cli import app
from aqr.option_book import (
    OPTION_BOOK_SCHEMA_VERSION,
    OPTION_CONSUMER_MUST_SUPPLY,
    build_option_book,
    load_option_book,
    validate_option_book,
    write_option_book,
)
from aqr.options.chain import ChainIndex
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket
from aqr.options.spec import Cadence, OptionSizing, OptionSpec, StructureSpec
from aqr.registry.db import Registry
from aqr.seal import current as current_seal
from tests.test_option_engine import chain_row, credit_spread_spec, make_underlying, trading_days

runner = CliRunner()


def _text(result: object) -> str:
    """Everything the command printed, with rich's box drawing flattened."""
    raw = str(getattr(result, "output", ""))
    stripped = "".join(" " if ord(ch) > 0x2500 else ch for ch in raw)
    return " ".join(stripped.split())


def _market(sessions: int = 40) -> OptionMarket:
    days = trading_days(400)
    chosen = [d for i, d in enumerate(days) if i >= 5 and i % 5 == 0][:sessions]
    rows: list[dict[str, str]] = []
    for session in chosen:
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
        ]
    return OptionMarket(underlying=make_underlying({}, days), chain=ChainIndex.from_rows(rows))


def _spec(**overrides: Any) -> OptionSpec:
    # ``width_delta`` rather than the fixture's ``width_points``: D5 prefers it,
    # and it is the field a book has to carry correctly, because a point width
    # resolves on 23% of sessions against a delta wing's 98%.
    fields: dict[str, Any] = {
        "name": "handoff_probe",
        "hypothesis": "Index puts carry a variance risk premium.",
        "structure": StructureSpec(type="put_credit_spread", width_delta=0.06),
        "cadence": Cadence(min_sessions_between_entries=2),
        "sizing": OptionSizing(risk_per_trade=0.02, max_concurrent=3),
    }
    fields.update(overrides)
    return credit_spread_spec(**fields)


def _book(spec: OptionSpec | None = None, market: OptionMarket | None = None) -> Any:
    return build_option_book(
        spec or _spec(),
        market or _market(),
        generated_at=datetime(2026, 9, 1, tzinfo=UTC),
        dataset_version="csv:test:1D+chain:test:40sessions",
        provenance={"database": "test"},
        seal=current_seal().certificate(),
        config=OptionBacktestConfig(initial_equity=1_000_000.0, settle_before=date(2030, 1, 1)),
    )


# --------------------------------------------------------------------------- #
# What the artefact carries
# --------------------------------------------------------------------------- #


def test_the_book_carries_the_rule_and_not_the_positions() -> None:
    book = _book()
    rule = book.as_dict()["rule"]

    assert rule["structure"] == "put_credit_spread"
    assert rule["dte"]["target"] == 28
    assert rule["anchor"]["delta"] == 0.16
    assert rule["width_delta"] == 0.06
    assert rule["cadence"]["min_sessions_between_entries"] == 2
    assert rule["sizing"]["risk_per_trade"] == 0.02


def test_the_book_names_no_strike_and_no_expiry() -> None:
    """D5: the vendor resamples about 24 rungs around the money every session and
    the expiries roll, so a strike is a fact about one snapshot. The executor
    gets the rule and resolves it against a live chain, which is also why the
    backend needs its own delta selection and cannot import ``aqr.options``."""
    payload = _book().as_dict()
    rule = json.dumps(payload["rule"])
    for absent in ("strike", "expiration", "contracts", "limit_price", "premium"):
        assert absent not in rule
        assert absent not in payload


def test_the_evidence_is_measured_rather_than_restated() -> None:
    """A run, not a copy of the spec. It is what refuses a rule that cannot fire,
    and it is the only number that lets anyone notice an executor opening ten
    structures a week from a rule whose research produced four cycles a year."""
    book = _book()
    assert book.evidence["trades"] > 0
    assert book.evidence["independent_cycles"] > 0
    assert book.evidence["independent_cycles"] <= book.evidence["trades"]
    assert book.evidence["last_entry_session"] <= book.as_of


def test_the_conventions_say_there_is_no_exit_and_why() -> None:
    book = _book()
    assert "held to expiry" in book.exit_convention.lower()
    assert "early assignment" in book.exit_convention.lower()
    assert "ask" in book.fill_convention and "bid" in book.fill_convention


def test_the_boundary_travels_inside_the_artefact() -> None:
    """A boundary recorded only in a design document is one the first person to
    consume the file will not see."""
    supplied = " ".join(OPTION_CONSUMER_MUST_SUPPLY).lower()
    assert "risk gate" in supplied
    assert "kill switch" in supplied
    assert "live option chain" in supplied
    assert "early assignment" in supplied
    assert list(OPTION_CONSUMER_MUST_SUPPLY) == _book().as_dict()["consumer_must_supply"]


def test_a_rule_that_never_fires_is_refused_rather_than_handed_off() -> None:
    """A book for a rule that cannot fire gives an executor something that looks
    like a strategy and does nothing, and the first person to notice would be
    whoever asked why the account had no positions a month later."""
    with pytest.raises(ValueError, match="opened no structures"):
        _book(_spec(entry="close > 100000"))


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def test_a_written_book_reads_back_and_hashes_to_what_was_recorded(tmp_path: Path) -> None:
    book = _book()
    path = tmp_path / "book.json"
    digest = write_option_book(book, path)

    payload = load_option_book(path)
    assert payload["spec_fingerprint"] == book.spec_fingerprint
    assert payload["schema_version"] == OPTION_BOOK_SCHEMA_VERSION

    from aqr.option_book import option_book_digest

    assert option_book_digest(payload) == digest


def test_every_fault_is_reported_at_once() -> None:
    """A consumer debugging a rejected book should need one round trip."""
    with pytest.raises(ValueError) as exc:
        validate_option_book({"schema_version": 1})
    assert exc.value.args[0].count(";") >= 5


def test_an_unbounded_structure_is_refused_at_the_file_boundary() -> None:
    """``StructureKind`` has no such member, which is exactly why it is checked
    here too: this validator also reads books it did not write, and a hand-edited
    artefact naming one would ask an executor to open a position with unbounded
    loss."""
    payload = _book().as_dict()
    payload["rule"]["structure"] = "naked_put"
    with pytest.raises(ValueError, match="unbounded loss"):
        validate_option_book(payload)


def test_a_book_naming_a_strike_is_refused() -> None:
    payload = _book().as_dict()
    payload["rule"]["strikes"] = [5480.0, 5450.0]
    with pytest.raises(ValueError, match="resolved against a live chain"):
        validate_option_book(payload)


def test_a_risk_fraction_outside_what_was_measured_is_refused() -> None:
    """D8a: the cycle count is silenced by sizing before it is silenced by the
    market, so a book at a size the research never ran is a different
    experiment."""
    payload = _book().as_dict()
    payload["rule"]["sizing"]["risk_per_trade"] = 0.5
    with pytest.raises(ValueError, match="never measured this rule at that size"):
        validate_option_book(payload)


def test_an_unknown_schema_version_is_refused() -> None:
    payload = _book().as_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="cannot know what its fields mean"):
        validate_option_book(payload)


# --------------------------------------------------------------------------- #
# ``aqr option-book``: the refusals, in the order that is the protocol
# --------------------------------------------------------------------------- #


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "research.sqlite"


def _register(db: Path, spec: OptionSpec, *, sealed: dict[str, Any] | None = None) -> None:
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)
        if sealed is not None:
            reg.preregister(
                spec.fingerprint(), selection_rule="the only candidate", seal_digest="d"
            )
            reg.record_sealed_run(spec.fingerprint(), result=sealed)


def test_an_unregistered_fingerprint_is_refused(db: Path) -> None:
    result = runner.invoke(app, ["option-book", "deadbeefdeadbeef", "--db", str(db)])
    assert result.exit_code != 0
    assert "cannot be traced back" in _text(result)


def test_an_equity_fingerprint_is_sent_to_the_other_command(db: Path) -> None:
    """The two artefacts are not interchangeable: one carries weights and one
    carries a structure and a delta."""
    from aqr.dsl.schema import StrategySpec, Universe

    equity = StrategySpec(
        name="an_equity_rule",
        entry="close > ema(20)",
        universe=Universe(symbols=("SPY",), timeframe="1D"),
    )
    with Registry(db) as reg:
        reg.upsert_strategy(equity)

    result = runner.invoke(app, ["option-book", equity.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "target-book" in _text(result)


def test_a_rule_with_an_unspent_seal_is_refused(db: Path) -> None:
    """Handing off before the seal is spent would trade a rule whose
    out-of-sample verdict is still owed, which is the loophole the whole
    pre-registration protocol exists to close."""
    spec = _spec()
    _register(db, spec)

    result = runner.invoke(app, ["option-book", spec.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "no sealed run" in _text(result)


def test_a_refuted_rule_is_refused(db: Path) -> None:
    spec = _spec()
    _register(db, spec, sealed={"measurement": {"refuted": True}})

    result = runner.invoke(app, ["option-book", spec.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "refuted" in _text(result)
