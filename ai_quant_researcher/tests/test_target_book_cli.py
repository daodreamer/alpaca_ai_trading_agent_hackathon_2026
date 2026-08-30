"""``aqr target-book``: the runner, its refusals, and the provenance it records.

There is no ``--dry-run`` here and the tests do not look for one. Nothing in this
project sends anything anywhere, so a dry run and a real run would be the same
run -- the flag would exist only to imply that its absence means something.

What the tests do check is the order of the refusals, because that order is the
protocol. A book is written for a strategy the registry knows, whose seal has
been spent, and whose sealed run did not refute it. Producing one for an
undeclared candidate would read the embargoed years for a rule whose
out-of-sample verdict is still owed, which is the loophole pre-registration
exists to close.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aqr.cli import app
from aqr.dsl.schema import StrategySpec, spec_from_dict
from aqr.registry.db import Registry
from aqr.target_book import validate_book

runner = CliRunner()


def _text(result: object) -> str:
    """Everything the command printed, with rich's box drawing flattened.

    Typer wraps an error in a panel and hard-wraps it at the terminal width, so
    a message asserted on verbatim would break the day the message got longer.
    """
    raw = str(getattr(result, "output", ""))
    stripped = "".join(" " if ord(ch) > 0x2500 else ch for ch in raw)
    return " ".join(stripped.split())


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
WINDOW = ["--source", "synthetic", "--start", "2015-01-01", "--end", "2020-01-01"]


def _spec() -> StrategySpec:
    return spec_from_dict(
        {
            "strategy": {
                "name": "handoff_probe",
                "hypothesis": "Leaders keep leading over a month.",
                "mode": "portfolio",
                "rank_by": "roc(20)",
                "hold": 2,
                "rebalance_every": 20,
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
                "sleeve": {"budget": 0.20, "idle": "benchmark"},
            }
        }
    )


def _measurement(refuted: bool) -> dict[str, object]:
    """A sealed result shaped like the real one, in the two states that matter."""
    residual = {"alpha": -0.2, "t_alpha": -3.0} if refuted else {"alpha": 0.16, "t_alpha": 2.2}
    return {
        "measurement": {
            "strategy": "handoff_probe",
            "observations": 498,
            "refuted": refuted,
            "residual": residual,
        },
        "seal": {"digest": "a" * 64, "tainted": False, "phase": "sealed"},
    }


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Path]:
    """A registry holding one strategy whose seal has been spent, not refuted."""
    path = tmp_path / "research.sqlite"
    with Registry(path) as reg:
        fingerprint = reg.upsert_strategy(_spec())
        reg.preregister(
            fingerprint,
            selection_rule="the only candidate in this test",
            seal_digest="b" * 64,
        )
        reg.record_sealed_run(fingerprint, result=_measurement(refuted=False))
    yield path


def _run(db: Path, out: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "target-book",
            _spec().fingerprint(),
            "--db",
            str(db),
            "--out",
            str(out),
            "--universe",
            "",
            *WINDOW,
            *extra,
        ],
    )


# --------------------------------------------------------------------------
# 4.3 -- the runner


def test_it_writes_a_book_and_prints_its_path(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "books"
    result = _run(db, out)
    assert result.exit_code == 0, result.stdout

    written = list(out.glob("*.json"))
    assert len(written) == 1
    assert written[0].name.startswith("handoff_probe-")
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    validate_book(payload)
    assert payload["spec_fingerprint"] == _spec().fingerprint()
    assert payload["weights"]


def test_the_book_carries_the_sealed_run_that_justified_it(db: Path, tmp_path: Path) -> None:
    """4.4. A book handed off is traceable to the hypothesis, the campaign and
    the sealed run, without trusting a write-up to have repeated them."""
    out = tmp_path / "books"
    assert _run(db, out).exit_code == 0

    payload = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["sealed_measurement"]["observations"] == 498
    assert provenance["sealed_measurement"]["refuted"] is False
    assert provenance["preregistration"]["selection_rule"].startswith("the only candidate")
    assert provenance["hypothesis"].startswith("Leaders keep leading")
    # The denominator travels with the book: a rule selected out of many trials
    # is a weaker claim than the same rule selected out of one.
    assert "distinct_hypotheses" in provenance


def test_the_book_is_recorded_against_the_fingerprint(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "books"
    assert _run(db, out).exit_code == 0

    with Registry(db) as reg:
        rows = reg.target_books(_spec().fingerprint())
    assert len(rows) == 1
    written = next(out.glob("*.json"))
    assert Path(rows[0]["path"]) == written
    assert rows[0]["as_of"] == rows[0]["book"]["as_of"]

    listing = runner.invoke(app, ["target-books", "--db", str(db)])
    assert listing.exit_code == 0
    # The digest is what the listing is for: it is how a file on disk is matched
    # back to the row that recorded it. Asserted on a prefix because rich
    # truncates a wide table to the terminal it is printed into.
    assert rows[0]["digest"][:8] in _text(listing)


def test_the_recorded_digest_detects_a_book_edited_afterwards(
    db: Path, tmp_path: Path
) -> None:
    out = tmp_path / "books"
    assert _run(db, out).exit_code == 0
    written = next(out.glob("*.json"))

    with Registry(db) as reg:
        recorded = reg.target_books(_spec().fingerprint())[0]["digest"]

    from aqr.target_book import book_digest

    assert book_digest(json.loads(written.read_text(encoding="utf-8"))) == recorded
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["weights"] = {"SPY": 1.0}
    written.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    assert book_digest(json.loads(written.read_text(encoding="utf-8"))) != recorded


def test_two_handoffs_are_two_rows_not_an_overwrite(db: Path, tmp_path: Path) -> None:
    """If the second book differs from the first, something between the spec and
    the artefact changed, and overwriting is the one thing that would hide it."""
    out = tmp_path / "books"
    assert _run(db, out).exit_code == 0
    assert _run(db, out).exit_code == 0

    with Registry(db) as reg:
        assert len(reg.target_books(_spec().fingerprint())) == 2


# --------------------------------------------------------------------------
# The refusals


def test_an_unregistered_strategy_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.sqlite"
    with Registry(empty):
        pass
    result = _run(empty, tmp_path / "books")
    assert result.exit_code != 0
    assert "in the registry" in _text(result)
    assert not (tmp_path / "books").exists()


def test_a_candidate_whose_seal_is_unspent_is_refused(tmp_path: Path) -> None:
    """The important refusal. Handing off before the seal is spent would read the
    embargoed years for a rule that still owes an out-of-sample verdict."""
    path = tmp_path / "research.sqlite"
    with Registry(path) as reg:
        reg.upsert_strategy(_spec())

    result = _run(path, tmp_path / "books")
    assert result.exit_code != 0
    assert "has no sealed run" in _text(result)
    assert not (tmp_path / "books").exists()


def test_a_refuted_candidate_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite"
    with Registry(path) as reg:
        fingerprint = reg.upsert_strategy(_spec())
        reg.preregister(fingerprint, selection_rule="the only one", seal_digest="c" * 64)
        reg.record_sealed_run(fingerprint, result=_measurement(refuted=True))

    result = _run(path, tmp_path / "books")
    assert result.exit_code != 0
    assert "was refuted by its sealed run" in _text(result)
    assert not (tmp_path / "books").exists()


def test_a_divergent_timeframe_is_refused(db: Path, tmp_path: Path) -> None:
    """``--timeframe`` picks the cache folder; the book is stamped with the
    spec's own timeframe. A divergent pair would stamp "1D" on weights built
    from other bars and still carry the sealed verdict, which was measured on
    the spec's timeframe only. Synthetic bars would load fine, so only this
    refusal stands between the two."""
    out = tmp_path / "books"
    result = _run(db, out, "--timeframe", "4h")
    assert result.exit_code != 0
    assert "declares 1D" in _text(result)
    assert not out.exists()


def test_a_spec_path_is_accepted_and_held_to_the_same_standard(
    db: Path, tmp_path: Path
) -> None:
    """A path is a convenience. Its fingerprint still has to be one the registry
    knows, or there is no sealed run to trace the book back to."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "strategy:\n"
        "  name: handoff_probe\n"
        "  hypothesis: Leaders keep leading over a month.\n"
        "  mode: portfolio\n"
        "  rank_by: roc(20)\n"
        "  hold: 2\n"
        "  rebalance_every: 20\n"
        "  universe: {symbols: [SPY, QQQ, IWM, DIA], timeframe: 1D}\n"
        "  sleeve: {budget: 0.2, idle: benchmark}\n",
        encoding="utf-8",
    )
    out = tmp_path / "books"
    result = runner.invoke(
        app,
        [
            "target-book",
            str(spec_file),
            "--db",
            str(db),
            "--out",
            str(out),
            "--universe",
            "",
            *WINDOW,
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert len(list(out.glob("*.json"))) == 1


def test_the_output_says_what_it_is_not(db: Path, tmp_path: Path) -> None:
    """The command's own output names the boundary. Somebody reading a terminal
    is the most likely person to assume a book that was written was also placed."""
    result = _run(db, tmp_path / "books")
    assert result.exit_code == 0
    lowered = _text(result).lower()
    assert "weights only" in lowered
    assert "kill switch" in lowered


def test_the_output_names_where_the_data_ends(db: Path, tmp_path: Path) -> None:
    """as_of comes from the last bar, not from the requested window, so the
    report shows both rather than letting the request stand in for the data.
    The end date here (2020-01-01) is bar-exclusive, so the data stops the
    session before it."""
    out = tmp_path / "books"
    result = _run(db, out)
    assert result.exit_code == 0, result.stdout
    text = _text(result)
    assert "2015-01-01 -> 2020-01-01 requested" in text
    assert "the data reaches 2019-12-31" in text
    assert "the book will be as of 2019-12-31" in text

    payload = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["as_of"] == "2019-12-31"
