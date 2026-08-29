"""The handoff: a target book, and the properties that make it worth executing.

The book is the interface between this project and whatever places orders, so
the tests here are about agreement rather than about performance. Three things
have to hold or the artefact is a liability:

* **It is what the backtest held.** The weights come off ``run_strategy``, the
  same entry point everything else in the project goes through. If a book for a
  historical session ever disagreed with the weights the backtest held on that
  session, the handoff would be describing a strategy nobody validated.
* **It is weights, and only weights.** No shares, no notionals, no account. The
  schema refuses those fields by name, because the first one to appear would be
  this project sizing a position.
* **It is reproducible and self-describing.** Same inputs, same bytes, same
  digest -- and enough provenance inside the file that a reader who has never
  heard of ``aqr`` can find out where it came from.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.portfolio import run_portfolio
from aqr.data.bars import Bars
from aqr.dsl.schema import StrategySpec, spec_from_dict
from aqr.target_book import (
    BOOK_SCHEMA_VERSION,
    TargetBook,
    book_digest,
    build_target_book,
    load_book,
    validate_book,
    write_book,
)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
N = 260
STEP = 86_400
T0 = 1_500_000_000

GENERATED_AT = datetime(2026, 8, 29, 13, 0, tzinfo=UTC)
CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def _bars(symbol: str, drift: float, n: int = N) -> Bars:
    """A deterministic ramp, so the cross-section has something unambiguous to sort."""
    t = np.arange(T0, T0 + n * STEP, STEP, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1e6),
    )


def _universe(n: int = N) -> dict[str, Bars]:
    return {s: _bars(s, 0.0002 * (i + 1), n) for i, s in enumerate(SYMBOLS)}


def _spec(**over: object) -> StrategySpec:
    body: dict[str, object] = {
        "name": "xs_momentum",
        "mode": "portfolio",
        "rank_by": "roc(20)",
        "hold": 2,
        "rebalance_every": 20,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "sleeve": {"budget": 0.20, "idle": "benchmark"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


def _book(data: dict[str, Bars] | None = None, **over: object) -> TargetBook:
    return build_target_book(
        _spec(**over),
        data if data is not None else _universe(),
        generated_at=GENERATED_AT,
        dataset_version="synthetic:1D:2017-07-14:2018-04-01",
        universe="test",
        provenance={"sealed_run_at": "2026-08-29T00:00:00+00:00"},
        seal={"digest": "0" * 64, "tainted": True, "phase": "research"},
        config=CONFIG,
    )


# --------------------------------------------------------------------------
# 4.1 -- the book comes off the validated code path


def test_the_book_matches_what_the_backtest_held() -> None:
    """The acceptance test for the whole phase.

    A book built from bars through session T must equal the weights the backtest
    held on T when it was run over the longer history. If these ever diverge, the
    handoff describes a different strategy than the one that was validated, and
    every number in the sealed run stops applying to what is actually traded.
    """
    full = _universe(N)
    backtest = run_portfolio(_spec(), full, CONFIG)

    # Truncate to a historical session and hand off *as of* that session.
    cut = 200
    truncated = {
        s: Bars(
            symbol=b.symbol,
            timeframe=b.timeframe,
            event_time=b.event_time[:cut],
            open=b.open[:cut],
            high=b.high[:cut],
            low=b.low[:cut],
            close=b.close[:cut],
            volume=b.volume[:cut],
        )
        for s, b in full.items()
    }
    book = _book(truncated)

    step = cut - 1
    assert book.as_of_event_time == int(backtest.timeline[step])
    assert book.weights == pytest.approx(backtest.weights_at(step))
    assert book.core_weights == pytest.approx(backtest.core_weights_at(step))
    assert book.sleeve_weights == pytest.approx(backtest.sleeve_weights_at(step))


def test_the_book_is_as_of_the_last_session_not_today() -> None:
    """``as_of`` names the session the weights were in force on.

    Stamping it with the wall clock would make a book produced on a Sunday claim
    to describe Sunday, and an executor would reconcile against a session that
    never happened.
    """
    book = _book()
    last = datetime.fromtimestamp(T0 + (N - 1) * STEP, tz=UTC)
    assert book.as_of == last.date().isoformat()
    assert book.generated_at == GENERATED_AT.isoformat()
    assert book.as_of != GENERATED_AT.date().isoformat()


def test_a_signal_spec_has_no_target_book() -> None:
    """A trigger strategy's positions are a consequence of fills, not a set of
    weights. Inventing weights for one would hand off a portfolio interpretation
    of a strategy nobody validated as a portfolio."""
    signal = spec_from_dict(
        {
            "strategy": {
                "name": "pullback",
                "entry": "close > ema(20)",
                "universe": {"symbols": ["AAA"], "timeframe": "1D"},
            }
        }
    )
    with pytest.raises(ValueError, match="only a portfolio spec"):
        build_target_book(
            signal,
            {"AAA": _bars("AAA", 0.001)},
            generated_at=GENERATED_AT,
            dataset_version="synthetic",
            universe="test",
            provenance={},
            seal={},
        )


def test_a_run_that_never_filled_produces_no_book() -> None:
    """An empty book reads as 'hold nothing', which is a position. A run too
    short to clear its warm-up has not decided to hold nothing -- it has not
    decided anything, and the two must not be confused at the interface."""
    with pytest.raises(ValueError, match="never filled"):
        _book(_universe(21))


def test_the_book_is_reproducible() -> None:
    """Same inputs, same bytes. A digest that moved between two identical runs
    could not be used to detect a file that was edited after the handoff."""
    first, second = _book(), _book()
    assert first.as_dict() == second.as_dict()
    assert book_digest(first.as_dict()) == book_digest(second.as_dict())


def test_the_core_holds_exactly_the_top_ranked_names() -> None:
    """The universe is monotone in symbol order, so the expected book is knowable
    without re-implementing the engine."""
    book = _book()
    assert sorted(book.core_weights) == ["EEE", "FFF"]
    assert sum(book.core_weights.values()) == pytest.approx(0.8)
    assert book.gross == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 4.2 -- the artefact


def test_the_artefact_carries_everything_needed_to_trace_it() -> None:
    payload = _book().as_dict()
    for field in (
        "spec_fingerprint",
        "spec_name",
        "as_of",
        "dataset_version",
        "universe",
        "seal",
        "provenance",
        "weights",
    ):
        assert field in payload, field
    assert payload["schema_version"] == BOOK_SCHEMA_VERSION
    assert payload["spec_fingerprint"] == _spec().fingerprint()
    assert payload["seal"]["tainted"] is True


def test_the_artefact_names_what_the_consumer_must_supply() -> None:
    """The boundary travels with the file. A consumer that reads only the weights
    and none of the design documents still learns that the risk gate, the
    reconciliation and the sizing are its own problem."""
    supplied = " ".join(_book().as_dict()["consumer_must_supply"]).lower()
    for owed in ("equity", "reconciliation", "risk gate", "kill switch", "fill journal"):
        assert owed in supplied, owed


def test_the_schema_refuses_a_book_that_sizes_a_position() -> None:
    """Shares, notionals and account equity are the consumer's to compute.
    A field naming one is this project sizing a position."""
    for field in ("shares", "quantity", "notional", "equity", "orders"):
        payload = _book().as_dict()
        payload[field] = {"AAA": 100}
        with pytest.raises(ValueError, match="weights only"):
            validate_book(payload)


def test_the_schema_refuses_a_missing_or_mistyped_field() -> None:
    payload = _book().as_dict()
    del payload["as_of"]
    payload["weights"] = "AAA:1.0"
    with pytest.raises(ValueError) as caught:
        validate_book(payload)
    assert "missing 'as_of'" in str(caught.value)
    assert "'weights' is str" in str(caught.value)


def test_the_schema_refuses_a_version_it_cannot_interpret() -> None:
    """A consumer pins the version. Reading a book from the future under the
    assumption that ``weights`` still means what it used to is how a silent
    execution bug happens."""
    payload = _book().as_dict()
    payload["schema_version"] = BOOK_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        validate_book(payload)


def test_the_schema_refuses_a_position_count_that_disagrees_with_the_weights() -> None:
    payload = _book().as_dict()
    payload["positions"] += 1
    with pytest.raises(ValueError, match="disagrees"):
        validate_book(payload)


def test_a_written_book_reads_back_identically(tmp_path: Path) -> None:
    book = _book()
    path = tmp_path / "book.json"
    digest = write_book(book, path)

    assert digest == book_digest(book.as_dict())
    assert load_book(path) == book.as_dict()
    # Readable by something that has never heard of aqr.
    assert json.loads(path.read_text(encoding="utf-8"))["weights"] == book.as_dict()["weights"]


def test_an_edited_book_no_longer_matches_its_digest(tmp_path: Path) -> None:
    """What the recorded digest is for: a file that changed after the handoff."""
    book = _book()
    path = tmp_path / "book.json"
    digest = write_book(book, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"]["FFF"] = 0.9
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    assert book_digest(json.loads(path.read_text(encoding="utf-8"))) != digest
