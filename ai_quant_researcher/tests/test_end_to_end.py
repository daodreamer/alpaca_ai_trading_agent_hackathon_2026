"""The whole chain, in one test: research → evaluate → preregister → sealed → book.

Every stage of this project has its own tests. What none of them check is that
the *artefacts* line up — that the fingerprint `evaluate` records is the one
`preregister` declares, that the sealed run spends the seal on that same
fingerprint, and that the target book handed off at the end can be traced back
through all of it. Those are the joints, and joints are where a five-command
pipeline breaks without any single command failing.

So this runs the real CLI commands against a temporary cache and a temporary
registry, and asserts after each one that the artefact it was supposed to leave
behind is there.

Two things are handled carefully rather than mocked:

**Every stage runs under its own ``scope(Seal())``, because in real life every
stage is its own process.** That is not a convenience. Writing the fixture cache
— which spans the embargo — taints whichever seal is ambient at the time, and if
that were the same seal the research stage recorded, ``preregister`` would refuse
the candidate for a tainted ancestry. It did, the first time this test was
written. The seal was right and the test was wrong: a process that has seen the
embargoed years may not also be the process that searches, and building the
cache is exactly such a process.

**The cache crosses the embargo, and the research half still cannot see past
it.** One CSV root holds 2019 through 2026. The research commands ask for
sessions ending 2024-08-30 and get exactly those, which is the arrangement the
real project runs under: the embargo is enforced by what is requested and by
``SealedProvider``, not by keeping two copies of the truth in a test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from aqr.cli import app
from aqr.cli_sealed import app as sealed_app
from aqr.data.bars import Bars
from aqr.data.providers import CsvProvider
from aqr.dsl.schema import StrategySpec, spec_from_dict
from aqr.registry.db import Registry
from aqr.seal import Seal, scope
from aqr.target_book import load_book

runner = CliRunner()

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
FIRST = datetime(2017, 1, 1, tzinfo=UTC)
LAST = datetime(2026, 8, 27, tzinfo=UTC)
SEARCH_END = "2024-08-30"
"""The last session the research half may see. The embargo starts 2024-09-01."""


def _text(result: object) -> str:
    raw = str(getattr(result, "output", ""))
    return " ".join("".join(" " if ord(c) > 0x2500 else c for c in raw).split())


def _sessions() -> np.ndarray:
    """Weekday stamps from ``FIRST`` to ``LAST``. The calendar the chain runs on."""
    stamps: list[int] = []
    day = FIRST
    while day <= LAST:
        if day.weekday() < 5:
            stamps.append(int(day.timestamp()))
        day += timedelta(days=1)
    return np.array(stamps, dtype=np.int64)


def _series(index: int, sessions: int) -> np.ndarray:
    """A seeded random walk with a per-symbol drift.

    Drift ordered by symbol so the cross-section has a real spread to rank on --
    the top-ranked names genuinely do outperform, which is what lets the sealed
    stage reach a determinate verdict instead of a coin flip. Seeded per symbol,
    so the whole chain is reproducible from this file alone.
    """
    rng = np.random.default_rng(1000 + index)
    drift = 0.00012 * (index + 1)
    steps = rng.normal(drift, 0.011, size=sessions)
    return 100.0 * np.exp(np.cumsum(steps))


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One CSV root spanning the embargo, written once for the whole chain.

    Written inside its own ``scope``: these bars run past the embargo, so
    materialising them taints whatever seal is ambient. In the real project this
    is ``aqr-sealed pull``, which is its own process for the same reason.
    """
    root = tmp_path_factory.mktemp("cache")
    event_time = _sessions()

    provider = CsvProvider(str(root))
    with scope(Seal()):
        for i, symbol in enumerate(SYMBOLS):
            close = _series(i, event_time.size)
            provider.write(
                Bars(
                    symbol=symbol,
                    timeframe="1D",
                    event_time=event_time,
                    open=close * 0.999,
                    high=close * 1.008,
                    low=close * 0.992,
                    close=close,
                    volume=np.full(event_time.size, 5e6),
                )
            )
    return root


def _spec() -> StrategySpec:
    return spec_from_dict(
        {
            "strategy": {
                "name": "e2e_cross_sectional_momentum",
                "hypothesis": "Names leading the cross-section over a month keep leading.",
                "mode": "portfolio",
                "rank_by": "roc(20)",
                "hold": 3,
                "rebalance_every": 10,
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
                "sleeve": {"budget": 0.20, "idle": "benchmark"},
            }
        }
    )


@pytest.fixture(scope="module")
def spec_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from aqr.dsl.loader import dumps

    path = tmp_path_factory.mktemp("specs") / "e2e.yaml"
    path.write_text(dumps(_spec()), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("runs") / "e2e.sqlite"


@pytest.fixture(scope="module")
def books(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("books")


def _search_args(cache: Path) -> list[str]:
    return [
        "--source", "csv",
        "--csv-root", str(cache),
        "--start", "2017-01-01",
        "--end", SEARCH_END,
        "--universe", "",
    ]


# --------------------------------------------------------------------------
# The chain, in order. Each test depends on the one before it, which is why they
# are numbered and why the fixtures are module-scoped: pytest runs them top to
# bottom within a file, and the registry each one writes to is the registry the
# next one reads.


def test_1_research_records_what_it_tried(cache: Path, db: Path) -> None:
    """The offline loop, so the chain needs no API key and no network.

    What is asserted is the *record*, not a verdict: the multiple-comparisons
    denominator is only honest if every attempt is written down, so an
    experiment count that did not move would be the failure here even if every
    hypothesis was rejected.
    """
    with scope(Seal()):
        result = runner.invoke(
            app,
            [
                "research",
                "--provider", "offline",
                "--iterations", "2",
                "--symbols", ",".join(SYMBOLS),
                "--db", str(db),
                *_search_args(cache),
            ],
        )
    assert result.exit_code == 0, _text(result)

    with Registry(db) as reg:
        assert reg.total_backtests() > 0
        assert reg.distinct_hypotheses() > 0


def test_2_evaluate_registers_the_candidate(cache: Path, db: Path, spec_file: Path) -> None:
    """The strategy the rest of the chain follows enters the registry here."""
    with scope(Seal()):
        result = runner.invoke(
            app,
            ["evaluate", str(spec_file), "--db", str(db), *_search_args(cache)],
        )
    # ACCEPT/PAPER exit 0, anything else exits 1. Either is a completed
    # evaluation; the chain is about the artefacts, not about the verdict.
    assert result.exit_code in (0, 1), _text(result)

    fingerprint = _spec().fingerprint()
    with Registry(db) as reg:
        record = reg.get_strategy(fingerprint)
        assert record is not None
        assert record.name == "e2e_cross_sectional_momentum"
        rows = reg.experiments(fingerprint=fingerprint)
        assert rows, "evaluate left no experiment behind"
        # 5.2: the schedule that decided the cost gate travels with the verdict.
        assert json.loads(rows[0]["costs"])["min_commission"] == 1.0


def test_3_preregister_declares_it_before_anything_is_read(db: Path) -> None:
    fingerprint = _spec().fingerprint()
    with scope(Seal()):
        result = runner.invoke(
            app,
            [
                "preregister", fingerprint,
                "--rule", "the only candidate in the end-to-end chain",
                "--db", str(db),
            ],
        )
    assert result.exit_code == 0, _text(result)

    with Registry(db) as reg:
        declaration = reg.preregistration(fingerprint)
        assert declaration is not None
        assert declaration.selection_rule.startswith("the only candidate")
        assert declaration.seal_digest
        # Declared, and not yet spent. The ordering is the protocol.
        assert reg.sealed_run(fingerprint) is None


def test_4_the_sealed_run_spends_the_seal_once(cache: Path, db: Path) -> None:
    """Under ``scope`` so promoting the ambient seal cannot leak into the suite."""
    fingerprint = _spec().fingerprint()
    args = [
        "run", fingerprint,
        "--db", str(db),
        "--csv-root", str(cache),
        "--universe", "",
        "--end", "2026-08-27",
    ]
    with scope(Seal()):
        first = runner.invoke(sealed_app, args)
    # 0 = not refuted, 1 = refuted. Both are completed runs.
    assert first.exit_code in (0, 1), _text(first)

    with Registry(db) as reg:
        spent = reg.sealed_run(fingerprint)
        assert spent is not None
        measurement = spent["result"]["measurement"]
        assert measurement["observations"] > 60
        assert measurement["looks"] == 1
        assert reg.sealed_look(fingerprint) == 1
        # The certificate proves the sealed process was the one that read them.
        assert spent["result"]["seal"]["phase"] == "sealed"
        assert spent["result"]["declaration"]["selection_rule"]

    # And there is no second one.
    with scope(Seal()):
        again = runner.invoke(sealed_app, args)
    assert again.exit_code != 0
    assert "no second one" in _text(again) or "already" in _text(again)


def test_5_the_target_book_is_written_and_traceable(
    cache: Path, db: Path, books: Path
) -> None:
    """The end of the chain: a file, and nothing placed anywhere.

    Also under ``scope`` — a book is built from bars through the present, so the
    process that builds one is tainted by construction and the certificate says
    so. That is the correct record, and it must not leak into the suite either.
    """
    fingerprint = _spec().fingerprint()
    with Registry(db) as reg:
        if (reg.sealed_run(fingerprint) or {}).get("result", {}).get(
            "measurement", {}
        ).get("refuted"):
            pytest.skip("the sealed run refuted the candidate; there is nothing to hand off")

    with scope(Seal()):
        result = runner.invoke(
            app,
            [
                "target-book", fingerprint,
                "--db", str(db),
                "--source", "csv",
                "--csv-root", str(cache),
                "--universe", "",
                "--out", str(books),
                "--start", "2022-01-01",
                "--end", "2026-08-27",
            ],
        )
    assert result.exit_code == 0, _text(result)

    written = list(books.glob("*.json"))
    assert len(written) == 1
    book = load_book(written[0])

    # Every link in the chain is reachable from the artefact alone.
    assert book["spec_fingerprint"] == fingerprint
    # Derived rather than hard-coded, and it pins a convention worth pinning: the
    # data window is half-open, so ``--end 2026-08-27`` stops at the session
    # *before* it. ``as_of`` names the last session the book actually saw, never
    # the date that was asked for -- an executor reconciling against a session
    # that was never in the book is the failure this field exists to prevent.
    requested_end = int(datetime(2026, 8, 27, tzinfo=UTC).timestamp())
    sessions = _sessions()
    last = int(sessions[sessions < requested_end][-1])
    assert book["as_of_event_time"] == last
    assert book["as_of"] == datetime.fromtimestamp(last, tz=UTC).date().isoformat()
    assert book["weights"]
    provenance = book["provenance"]
    assert provenance["sealed_run_at"]
    assert provenance["preregistration"]["selection_rule"].startswith("the only candidate")
    assert provenance["distinct_hypotheses"] > 0
    assert provenance["sealed_look"] == 1
    assert provenance["sealed_looks_total"] == 1
    # Reading the present is what a handoff is, and the seal records it.
    assert book["seal"]["tainted"] is True

    with Registry(db) as reg:
        rows = reg.target_books(fingerprint)
        assert len(rows) == 1
        assert Path(rows[0]["path"]) == written[0]


def test_6_the_suite_seal_survived_the_chain() -> None:
    """The stages that read the embargoed years ran inside ``scope``.

    If either had leaked, the ambient seal would be tainted for every test that
    follows, and the boundary the whole project rests on would have been broken
    by its own test suite.
    """
    from aqr.seal import Phase, current

    assert current().phase is Phase.RESEARCH
