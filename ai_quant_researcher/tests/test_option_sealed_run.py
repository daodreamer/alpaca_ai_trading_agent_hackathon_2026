"""The option sealed run — specs/10-options-research.md D3, D8, and PLAN O5.

Three things are under test, and each of them is a way the one-shot protocol
could quietly stop meaning anything.

**The underlying must be pulled raw.** specs/10 D0: option strikes are set in
raw terms and do not move for an ordinary dividend, so a dividend-adjusted close
compared against a strike reports a moneyness the trade never had — SPY's real
close on 2019-11-22 was 311.02 and the adjusted series says 282.10. Nothing
raises; the backtest completes and reports plausible numbers. So the sealed pull
has to be able to say ``--adjustment raw``, and the choice has to reach the
provider rather than being validated and dropped.

**The settlement boundary moves in the sealed phase.** D3 refuses an entry whose
expiry crosses the embargo. In the research phase that is the point; in the
sealed phase the reserved window *is* what is being measured, so a run left at
the default would refuse every entry in the years it came to read, report zero
trades and spend the seal on nothing.

**The sizing may not drift.** D8a measured the same rule producing 21
independent cycles at 1% of equity and 57 at 2%, with nothing about the rule
changed. A sealed run at a different size is a different experiment wearing the
first one's pre-registration.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from aqr.cli_sealed import app
from aqr.data.bars import Bars
from aqr.dsl.schema import StrategySpec, Universe
from aqr.options.chain import ChainIndex
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket
from aqr.options.sealed import measure_sealed_option_window
from aqr.options.spec import Cadence, OptionSizing, OptionSpec, StructureSpec
from aqr.registry.db import Registry
from aqr.seal import Seal, scope
from tests.test_option_engine import chain_row, credit_spread_spec, make_underlying, trading_days

runner = CliRunner()

BOUNDARY = datetime(2023, 7, 3, tzinfo=UTC)


def _text(result: object) -> str:
    raw = str(getattr(result, "output", ""))
    stripped = "".join(" " if ord(ch) > 0x2500 else ch for ch in raw)
    return " ".join(stripped.split())


def _spec(**overrides: Any) -> OptionSpec:
    fields: dict[str, Any] = {
        "name": "sealed_probe",
        "hypothesis": "Index puts carry a variance risk premium.",
        "structure": StructureSpec(type="put_credit_spread", width_delta=0.06),
        "cadence": Cadence(min_sessions_between_entries=2),
        "sizing": OptionSizing(risk_per_trade=0.02, max_concurrent=3),
    }
    fields.update(overrides)
    return credit_spread_spec(**fields)


def _market() -> OptionMarket:
    """Sessions on both sides of the boundary, so the clip has something to do."""
    days = trading_days(500, start=date(2022, 1, 3))
    chosen = [d for i, d in enumerate(days) if i >= 5 and i % 5 == 0]
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


def _config() -> OptionBacktestConfig:
    # The sealed phase's own boundary: the chain's end, not the research
    # embargo. Left at the default this fixture would refuse every entry.
    return OptionBacktestConfig(initial_equity=1_000_000.0, settle_before=date(2030, 1, 1))


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #


def test_only_sessions_from_the_boundary_onward_are_scored() -> None:
    """The bars start before the window and the measurement does not. Truncating
    the market instead would silently change the rule -- and here it would also
    drop every structure opened before the boundary that settles inside it."""
    measurement = measure_sealed_option_window(
        _spec(), _market(), since=BOUNDARY, config=_config()
    )
    assert measurement.first_session is not None
    assert measurement.first_session >= BOUNDARY
    assert measurement.backtest_sessions > measurement.observations


def test_the_window_can_refute_and_can_never_confirm() -> None:
    """A property rather than a comment: anything reading this object has to
    handle the answer being no. 25 independent 28-DTE cycles is not enough to
    establish an edge at any plausible effect size (D8)."""
    measurement = measure_sealed_option_window(
        _spec(), _market(), since=BOUNDARY, config=_config()
    )
    assert measurement.can_confirm is False
    assert "cannot confirm" in measurement.summary()


def test_the_multiplicity_bar_rises_with_the_look_count() -> None:
    one = measure_sealed_option_window(_spec(), _market(), since=BOUNDARY, config=_config())
    seven = measure_sealed_option_window(
        _spec(), _market(), since=BOUNDARY, config=_config(), looks=7
    )
    assert seven.significance_bar > one.significance_bar


def test_the_benchmark_is_the_underlying_over_the_same_window() -> None:
    """A short put spread is a levered, capped long position, so most of what it
    earns is the drift it was exposed to. Comparing a one-year strategy return
    against a five-year index return would attribute the difference to the
    rule."""
    measurement = measure_sealed_option_window(
        _spec(), _market(), since=BOUNDARY, config=_config()
    )
    assert measurement.benchmark_return == pytest.approx(0.0, abs=1e-9)
    # A flat underlying: the fixture's closes never move, so the benchmark earns
    # nothing and whatever the rule earned is the rule's.
    assert measurement.observations > 0


def test_a_window_with_no_sessions_says_so_rather_than_reporting_zeroes() -> None:
    """A plausible-looking figure nobody can check is worse than an absent one."""
    measurement = measure_sealed_option_window(
        _spec(), _market(), since=datetime(2030, 1, 1, tzinfo=UTC), config=_config()
    )
    assert measurement.observations == 0
    assert measurement.residual is None
    assert "no sessions" in measurement.note


# --------------------------------------------------------------------------- #
# ``aqr-sealed option-run``: the refusals, before anything is read
# --------------------------------------------------------------------------- #


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "research.sqlite"


def test_an_undeclared_candidate_is_refused(db: Path) -> None:
    spec = _spec()
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)

    with scope(Seal()):
        result = runner.invoke(app, ["option-run", spec.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "not pre-registered" in _text(result)


def test_an_equity_candidate_is_sent_to_the_other_command(db: Path) -> None:
    """The two measure different windows and are counted as different
    denominators; running one through the other's command would spend the wrong
    seal and file the look under the wrong family."""
    equity = StrategySpec(
        name="an_equity_rule",
        entry="close > ema(20)",
        universe=Universe(symbols=("SPY",), timeframe="1D"),
    )
    with Registry(db) as reg:
        reg.upsert_strategy(equity)
        reg.preregister(equity.fingerprint(), selection_rule="the best", seal_digest="d")

    with scope(Seal()):
        result = runner.invoke(app, ["option-run", equity.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "cli_sealed run" in _text(result)


def test_a_spent_seal_is_refused(db: Path) -> None:
    spec = _spec()
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)
        reg.preregister(spec.fingerprint(), selection_rule="the best", seal_digest="d")
        reg.record_sealed_run(spec.fingerprint(), result={"measurement": {}})

    with scope(Seal()):
        result = runner.invoke(app, ["option-run", spec.fingerprint(), "--db", str(db)])
    assert result.exit_code != 0
    assert "no second one" in _text(result)


def test_a_sizing_that_does_not_match_the_declaration_is_refused(db: Path) -> None:
    spec = _spec()
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)
        reg.preregister(spec.fingerprint(), selection_rule="the best", seal_digest="d")

    with scope(Seal()):
        result = runner.invoke(
            app,
            [
                "option-run",
                spec.fingerprint(),
                "--db",
                str(db),
                "--risk-per-trade",
                "0.05",
            ],
        )
    assert result.exit_code != 0
    text = _text(result)
    assert "0.02" in text and "different experiment" in text


def test_every_refusal_happens_before_the_process_is_promoted(db: Path) -> None:
    """A rejected candidate must cost no seal at all, which means every check
    runs in a process that has still read nothing."""
    spec = _spec()
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)

    with scope(Seal()) as seal:
        runner.invoke(app, ["option-run", spec.fingerprint(), "--db", str(db)])
        assert seal.phase.name == "RESEARCH"
        assert not seal.loads


# --------------------------------------------------------------------------- #
# ``aqr-sealed pull``: the adjustment reaches the provider
# --------------------------------------------------------------------------- #


class RecordingAlpaca:
    """The provider surface ``pull`` uses, recording what it was constructed with."""

    seen: list[dict[str, Any]] = []

    def __init__(self, *, feed: str = "sip", adjustment: str = "all") -> None:
        self.feed = feed
        self.adjustment = adjustment
        RecordingAlpaca.seen.append({"feed": feed, "adjustment": adjustment})
        self.symbols: list[str] = []

    def dataset_version(self, timeframe: str) -> str:
        return f"alpaca:{self.feed}:{self.adjustment}:test:{timeframe}"

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        self.symbols.append(symbol)
        days = trading_days(300, start=date(2019, 1, 2))
        bars = make_underlying({}, days)
        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=bars.event_time,
            open=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=bars.volume,
        )


def test_an_unknown_adjustment_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    with scope(Seal()) as seal:
        result = runner.invoke(
            app,
            ["pull", "--symbols", "SPY", "--adjustment", "nominal", "--csv-root", str(tmp_path)],
        )
        assert seal.phase.name == "RESEARCH"
    assert result.exit_code != 0
    assert "unknown adjustment" in _text(result)


def test_the_adjustment_reaches_the_provider(tmp_path: Path, monkeypatch: Any) -> None:
    """O5.1's whole point. A flag that is validated and then dropped would leave
    the sealed underlying dividend-adjusted, and nothing downstream would raise:
    the parity check in ``tests/test_option_cache_claims.py`` is the only thing
    that can see it, and it runs against a cache this command has already
    written."""
    import aqr.cli_sealed as sealed_cli

    RecordingAlpaca.seen.clear()
    monkeypatch.setattr(sealed_cli, "AlpacaProvider", RecordingAlpaca)

    with scope(Seal()):
        result = runner.invoke(
            app,
            [
                "pull",
                "--symbols",
                "SPY",
                "--adjustment",
                "raw",
                "--csv-root",
                str(tmp_path),
                "--start",
                "2019-01-01",
                "--end",
                "2020-01-01",
            ],
        )

    assert result.exit_code == 0, _text(result)
    assert RecordingAlpaca.seen == [{"feed": "sip", "adjustment": "raw"}]
    # And the explicit symbol list won outright rather than being merged with the
    # 600-name universe file: quietly adding those to this root would make it a
    # second equity cache pulled with the wrong adjustment for equity research.
    assert (tmp_path / "1D" / "SPY.csv").exists()
    assert len(list((tmp_path / "1D").glob("*.csv"))) == 1


def test_preregistering_an_option_rule_points_at_the_option_sealed_run(
    db: Path,
) -> None:
    """The two sealed runs read different windows, charge different costs and are
    counted as different denominators. Naming the equity command here would send
    somebody to one that refuses -- and before ``option-run`` existed it would
    have read as the right one."""
    from aqr.cli import app as research_app

    spec = _spec()
    with Registry(db) as reg:
        reg.upsert_option_strategy(spec)

    with scope(Seal()):
        result = runner.invoke(
            research_app,
            ["preregister", spec.fingerprint(), "--rule", "the only one", "--db", str(db)],
        )

    assert result.exit_code == 0, _text(result)
    text = _text(result)
    assert "cli_sealed option-run" in text
    assert "family option" in text
