"""The sealed entry point. A separate process, on purpose.

    aqr-sealed audit                     what the sealed cache holds
    aqr-sealed pull                      build the sealed cache
    aqr-sealed run FINGERPRINT           spend the seal, once
    aqr-sealed option-run FINGERPRINT    the same, on the sealed option chain

This is the only module in the project that may read the embargoed years, and
almost everything about it is a restriction:

**A separate process, not a flag.** ``aqr.cli`` carries ``research``. A
``--sealed`` switch there would put "search" and "read the answer" in one
process, and the phase separation would be a convention again. So this file
exists, it does not import ``aqr.cli``, and
``test_the_sealed_entry_point_does_not_import_the_research_cli`` fails the build
if that changes.

**No agent layer.** A sealed run executes one pre-registered spec. There is
nothing to propose, and a proposer here would be a model choosing hypotheses
with the answer in front of it.

**One spec, from the registry, pre-registered.** Not from a file on disk. A YAML
path is editable between the declaration and the run; a fingerprint is not.

**One shot, per candidate.** :meth:`Registry.record_sealed_run` refuses a second
run *on the same fingerprint* rather than overwriting the first, and the refusal
is the feature. A genuinely new hypothesis is entitled to its own sealed run --
otherwise a research loop would have to stop after its first strategy. What that
costs is multiplicity, so every run records which look it was and the alpha's
significance bar rises with the count. Counting the looks is the defence; nothing
here forbids the seventh.

What the seal proves, said once so no report has to be trusted to repeat it: the
embargoed data was not read during the search. It does not prove the embargoed
*period* did not inform a decision -- the researcher lived through it and every
model in ``aqr providers`` has a training cutoff after it. The certificate
records that exposure rather than denying it.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from aqr.backtest.engine import BacktestConfig
from aqr.data.alpaca import AlpacaProvider
from aqr.data.bars import Bars, ensure_utc
from aqr.data.embargo import SealedProvider, SealToken, audit_cache_root
from aqr.data.providers import CsvProvider, Provider
from aqr.data.quality import inspect as inspect_bars
from aqr.data.universes import (
    PointInTimeUniverse,
    is_point_in_time,
    load_point_in_time,
)
from aqr.dsl.loader import loads as loads_spec
from aqr.option_data import sealed_option_market
from aqr.options.costs import IBKR_OPTIONS, OptionCostModel
from aqr.options.engine import OptionBacktestConfig
from aqr.options.sealed import measure_sealed_option_window
from aqr.options.spec import loads_option_spec
from aqr.registry.db import OPTION, PreregistrationError, Registry
from aqr.seal import EMBARGO_START, current, enter_sealed_phase
from aqr.validation.sealed import measure_sealed_window, multiplicity_bar

app = typer.Typer(
    add_completion=False,
    help=(
        "The sealed entry point. Reads the embargoed years for one pre-registered "
        "candidate, once. It cannot search."
    ),
    no_args_is_help=True,
)
console = Console()

SEALED_SP500_ROOT = "data-sp500-sealed"
"""Where the sealed S&P 500 cache lives. The mirror of ``data-sp500``."""

SEALED_OPTIONS_ROOT = "data-options-sealed"
SEALED_UNDERLYING_ROOT = "data-options-underlying-sealed"
"""The option pair, and the second one is the easy mistake.

The sealed option chain has a mirror on disk already. Its *underlying* does not
come from ``data-sp500-sealed``: specs/10 D0 requires the bars a strike is
compared against to be pulled raw, and the equity cache is dividend-adjusted.
SPY's real close on 2019-11-22 was 311.02 and the adjusted series says 282.10 --
a ten percent error that no arithmetic notices, because nothing is wrong with the
arithmetic. Build this root with::

    python -m aqr.cli_sealed pull --symbols SPY --adjustment raw \\
        --csv-root data-options-underlying-sealed --timeframe 1D
"""

# Enough history before the embargo that every feature is warm on the first
# sealed session. ``rs_rank(126)`` is the longest lookback the DSL offers and
# 1200 calendar days clears it with room for holidays and thin names.
WARMUP_DAYS = 1200

# Alpaca serves the consolidated tape on a delay to accounts without a
# real-time SIP entitlement, and a request whose end is inside that delay is
# refused with a 403 for every symbol -- not a partial response. So "today"
# means yesterday here. Named rather than inlined because the sealed run and
# the pull have to agree, or the run would score sessions the cache lacks.
FEED_DELAY_DAYS = 1


def _latest_available() -> datetime:
    return datetime.now(UTC) - timedelta(days=FEED_DELAY_DAYS)


def _sealed(inner: Provider, label: str) -> SealedProvider:
    """The only place a token is minted, and it is minted for one provider.

    ``test_no_module_outside_the_embargo_layer_constructs_a_seal_token`` allows
    this file to name ``SealToken`` and no other outside ``embargo.py``.
    """
    return SealedProvider(inner, token=SealToken(), label=label)


def _promote() -> None:
    """Enter the sealed phase before anything is read, and say so.

    Refused if this process has already touched data, which is what stops a
    search from promoting itself into a clean certificate.
    """
    enter_sealed_phase()
    console.print(
        f"[cyan]phase: {current().phase.value}[/cyan]  "
        f"run {current().run_id[:12]}  embargo {EMBARGO_START.date().isoformat()}"
    )


def _parse_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #


@app.command()
def audit(
    root: str = typer.Option(SEALED_SP500_ROOT, help="Cache root to audit."),
) -> None:
    """What a cache root holds, read from the files rather than from a promise.

    Deliberately available here as well as in ``aqr seal-check``: after building
    the sealed cache, the question "did the embargoed rows actually arrive" is
    asked in this process, and answering it should not need the other one.
    """
    report = audit_cache_root(Path(root))
    table = Table(box=None, pad_edge=False)
    for column in ("root", "files", "latest bar", "past embargo"):
        table.add_column(column)
    table.add_row(
        root,
        str(report.files),
        report.latest.date().isoformat() if report.latest else "-",
        str(len(report.offenders)) if report.offenders else "none",
    )
    console.print(table)
    console.print()
    console.print(
        f"the sealed root is [i]expected[/i] to hold bars past "
        f"{EMBARGO_START.date().isoformat()}. A count of 'none' here means the "
        "sealed cache was never built, not that it is clean."
    )


@app.command()
def pull(
    universe_file: str = typer.Option(
        "data-universes/sp500_pit.json",
        help="Point-in-time universe. Every ticker ever a member is pulled, which "
        "is the point: the names that left are exactly the ones whose absence "
        "creates survivorship bias.",
    ),
    symbols: str = typer.Option(
        "",
        help="Comma-separated symbols to pull instead of the universe file. For "
        "the option underlying, which is one name and is not an index member "
        "list: `--symbols SPY`.",
    ),
    adjustment: str = typer.Option(
        "all",
        help="Alpaca price adjustment: all, split, dividend or raw. 'all' is right "
        "for equity research and WRONG for the option underlying -- strikes are "
        "set in raw terms and do not move for a dividend, so an adjusted close "
        "compared against a strike reports a moneyness the trade never had "
        "(specs/10 D0). Pull data-options-underlying-sealed with 'raw'.",
    ),
    start: str = typer.Option("2016-01-01", help="First bar to request."),
    end: str = typer.Option(
        "", help="Last bar. Default: yesterday -- the tape is served on a delay."
    ),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(SEALED_SP500_ROOT, help="Where to write."),
    feed: str = typer.Option("sip", help="Alpaca feed. IEX is too sparse to research on."),
    force: bool = typer.Option(False, help="Re-fetch symbols already cached."),
    keep_suspect: bool = typer.Option(
        False, help="Cache series that failed the quality check anyway."
    ),
) -> None:
    """Build the sealed cache: the same tickers, full history, embargo included.

    Separate from ``aqr pull`` because ``aqr pull`` wraps every provider in
    ``ResearchProvider`` and clamps the request at the embargo. That clamp is
    correct there and is exactly what has to be absent here, and a flag that
    turned it off would put both behaviours in the process that also runs
    ``research``.
    """
    if adjustment not in ("all", "split", "dividend", "raw"):
        raise typer.BadParameter(
            f"unknown adjustment {adjustment!r}; use all, split, dividend or raw"
        )
    _promote()
    window = (_parse_date(start), _parse_date(end) if end else _latest_available())
    # An explicit list wins outright rather than being merged with the universe
    # file: the option underlying is one name pulled into its own root, and
    # quietly adding 600 tickers to that root would make it a second equity
    # cache pulled with the wrong adjustment for equity research.
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    tickers = wanted or _tickers(Path(universe_file))
    console.print(
        f"{len(tickers)} tickers, {window[0].date()} -> {window[1].date()}, "
        f"feed {feed}, adjustment {adjustment} -> {csv_root}/{timeframe}"
    )

    provider = _sealed(AlpacaProvider(feed=feed, adjustment=adjustment), "alpaca")
    cache = CsvProvider(csv_root)
    fetched, skipped, failed, suspect = 0, 0, [], 0
    for symbol in tickers:
        target = cache.path_for(symbol, timeframe)
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            bars = provider.load(symbol, *window, timeframe)
        except Exception as exc:
            # One delisted ticker must not abandon the other six hundred.
            failed.append(f"{symbol} ({type(exc).__name__}: {exc})")
            continue
        report = inspect_bars(bars, requested_start=window[0])
        if not report.ok:
            suspect += 1
            console.print(f"[yellow]{report}[/yellow]")
            if not keep_suspect:
                continue
        cache.write(bars)
        fetched += 1

    console.print()
    console.print(
        f"{fetched} fetched, {skipped} already cached, {len(failed)} failed, "
        f"{suspect} failed the quality check"
    )
    for line in failed[:40]:
        console.print(f"[red]{line}[/red]")
    if len(failed) > 40:
        console.print(f"[red]... and {len(failed) - 40} more[/red]")
    console.print(f"seal digest {current().digest[:16]}, {len(current().loads)} loads")


@app.command()
def backfill(
    start: str = typer.Option(
        "",
        help=(
            "First bar to copy from the research cache. "
            f"Default: {WARMUP_DAYS} days before the embargo."
        ),
    ),
    timeframe: str = typer.Option("1D"),
    sealed_root: str = typer.Option(SEALED_SP500_ROOT, help="Sealed cache to extend."),
    research_root: str = typer.Option(
        "data-sp500", help="Research cache to copy warm-up bars from."
    ),
    dry_run: bool = typer.Option(False, help="Show what would be copied without writing."),
) -> None:
    """Copy pre-embargo bars from the research cache into the sealed cache.

    The sealed cache only needs embargoed years for the seal to be meaningful,
    but long-lookback features need history before the embargo to be warm on the
    first sealed session. Re-pulling that history from the vendor is slow; this
    command reuses the research cache, which already holds the same bars up to
    the embargo boundary.
    """
    first = _parse_date(start) if start else EMBARGO_START - timedelta(days=WARMUP_DAYS)
    first_ts = int(first.timestamp())

    sealed_dir = Path(sealed_root) / timeframe
    research_dir = Path(research_root) / timeframe
    if not sealed_dir.exists():
        raise typer.BadParameter(f"no sealed cache at {sealed_dir}")
    if not research_dir.exists():
        raise typer.BadParameter(f"no research cache at {research_dir}")

    backfilled = 0
    rows_added = 0
    earliest: datetime | None = None

    for sealed_path in sorted(sealed_dir.glob("*.csv")):
        symbol = sealed_path.stem
        if symbol == "__CANARY__":
            continue
        research_path = research_dir / f"{symbol}.csv"
        if not research_path.exists():
            continue

        with sealed_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "available_time",
            ]
            sealed_rows = list(reader)

        if sealed_rows:
            first_sealed = min(
                ensure_utc(datetime.fromisoformat(row["timestamp"])) for row in sealed_rows
            )
            first_sealed_ts = int(first_sealed.timestamp())
        else:
            # Empty sealed file: usually a delisted name the vendor pull could not
            # fill past the embargo. The research cache still has its pre-embargo
            # history, and that is enough for the membership table to explain the
            # absence once the timeline is extended backward.
            first_sealed_ts = int(EMBARGO_START.timestamp())

        prefix_rows: list[dict[str, str]] = []
        with research_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ts = int(ensure_utc(datetime.fromisoformat(row["timestamp"])).timestamp())
                if first_ts <= ts < first_sealed_ts:
                    prefix_rows.append(row)
        if not prefix_rows:
            continue

        backfilled += 1
        rows_added += len(prefix_rows)
        prefix_first = ensure_utc(datetime.fromisoformat(prefix_rows[0]["timestamp"]))
        if earliest is None or prefix_first < earliest:
            earliest = prefix_first

        if dry_run:
            continue

        combined = prefix_rows + sealed_rows
        combined.sort(key=lambda r: r["timestamp"])
        with sealed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)

    summary = (
        f"{backfilled} symbol(s) backfilled, {rows_added} rows added "
        f"from {first.date().isoformat()}"
    )
    if earliest is not None:
        summary += f", earliest new bar {earliest.date().isoformat()}"
    console.print(summary)
    if dry_run:
        console.print("[yellow]dry run: no files written[/yellow]")


@app.command()
def declared(
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
) -> None:
    """List every pre-registered candidate and whether its seal has been spent."""
    with Registry(db) as registry:
        rows = registry.preregistrations()
        table = Table(box=None, pad_edge=False)
        for column in ("fingerprint", "strategy", "declared", "sealed run", "rule"):
            table.add_column(column)
        for row in rows:
            strategy = registry.get_strategy(row.fingerprint)
            spent = registry.sealed_run(row.fingerprint)
            table.add_row(
                row.fingerprint,
                strategy.name if strategy else "?",
                row.declared_at[:19],
                spent["sealed_run_at"][:19] if spent else "[green]unspent[/green]",
                row.selection_rule[:60],
            )
        console.print(table if rows else "[yellow]nothing pre-registered[/yellow]")


@app.command()
def run(
    fingerprint: str = typer.Argument(..., help="The pre-registered candidate."),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
    csv_root: str = typer.Option(SEALED_SP500_ROOT, help="Sealed cache to read."),
    universe: str = typer.Option(
        "sp500_pit", help="Point-in-time universe gating what may be held each session."
    ),
    timeframe: str = typer.Option("1D"),
    end: str = typer.Option(
        "", help="Last session to score. Default: yesterday."
    ),
    allow_unrecorded_ancestry: bool = typer.Option(
        False,
        help="Proceed when experiments in the ancestry predate the seal column. "
        "Their state is unknown, not clean, and the report says so.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Spend the seal on one pre-registered candidate.

    The order of the checks is the protocol: declared, then untainted, then
    read. Reading first and checking after would already have spent the thing
    the checks protect.
    """
    with Registry(db) as registry:
        declaration = registry.preregistration(fingerprint)
        if declaration is None:
            raise typer.BadParameter(
                f"{fingerprint} is not pre-registered. Declare it first with "
                "`aqr preregister`, before any sealed bar is read."
            )
        if registry.sealed_run(fingerprint) is not None:
            spent = registry.sealed_run(fingerprint)
            assert spent is not None
            raise typer.BadParameter(
                f"the seal was spent on {spent['sealed_run_at']} and there is no "
                "second one. The first result stands."
            )
        strategy = registry.get_strategy(fingerprint)
        if strategy is None:
            raise typer.BadParameter(f"no strategy {fingerprint!r} in {db}")

        taint = registry.ancestry_taint(fingerprint)
        console.print(f"[dim]{taint}[/dim]")
        if not taint.clean:
            raise typer.BadParameter(
                f"{fingerprint} is disqualified: {len(taint.tainted)} experiment(s) in "
                "its ancestry ran under a tainted seal. A rule selected with the "
                "answer in view cannot be validated against the answer."
            )
        if taint.unrecorded and not allow_unrecorded_ancestry:
            raise typer.BadParameter(
                f"{taint.unrecorded} experiment(s) in the ancestry predate the seal "
                "column, so their state is unknown rather than clean. Pass "
                "--allow-unrecorded-ancestry to proceed with that on the record."
            )

        spec = loads_spec(strategy.spec_yaml)
        console.print(
            f"running [bold]{spec.name}[/bold] [{fingerprint}]\n"
            f"declared {declaration.declared_at[:19]} under seal "
            f"{declaration.seal_digest[:16]}\n"
            f"rule: {declaration.selection_rule}"
        )

        # Promoted only now: every refusal above happens in a process that has
        # still read nothing, so a rejected candidate costs no seal at all.
        _promote()

        last = _parse_date(end) if end else _latest_available()
        first = datetime.fromtimestamp(
            EMBARGO_START.timestamp() - WARMUP_DAYS * 86_400, tz=UTC
        )
        membership = load_point_in_time(universe) if is_point_in_time(universe) else None
        data = _load_sealed(spec, csv_root, timeframe, first, last)
        actual_first_ts = min(int(bars.event_time[0]) for bars in data.values())
        actual_first = datetime.fromtimestamp(actual_first_ts, tz=UTC)
        primary = next(iter(data))
        warmup_sessions = sum(
            1 for t in data[primary].event_time if t < EMBARGO_START.timestamp()
        )
        console.print(
            f"{len(data)} of {len(spec.universe.symbols)} symbols loaded from "
            f"{csv_root}, requested warm-up from {first.date().isoformat()}, "
            f"actual earliest bar {actual_first.date().isoformat()} "
            f"({warmup_sessions} sessions before embargo)"
        )
        if actual_first.date() > first.date():
            console.print(
                f"[yellow]warning: sealed cache starts only "
                f"{(EMBARGO_START - actual_first).days} days before the embargo; "
                f"long-lookback features may be cold-started. "
                f"Rebuild with `aqr-sealed pull --start 2016-01-01 --force`.[/yellow]"
            )

        # This reading's ordinal, counted before it is recorded. One-shot is per
        # candidate, not per window: a new hypothesis is entitled to its own
        # sealed run, and a loop that could never take a second one would have to
        # stop after its first strategy. What it is not entitled to is silence
        # about how many candidates the window has now screened.
        looks = registry.sealed_looks() + 1
        if looks > 1:
            console.print(
                f"[yellow]look {looks} at the sealed window[/yellow] — "
                f"{looks - 1} candidate(s) were screened against it before this one, "
                f"so the bar for the alpha is t >= {multiplicity_bar(looks):.2f}"
            )

        measurement = measure_sealed_window(
            spec,
            data,
            since=EMBARGO_START,
            membership=membership,
            config=BacktestConfig(),
            looks=looks,
        )
        record = {
            "measurement": measurement.as_dict(),
            "seal": current().certificate(),
            "declaration": {
                "declared_at": declaration.declared_at,
                "selection_rule": declaration.selection_rule,
                "seal_digest": declaration.seal_digest,
            },
            "looks": looks,
            "ancestry": {
                "experiments": taint.experiments,
                "campaigns": list(taint.campaigns),
                "unrecorded": taint.unrecorded,
            },
            "csv_root": csv_root,
            "universe": universe,
        }
        try:
            stamp = registry.record_sealed_run(fingerprint, result=record)
        except PreregistrationError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if as_json:
        console.print_json(json.dumps(record, default=str))
    else:
        console.print()
        console.print(measurement.summary())
        console.print()
        console.print(f"[dim]recorded at {stamp}; the seal for {fingerprint} is spent[/dim]")
    raise typer.Exit(1 if measurement.refuted else 0)


# --------------------------------------------------------------------------- #


def _tickers(path: Path) -> list[str]:
    """Every ticker ever a member, not the ones that are members today.

    The names that left the index are exactly the ones whose absence creates
    survivorship bias, so the sealed cache has to hold them too or it is not a
    mirror of the research cache.
    """
    if not path.exists():
        raise typer.BadParameter(f"no universe file at {path}")
    return sorted(PointInTimeUniverse.from_json(path).all_symbols())


def _load_sealed(
    spec: object, csv_root: str, timeframe: str, start: datetime, end: datetime
) -> dict[str, Bars]:
    """Load the universe of the spec through the sealed provider, tolerating gaps.

    A name that was delisted inside the window has no file, and refusing the run
    over one of six hundred would make the sealed run impossible to perform for
    the reason the point-in-time universe exists.
    """
    symbols = list(getattr(getattr(spec, "universe", None), "symbols", ()) or ())
    provider = _sealed(CsvProvider(csv_root), "csv")
    data: dict[str, Bars] = {}
    missing: list[str] = []
    for symbol in symbols:
        try:
            bars = provider.load(symbol, start, end, timeframe)
        except Exception:
            missing.append(symbol)
            continue
        if len(bars):
            data[symbol] = bars
        else:
            missing.append(symbol)
    if missing:
        console.print(f"[yellow]no sealed bars for {len(missing)} symbols[/yellow]")
    if not data:
        raise typer.BadParameter(
            f"no symbol returned sealed bars from {csv_root}; run `aqr-sealed pull` first"
        )
    return data


# --------------------------------------------------------------------------- #
# The option sealed run — specs/10-options-research.md D3, D8
# --------------------------------------------------------------------------- #


@app.command("option-run")
def option_run(
    fingerprint: str = typer.Argument(..., help="The pre-registered option candidate."),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
    chain_root: str = typer.Option(
        SEALED_OPTIONS_ROOT, help="Sealed option chain cache to read."
    ),
    underlying_root: str = typer.Option(
        SEALED_UNDERLYING_ROOT,
        help="Sealed raw-adjusted bar cache for the underlying. Not the sealed "
        "equity cache: option strikes are set in raw terms and an adjusted "
        "close reports a moneyness the trade never had (specs/10 D0, D2a).",
    ),
    timeframe: str = typer.Option("1D"),
    equity: float = typer.Option(
        100_000.0,
        help="Account the measurement is sized against. Must be the size the "
        "research used, or the cycle count is a different experiment (D8a).",
    ),
    risk_per_trade: float = typer.Option(
        0.0,
        help="Override the spec's own risk fraction. Refused unless it matches "
        "what was pre-registered -- see the command's help.",
    ),
    allow_unrecorded_ancestry: bool = typer.Option(
        False,
        help="Proceed when experiments in the ancestry predate the seal column. "
        "Their state is unknown, not clean, and the report says so.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Spend the seal on one pre-registered option rule.

    The order of the checks is the protocol, exactly as in ``run``: declared,
    then untainted, then read. Reading first and checking after would already
    have spent the thing the checks protect.

    Two things are specific to options and both are refusals rather than
    conveniences.

    **The settlement boundary moves.** ``OptionBacktestConfig.settle_before``
    defaults to the research embargo, which refuses any entry expiring inside
    the reserved window (D3). In the sealed phase the reserved window is the
    part being measured, so the boundary becomes the sealed chain's own end --
    otherwise this command would refuse every entry in the years it came to
    read, report zero trades, and spend the seal on nothing.

    **The sizing may not drift.** specs/10 D8a measured the same rule producing
    21 independent cycles at 1% of equity and 57 at 2%, with nothing about the
    rule changed. A sealed run at a different size than the research is a
    different experiment wearing the pre-registration of the first, so
    ``--risk-per-trade`` is refused unless it matches the spec.
    """
    with Registry(db) as registry:
        declaration = registry.preregistration(fingerprint)
        if declaration is None:
            raise typer.BadParameter(
                f"{fingerprint} is not pre-registered. Declare it first with "
                "`aqr preregister`, before any sealed session is read."
            )
        spent_run = registry.sealed_run(fingerprint)
        if spent_run is not None:
            raise typer.BadParameter(
                f"the seal was spent on {spent_run['sealed_run_at']} and there is no "
                "second one. The first result stands."
            )
        strategy = registry.get_strategy(fingerprint)
        if strategy is None:
            raise typer.BadParameter(f"no strategy {fingerprint!r} in {db}")
        if strategy.family != OPTION:
            raise typer.BadParameter(
                f"{strategy.name} [{fingerprint}] is an {strategy.family} strategy; "
                "spend its seal with `python -m aqr.cli_sealed run`. The two measure "
                "different windows and are counted as different denominators."
            )

        taint = registry.ancestry_taint(fingerprint)
        console.print(f"[dim]{taint}[/dim]")
        if not taint.clean:
            raise typer.BadParameter(
                f"{fingerprint} is disqualified: {len(taint.tainted)} experiment(s) in "
                "its ancestry ran under a tainted seal. A rule selected with the "
                "answer in view cannot be validated against the answer."
            )
        if taint.unrecorded and not allow_unrecorded_ancestry:
            raise typer.BadParameter(
                f"{taint.unrecorded} experiment(s) in the ancestry predate the seal "
                "column, so their state is unknown rather than clean. Pass "
                "--allow-unrecorded-ancestry to proceed with that on the record."
            )

        spec = loads_option_spec(strategy.spec_yaml)
        if risk_per_trade and abs(risk_per_trade - spec.sizing.risk_per_trade) > 1e-12:
            raise typer.BadParameter(
                f"--risk-per-trade {risk_per_trade} but {spec.name} was pre-registered "
                f"at {spec.sizing.risk_per_trade}. specs/10 D8a: the same rule produced "
                "21 independent cycles at 1% and 57 at 2% with nothing about the rule "
                "changed, so a sealed run at another size is a different experiment "
                "wearing this one's declaration."
            )
        # Both caches, checked as *paths* before anything is read. A missing
        # sealed underlying is the likely first failure here -- the sealed chain
        # has existed since the embargo split and its raw-adjusted underlying is
        # a separate pull -- and without this it surfaces as a FileNotFoundError
        # from three layers down, in a process that has already promoted itself.
        # Nothing would be lost (the seal is spent by ``record_sealed_run``, at
        # the end), but a one-shot command should fail with the command that
        # fixes it rather than with a traceback.
        chain_file = Path(chain_root) / "option_chain" / f"{spec.underlying}.csv"
        bar_file = Path(underlying_root) / timeframe / f"{spec.underlying}.csv"
        if not chain_file.exists():
            raise typer.BadParameter(
                f"no sealed option chain at {chain_file}. Build it with "
                "`aqr options-pull` and then `aqr options-embargo`."
            )
        if not bar_file.exists():
            raise typer.BadParameter(
                f"no sealed underlying bars at {bar_file}. Settlement reads the "
                f"underlying's close on each expiration date, so the sealed run "
                f"cannot settle anything without them. Pull them RAW -- an "
                f"adjusted close compared against a strike reports a moneyness "
                f"the trade never had (specs/10 D0):\n"
                f"  python -m aqr.cli_sealed pull --symbols {spec.underlying} "
                f"--adjustment raw --csv-root {underlying_root} "
                f"--timeframe {timeframe}"
            )

        console.print(
            f"running [bold]{spec.name}[/bold] [{fingerprint}]\n"
            f"{spec.structure.type} on {spec.underlying}, "
            f"{spec.structure.dte.target} DTE, anchor delta {spec.structure.anchor.delta}\n"
            f"declared {declaration.declared_at[:19]} under seal "
            f"{declaration.seal_digest[:16]}\n"
            f"rule: {declaration.selection_rule}"
        )

        # Promoted only now: every refusal above happens in a process that has
        # still read nothing, so a rejected candidate costs no seal at all.
        _promote()

        market, dataset_version = sealed_option_market(
            spec.underlying,
            provider=_sealed(CsvProvider(underlying_root), "csv"),
            chain_root=chain_root,
            timeframe=timeframe,
            since=EMBARGO_START.date(),
        )
        sessions = market.chain.sessions
        if not sessions:
            raise typer.BadParameter(
                f"no sealed chain sessions in {chain_root}; run `aqr options-pull` "
                "and `aqr options-embargo` first."
            )
        embargoed = [s for s in sessions if s >= EMBARGO_START.date()]
        console.print(
            f"{len(sessions)} chain sessions {sessions[0]} -> {sessions[-1]}, "
            f"{len(embargoed)} of them past the embargo; "
            f"{len(market.underlying)} underlying bars from {underlying_root}"
        )
        console.print(f"[dim]{dataset_version}[/dim]")
        if not embargoed:
            raise typer.BadParameter(
                f"the sealed chain stops at {sessions[-1]}, before the embargo at "
                f"{EMBARGO_START.date()}. There is nothing to measure, and spending "
                "the seal on it would spend it on the research window twice."
            )

        # D3, in the sealed phase: the boundary is the sealed root's own end, not
        # the research embargo. Left at the default this run would refuse every
        # entry expiring after 2024-09-01 -- which is all of them.
        settle_before = sessions[-1] + timedelta(days=1)
        config = OptionBacktestConfig(
            initial_equity=equity,
            costs=spec_costs(),
            settle_before=settle_before,
        )

        # This reading's ordinal within the option family. The equity sealed
        # window is different data and a candidate screened against it has not
        # consumed a look at this one.
        looks = registry.sealed_looks(family=OPTION) + 1
        if looks > 1:
            console.print(
                f"[yellow]look {looks} at the sealed option window[/yellow] — "
                f"{looks - 1} candidate(s) were screened against it before this one, "
                f"so the bar for the alpha is t >= {multiplicity_bar(looks):.2f}"
            )

        measurement = measure_sealed_option_window(
            spec, market, since=EMBARGO_START, config=config, looks=looks
        )
        record = {
            "measurement": measurement.as_dict(),
            "seal": current().certificate(),
            "declaration": {
                "declared_at": declaration.declared_at,
                "selection_rule": declaration.selection_rule,
                "seal_digest": declaration.seal_digest,
            },
            "looks": looks,
            "ancestry": {
                "experiments": taint.experiments,
                "campaigns": list(taint.campaigns),
                "unrecorded": taint.unrecorded,
            },
            "chain_root": chain_root,
            "underlying_root": underlying_root,
            "dataset_version": dataset_version,
            "settle_before": settle_before.isoformat(),
            "initial_equity": equity,
            "risk_per_trade": spec.sizing.risk_per_trade,
            "family": OPTION,
        }
        try:
            stamp = registry.record_sealed_run(fingerprint, result=record)
        except PreregistrationError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if as_json:
        console.print_json(json.dumps(record, default=str))
    else:
        console.print()
        console.print(measurement.summary())
        console.print()
        console.print(
            "[dim]about 25 independent 28-DTE cycles: this window is entitled to say "
            "the rule stopped working and is not entitled to say it works "
            "(specs/10 D8). No artefact reading this result may word it "
            "otherwise.[/dim]"
        )
        console.print(f"[dim]recorded at {stamp}; the seal for {fingerprint} is spent[/dim]")
    raise typer.Exit(1 if measurement.refuted else 0)


def spec_costs() -> OptionCostModel:
    """The schedule a sealed option run is charged under.

    ``IBKR_OPTIONS`` -- the same default ``OptionBacktestConfig`` carries, so the
    sealed window is charged what the search window was charged. Named in its own
    function rather than inlined because the alternative, a ``--costs`` option,
    would let a sealed run be re-charged more cheaply than the research that
    selected the rule, and that is a way to turn a refutation into a pass.
    """
    return IBKR_OPTIONS


def main() -> None:
    app()


if __name__ == "__main__":
    main()
