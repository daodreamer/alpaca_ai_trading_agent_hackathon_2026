"""Command line interface.

    aqr features                     list the DSL vocabulary
    aqr validate STRATEGY.yaml       parse and sanity-check without trading
    aqr backtest STRATEGY.yaml       one in-sample run with costs
    aqr walkforward STRATEGY.yaml    the number that actually counts
    aqr evaluate STRATEGY.yaml       the full gauntlet, scored and recorded
    aqr research                     the LLM research loop
    aqr registry / experiments       what has been tried and what survived
    aqr preregister FINGERPRINT      declare a candidate for the sealed run
    aqr target-book FINGERPRINT      write the book. Nothing here places it
    aqr costs                        what an order costs, under each schedule

Options research lives beside it, with its own vocabulary and its own search
budget (specs/10 D8 -- the two denominators must not be mixed):

    aqr option-features              structures and features an option rule may name
    aqr option-research              the option research loop, capped at 20 hypotheses
    aqr option-book FINGERPRINT      write the rule. Strikes are resolved by the executor

Every command that touches data takes ``--source`` (synthetic or yahoo) plus a
window, so a result can be reproduced from the command line that produced it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from aqr.agent.option_prompt import STRUCTURE_CATALOGUE
from aqr.agent.option_proposer import OptionProposer
from aqr.agent.option_research import (
    OPTION_SEARCH_BUDGET,
    WIDE_SEARCH_WARNING_AT,
    OptionResearchConfig,
    OptionResearchLoop,
)
from aqr.agent.proposer import Proposer
from aqr.agent.research import ResearchConfig, ResearchLoop
from aqr.backtest.costs import PRESETS, CostModel, preset
from aqr.backtest.engine import BacktestConfig
from aqr.backtest.metrics import compute_metrics
from aqr.backtest.run import run_strategy
from aqr.data.alpaca import AlpacaProvider
from aqr.data.bars import Bars, bars_per_year
from aqr.data.cross_source import compare as compare_sources
from aqr.data.embargo import (
    SP500_RESEARCH_ROOT,
    SP500_SEALED_ROOT,
    ResearchProvider,
    audit_cache_root,
)
from aqr.data.ibkr import (
    GATEWAY_PAPER_PORT,
    TWS_PAPER_PORT,
    IbkrProvider,
    probe_durations,
    probe_requests,
)
from aqr.data.option_embargo import (
    audit_option_root,
    split_at_embargo,
    write_option_canary,
)
from aqr.data.options_chain import (
    CHAIN_COLUMNS,
    VOLATILITY_COLUMNS,
    download_table,
    split_table,
)
from aqr.data.providers import CsvProvider, Provider, SyntheticProvider, YFinanceProvider
from aqr.data.quality import inspect as inspect_bars
from aqr.data.universes import (
    PIT_CSV_ROOTS,
    PointInTimeUniverse,
    is_point_in_time,
    load_point_in_time,
    universe_names,
)
from aqr.data.universes import resolve as resolve_universe
from aqr.dsl.loader import load_file
from aqr.dsl.loader import loads as loads_spec
from aqr.dsl.schema import StrategySpec
from aqr.dsl.validator import validate_against
from aqr.features.regime import regime_series
from aqr.features.registry import REGISTRY
from aqr.option_book import build_option_book, write_option_book
from aqr.option_data import (
    DEFAULT_UNDERLYING_ROOT,
    research_option_market,
)
from aqr.options.costs import PRESETS as OPTION_PRESETS
from aqr.options.engine import OptionBacktestConfig
from aqr.options.features import OPTION_FEATURES
from aqr.options.spec import loads_option_spec
from aqr.pipeline import evaluate_candidate
from aqr.registry.db import (
    EQUITY,
    OPTION,
    PreregistrationError,
    Registry,
    StrategyRecord,
)
from aqr.seal import CANARY_SYMBOL, EMBARGO_START
from aqr.seal import current as current_seal
from aqr.target_book import build_target_book, write_book
from aqr.validation.holdout import run_holdout
from aqr.validation.sealed import multiplicity_bar
from aqr.validation.splits import TEST_BARS, TRAIN_BARS
from aqr.validation.walkforward import run_walk_forward

app = typer.Typer(
    add_completion=False,
    help="AI Quant Researcher — LLM-driven strategy research with an honest scorer.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_SYMBOLS = "SPY,QQQ,IWM,DIA"
SYMBOLS_HELP = (
    "Extra symbols to load. The strategy's own universe is always included, so "
    "a run can never silently trade a subset of what the spec asks for."
)
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-01-01"

SEALED_SP500_ROOT = "data-sp500-sealed"
"""The only cache root that holds sessions up to the present.

Named here as well as in ``cli_sealed`` rather than imported from it: this
process must not import the sealed entry point, which is what keeps "search" and
"read the embargoed years" in two binaries.
"""

# A book needs every feature warm on its final session, and ``rs_rank(126)`` is
# the longest lookback the DSL offers. 1200 calendar days clears it with room for
# holidays and thin names, and costs far less than reloading fifteen years to
# answer a question about one session.
TARGET_BOOK_WARMUP_DAYS = 1200


def _parse_date(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


SOURCES = ("synthetic", "csv", "yahoo", "alpaca", "ibkr")


# Which process is listening on which port is a property of the running TWS or
# Gateway, not of this code. Two of the four are the real account, and a user
# skimming a log line will not skim past "LIVE".
_IBKR_ENDPOINTS: dict[int, str] = {
    TWS_PAPER_PORT: "TWS paper",
    7496: "TWS LIVE",
    GATEWAY_PAPER_PORT: "IB Gateway paper",
    4001: "IB Gateway LIVE",
}


def describe_endpoint(port: int) -> str:
    """Which gateway a port belongs to, named rather than guessed."""
    known = _IBKR_ENDPOINTS.get(port)
    return f"{known} (port {port})" if known else f"unrecognised port {port}"


def _provider(
    source: str,
    csv_root: str,
    *,
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = TWS_PAPER_PORT,
    ibkr_client_id: int = 17,
    ibkr_what_to_show: str = "ADJUSTED_LAST",
) -> Provider:
    """One place that knows how to build each provider.

    ``pull`` and ``_load`` must never disagree about what ``--source alpaca``
    means, or a cached file will carry bars a later run cannot reproduce.
    """
    if source == "synthetic":
        return SyntheticProvider()
    if source == "csv":
        return CsvProvider(csv_root)
    if source == "yahoo":
        return YFinanceProvider()
    if source == "alpaca":
        return AlpacaProvider()
    if source == "ibkr":
        return IbkrProvider(
            host=ibkr_host,
            port=ibkr_port,
            client_id=ibkr_client_id,
            what_to_show=ibkr_what_to_show,
        )
    raise typer.BadParameter(f"unknown source {source!r}; use one of {', '.join(SOURCES)}")


def _dataset_version(provider: Provider, source: str, start: str, end: str, timeframe: str) -> str:
    """What produced these bars, recorded with every experiment.

    A provider that can name its own feed and adjustment does so; two runs whose
    dataset versions differ are not comparable, and comparing them anyway is how
    a data change gets attributed to a strategy change.
    """
    describe = getattr(provider, "dataset_version", None)
    detail = describe(timeframe) if callable(describe) else f"{source}:{timeframe}"
    return f"{detail}:{start}:{end}"


DEFAULT_CSV_ROOT = "data-sp500"
DEFAULT_OPTIONS_ROOT = "data-options"
DEFAULT_OPTIONS_SEALED_ROOT = "data-options-sealed"


def _csv_root_for(universe: str, csv_root: str) -> str:
    """Point a point-in-time universe at the cache that actually holds it.

    ``data/`` was pulled from today's constituent list and is missing every name
    that left the index, so running ``--universe sp500_pit`` against it would
    quietly reintroduce the survivorship bias the universe exists to remove.
    Only the default is redirected; an explicit ``--csv-root`` is obeyed.
    """
    if csv_root != DEFAULT_CSV_ROOT or not is_point_in_time(universe):
        return csv_root
    root = PIT_CSV_ROOTS[universe.strip().lower()]
    console.print(f"[dim]--csv-root defaulted to {root} for {universe}[/dim]")
    return root


def _load(
    source: str,
    symbols: list[str],
    start: str,
    end: str,
    timeframe: str,
    csv_root: str,
    *,
    universe: str = "",
    tolerant: bool = False,
) -> tuple[dict[str, Bars], dict[str, list[str]] | None, PointInTimeUniverse | None, str]:
    """Load bars, regime labels, and the membership table the run is gated by.

    ``tolerant`` decides what a missing symbol means. Exploring a named universe,
    a delisted or newly-listed ticker is expected and skipping it is correct. But
    when a strategy file names its universe, that list is a contract: quietly
    trading a subset would make two runs of "the same" strategy incomparable, so
    the failure propagates.

    ``universe`` is the *name* the run was launched with, and it is what decides
    whether a membership table comes back. Resolving it here rather than in each
    command is what keeps a point-in-time universe from being loaded by one
    command and gated by none.
    """
    membership = load_point_in_time(universe) if is_point_in_time(universe) else None
    # "".split(",") is [""], and the synthetic provider will cheerfully generate
    # bars for a symbol named "". Catching it here keeps a mis-specified run
    # from producing plausible-looking results for an instrument that does not
    # exist.
    blank = [s for s in symbols if not s.strip()]
    if blank or not symbols:
        raise typer.BadParameter("symbol list is empty or contains a blank entry")
    window = (_parse_date(start), _parse_date(end))
    provider = _provider(source, _csv_root_for(universe, csv_root))

    data: dict[str, Bars] = {}
    failures: list[str] = []
    for symbol in symbols:
        try:
            data[symbol] = provider.load(symbol, *window, timeframe)
        except Exception as exc:
            if not tolerant:
                raise
            failures.append(f"{symbol} ({type(exc).__name__})")
    if failures:
        console.print(f"[yellow]no data for {len(failures)}: {', '.join(failures)}[/yellow]")
    if not data:
        raise typer.BadParameter("no symbol returned any data")

    if isinstance(provider, SyntheticProvider):
        # The simulator knows the regime it generated; nothing estimated can
        # beat ground truth, so use it when it exists.
        labels = {s: provider.regimes(s, *window, timeframe) for s in data}
    else:
        # For real bars the regime is itself an estimate -- a causal one, from
        # trailing windows only. Passing nothing here used to leave the
        # evaluator's regime term at a hard-coded 0.5, which is a tenth of every
        # real-data score standing in for a measurement nobody made.
        labels = {s: regime_series(b) for s, b in data.items()}
    return data, labels, membership, _dataset_version(provider, source, start, end, timeframe)


def _symbols_for(spec: object, requested: str) -> list[str]:
    """Union of the strategy's universe and anything the user asked for.

    Loading fewer symbols than the universe names would either crash or, worse,
    quietly evaluate a different strategy than the file describes.
    """
    universe = list(getattr(getattr(spec, "universe", None), "symbols", ()) or ())
    extra = [s.strip().upper() for s in requested.split(",") if s.strip()]
    seen: list[str] = []
    for symbol in [*universe, *extra]:
        if symbol not in seen:
            seen.append(symbol)
    return seen


def _universe(name: str, symbols: str, limit: int = 0) -> list[str]:
    """Resolve ``--universe`` and ``--symbols`` into one list.

    A named universe is a convenience, not a guarantee: constituents change and
    some names have too little history for a long backtest. Whatever the loader
    cannot fetch is dropped there, loudly, rather than here.
    """
    names: list[str] = []
    if name.strip():
        names = resolve_universe(name, limit or None)
    names += [s.strip().upper() for s in symbols.split(",") if s.strip()]
    unique: list[str] = []
    for symbol in names:
        if symbol not in unique:
            unique.append(symbol)
    if not unique:
        raise typer.BadParameter("no symbols: pass --symbols or --universe")
    return unique


def _proposer(provider: str, model: str) -> Proposer:
    """Build the proposer for ``--provider``.

    Imported lazily and per-branch so the offline loop never needs an LLM SDK
    installed, and so a missing API key surfaces as a clear message here rather
    than as an exception ten iterations into a run.
    """
    from aqr.agent.proposer import AnthropicProposer, DeepSeekProposer, HeuristicProposer

    choice = provider.strip().lower()
    if choice == "offline":
        return HeuristicProposer()
    if choice == "deepseek":
        return DeepSeekProposer(model or "deepseek-chat")
    if choice == "anthropic":
        return AnthropicProposer(model or "claude-opus-5")
    raise typer.BadParameter(
        f"unknown provider {provider!r}; use offline, deepseek or anthropic"
    )


COSTS_HELP = (
    "Named cost schedule: ibkr_fixed (the default, $0.005/share with a $1 order "
    "floor), alpaca (commission-free — the venue these bars come from), or zero. "
    "Which one is right depends on where the book is executed. Run `aqr costs` "
    "to see what each charges on a position of the size you actually hold: on a "
    "wide book at a small account the order floor, not the spread, is the "
    "dominant cost and the cost gate is fatal."
)


def _costs(
    spread_bps: float, slippage_bps: float, commission: float, schedule: str = ""
) -> CostModel:
    """Build the cost model, from a named schedule or from the individual knobs.

    A named schedule wins outright rather than being merged with the knobs:
    half a preset and half a set of overrides is a schedule no broker offers, and
    the record would name one that was not used.
    """
    if schedule.strip():
        try:
            return preset(schedule)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    return CostModel(
        commission_per_share=commission, spread_bps=spread_bps, slippage_bps=slippage_bps
    )


# --------------------------------------------------------------------------- #


@app.command()
def features() -> None:
    """List every feature the strategy DSL accepts."""
    table = Table("feature", "arity", "description", title=f"{len(REGISTRY)} features")
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        table.add_row(name, str(spec.arity), spec.doc)
    console.print(table)


@app.command()
def providers() -> None:
    """Which LLM credentials are visible, and where they were found.

    Reports the length of each key, never the key. "Is it loaded" is a question
    people otherwise answer by echoing a secret into a terminal.
    """
    from aqr.config import describe, load_env_files

    files = load_env_files()
    if files:
        console.print("loaded: " + ", ".join(str(f) for f in files))
    else:
        console.print("[yellow]no .env.local or .env found above the working directory[/yellow]")

    table = Table("provider", "variable", "status")
    for status in describe(load=False):
        mark = f"[green]set ({status.length} chars)[/green]" if status.present else "[dim]-[/dim]"
        table.add_row(status.provider, status.variable, mark)
    console.print(table)
    console.print("\nUse one with: aqr research --provider deepseek")


@app.command("ibkr-check")
def ibkr_check(
    symbol: str = typer.Option("AAPL", help="A liquid US stock to test with."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(
        TWS_PAPER_PORT,
        help="7497 TWS paper, 7496 TWS live, 4002 Gateway paper, 4001 Gateway live.",
    ),
    client_id: int = typer.Option(17, help="IBKR API client id. Must be unused."),
    timeout: float = typer.Option(30.0, help="Seconds to wait per request."),
) -> None:
    """Find out which historical-data requests this TWS will actually answer.

    A failing `reqHistoricalData` is ambiguous: ADJUSTED_LAST refusing an
    explicit end date, a missing market-data subscription, an unaccepted
    duration and a short timeout all look identical from outside. This sends
    every plausible shape once and reports what came back.

    Reads only. It qualifies a contract and asks for bars; it touches no account
    or order state, and cannot place one.
    """
    console.print(f"[cyan]connecting to {host}: {describe_endpoint(port)}[/cyan]")
    provider = IbkrProvider(host=host, port=port, client_id=client_id)
    ib = provider._connect()  # noqa: SLF001 - the probe is part of this module's job
    try:
        results = probe_requests(ib, symbol.upper(), timeout=timeout)
    finally:
        provider.close()

    console.print()
    console.print(f"{symbol.upper()} 1 day bars:")
    for result in results:
        colour = "green" if not result.error else "red"
        console.print(f"[{colour}]{result}[/{colour}]")

    working = [r for r in results if not r.error]
    console.print()
    if not working:
        console.print(
            "[red]Nothing worked.[/red] Every shape failing points at the account rather "
            "than the request: US equity historical bars need a market-data subscription "
            "(Account -> Manage Account -> Market Data Subscriptions). Check TWS's own "
            "chart for this symbol -- if that is empty too, it is the subscription."
        )
        raise typer.Exit(code=1)
    adjusted = [r for r in working if r.what_to_show == "ADJUSTED_LAST"]
    undated_only = bool(adjusted) and not any(r.dated for r in adjusted)
    if undated_only:
        console.print(
            "[yellow]ADJUSTED_LAST works only without an end date[/yellow], so it cannot "
            "be chunked: all of its history has to arrive in one request. How much that "
            "is, is an account and build fact rather than a documented one --"
        )
        console.print()
        console.print("how much history one request returns:")
        provider = IbkrProvider(host=host, port=port, client_id=client_id)
        ib = provider._connect()  # noqa: SLF001
        try:
            ladder = probe_durations(ib, symbol.upper(), timeout=max(timeout, 60.0))
        finally:
            provider.close()
        for step in ladder:
            colour = "green" if not step.error else "red"
            console.print(f"[{colour}]  {step.what_to_show:<6} {_ladder_line(step)}[/{colour}]")
        best = max((s for s in ladder if not s.error), key=lambda s: s.bars, default=None)
        if best:
            years = best.bars / 252
            console.print()
            console.print(
                f"deepest working window: [green]{best.what_to_show}[/green] "
                f"-> {best.bars} bars (~{years:.0f} years)"
            )
    console.print()
    console.print(
        "Use: aqr pull --source ibkr "
        f"--ibkr-port {port} --ibkr-what-to-show {working[0].what_to_show}"
    )


def _symbols_from_universe_file(path: str, extra: list[str]) -> list[str]:
    """Every ticker ever a member, plus anything named explicitly."""
    universe = PointInTimeUniverse.from_json(path)
    return sorted(set(universe.all_symbols()) | set(extra))


def _ladder_line(step: object) -> str:
    bars = getattr(step, "bars", 0)
    error = getattr(step, "error", None)
    return f"FAILED  {str(error)[:60]}" if error else f"ok      {bars} bars"


@app.command()
def pull(
    universe: str = typer.Option(
        "", help=f"Named universe: {', '.join(universe_names())}. Combines with --symbols."
    ),
    symbols: str = typer.Option("", help="Comma-separated symbols to fetch."),
    universe_limit: int = typer.Option(0, help="Take only the largest N of a named universe."),
    source: str = typer.Option("alpaca", help="Where to fetch from: alpaca, ibkr or yahoo."),
    feed: str = typer.Option(
        "sip", help="Alpaca feed. 'sip' is the consolidated tape; 'iex' is one sparse venue."
    ),
    adjustment: str = typer.Option(
        "all",
        help="Alpaca price adjustment: all, split, dividend or raw. 'all' is right for "
        "equity research, where an unadjusted split is a -75% return that never "
        "happened. It is WRONG for an options underlying: strikes are set in raw "
        "terms and do not move for an ordinary dividend, so an adjusted close "
        "compared against a strike reports a moneyness the trade never had. Use "
        "'raw' for anything that will be settled against an option contract.",
    ),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D", help="1m, 5m, 15m, 30m, 1h, 1D or 1W."),
    csv_root: str = typer.Option(
        DEFAULT_CSV_ROOT, help="Cache root: <root>/<timeframe>/<symbol>.csv."
    ),
    force: bool = typer.Option(False, help="Refetch symbols already cached."),
    keep_suspect: bool = typer.Option(
        False, help="Cache series that failed the quality check anyway. Say why in the journal."
    ),
    ibkr_host: str = typer.Option("127.0.0.1", help="TWS / IB Gateway host."),
    ibkr_port: int = typer.Option(
        TWS_PAPER_PORT,
        help="7497 TWS paper, 7496 TWS live, 4002 Gateway paper, 4001 Gateway live.",
    ),
    ibkr_client_id: int = typer.Option(17, help="IBKR API client id. Must be unused."),
    universe_file: str = typer.Option(
        "",
        help="Point-in-time universe JSON. Pulls every ticker ever a member, which is "
        "the whole point: a pull driven by today's members is missing exactly the "
        "names whose absence created the survivorship bias.",
    ),
    ibkr_what_to_show: str = typer.Option(
        "ADJUSTED_LAST",
        help="ADJUSTED_LAST is split- and dividend-adjusted but IBKR may refuse it "
        "with an explicit end date; TRADES always works and is not adjusted. "
        "Run `aqr ibkr-check` to find out which this TWS accepts.",
    ),
) -> None:
    """Fetch real bars into the local cache, so every later run is offline.

    This is the only command that reaches a network for market data. Everything
    downstream reads ``--source csv``, which means a research run is
    reproducible: the same command a year from now sees the same bars, rather
    than whatever the vendor has since restated.
    """
    if source == "csv":
        raise typer.BadParameter("pulling from the cache into the cache does nothing")
    if source == "synthetic":
        raise typer.BadParameter("the synthetic provider needs no cache; it is already offline")

    if universe_file:
        extra = _universe(universe, symbols, universe_limit) if (universe or symbols) else []
        requested = _symbols_from_universe_file(universe_file, extra)
    else:
        requested = _universe(universe, symbols, universe_limit)
    if source == "alpaca":
        provider: Provider = AlpacaProvider(feed=feed, adjustment=adjustment)
    else:
        provider = _provider(
            source,
            csv_root,
            ibkr_host=ibkr_host,
            ibkr_port=ibkr_port,
            ibkr_client_id=ibkr_client_id,
            ibkr_what_to_show=ibkr_what_to_show,
        )
    if source == "ibkr":
        # Said before the connection, not after: two of the four ports are the
        # real account. Nothing here can place an order, but that is a claim
        # about today's code, and the user is the one carrying the risk.
        console.print(f"[cyan]connecting to {ibkr_host}: {describe_endpoint(ibkr_port)}[/cyan]")
    # Wrapped by default. Without this, ``aqr pull --end 2026-08-27`` writes
    # embargoed bars into whatever root is named and only ``seal-check`` notices,
    # after the fact. The wrapper clamps the request before it is sent, so the
    # rows never exist locally to be read by accident.
    #
    # There is deliberately no --sealed escape hatch here. The embargoed years
    # are fetched by a separate entry point in a separate process, because this
    # module also runs `research`: a flag that let one process both search and
    # read the sealed years would make the phase separation a convention again.
    # ``test_no_module_outside_the_embargo_layer_constructs_a_seal_token`` fails
    # the build if this file so much as names the token.
    provider = ResearchProvider(provider, label=source)

    cache = CsvProvider(csv_root)
    window = (_parse_date(start), _parse_date(end))

    fetched, skipped, failed = 0, 0, []
    suspect: list[object] = []
    for symbol in requested:
        target = cache.path_for(symbol, timeframe)
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            bars = provider.load(symbol, *window, timeframe)
        except Exception as exc:
            # One delisted ticker must not abandon the other forty-nine.
            failed.append(f"{symbol} ({type(exc).__name__}: {exc})")
            continue
        # Check before writing. A series with a two-year hole in it is worse
        # than no series: the backtester indexes positionally, so the hole is
        # invisible to every defence downstream of here.
        report = inspect_bars(bars, requested_start=window[0])
        if report.ok:
            cache.write(bars)
            fetched += 1
            console.print(f"[green]{report}[/green]")
        else:
            suspect.append(report)
            console.print(f"[yellow]{report}[/yellow]")
            if not keep_suspect:
                continue
            cache.write(bars)
            fetched += 1

    close = getattr(provider, "close", None)
    if callable(close):
        close()

    console.print()
    console.print(
        f"{fetched} fetched, {skipped} already cached, {len(failed)} failed "
        f"-> {Path(csv_root) / timeframe}"
    )
    console.print(f"dataset version: {_dataset_version(provider, source, start, end, timeframe)}")
    for line in failed:
        console.print(f"[red]{line}[/red]")
    if suspect:
        verb = "cached anyway" if keep_suspect else "not cached"
        console.print(
            f"[yellow]{len(suspect)} series failed the quality check and were "
            f"{verb}. Try --feed sip, a later --start, or another --source.[/yellow]"
        )


@app.command()
def compare(
    left: str = typer.Option(..., help="First cache root, e.g. data-yahoo."),
    right: str = typer.Option(..., help="Second cache root, e.g. data-ibkr."),
    symbols: str = typer.Option("", help="Comma-separated. Default: everything in both."),
    timeframe: str = typer.Option("1D"),
    tolerance: float = typer.Option(
        0.01, help="Daily-return difference that counts as a disagreement."
    ),
    quiet: bool = typer.Option(False, help="Only print symbols that disagree."),
) -> None:
    """Check two cached sources against each other, offline.

    Two correctly adjusted series for one instrument may disagree about the
    price *level* -- different adjustment bases, different dividend treatment --
    but they cannot disagree about a daily *return*. So the comparison is on
    returns, and a day where one vendor says -49.7% and the other says +0.3% is
    one of them failing to adjust a split.

    This cannot say which vendor is right. It says which dates to look at, which
    is the part that does not scale by hand.
    """
    left_root, right_root = Path(left) / timeframe, Path(right) / timeframe
    for root in (left_root, right_root):
        if not root.exists():
            raise typer.BadParameter(f"no cache at {root}")

    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    shared = sorted(
        {p.stem for p in left_root.glob("*.csv")} & {p.stem for p in right_root.glob("*.csv")}
    )
    if wanted:
        shared = [s for s in shared if s in wanted]
    if not shared:
        raise typer.BadParameter("no symbol is cached in both roots")

    left_provider, right_provider = CsvProvider(left), CsvProvider(right)
    far_past = datetime(1990, 1, 1, tzinfo=UTC)
    far_future = datetime(2100, 1, 1, tzinfo=UTC)

    disagreeing = 0
    incomparable = 0
    split_notes: list[str] = []
    for symbol in shared:
        try:
            a = left_provider.load(symbol, far_past, far_future, timeframe)
            b = right_provider.load(symbol, far_past, far_future, timeframe)
        except Exception as exc:
            console.print(f"[red]{symbol}: {type(exc).__name__}: {exc}[/red]")
            continue
        report = compare_sources(
            a, b, left_name=left, right_name=right, tolerance=tolerance
        )
        if not report.comparable:
            # Not a disagreement: nothing was measured. Saying otherwise
            # sends the reader after a data problem that does not exist.
            incomparable += 1
            if not quiet:
                console.print(f"[dim]{report}[/dim]")
            continue
        if report.agrees:
            if not quiet:
                console.print(f"[green]{report}[/green]")
            continue
        disagreeing += 1
        console.print(f"[yellow]{report}[/yellow]")
        split_notes += report.suspected_splits()

    console.print()
    console.print(
        f"{len(shared) - incomparable} symbols compared, {disagreeing} disagree "
        f"beyond {tolerance:.1%}"
        + (f", {incomparable} had no shared sessions" if incomparable else "")
    )
    if split_notes:
        console.print(
            f"[yellow]{len(split_notes)} look like unadjusted corporate actions.[/yellow] "
            "Prefer the source that already adjusted them."
        )


@app.command("inspect")
def inspect_cache(
    symbols: str = typer.Option("", help="Comma-separated symbols. Default: everything cached."),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT, help="Cache root to read."),
    verbose: bool = typer.Option(False, help="List the exact dates behind each complaint."),
    problems_only: bool = typer.Option(False, help="Hide series with nothing to report."),
) -> None:
    """Re-check cached bars, offline.

    A pull reports quality once and then the cache is opaque. Verifying a
    complaint -- is that 45% move a real earnings gap or an unadjusted split? --
    should not need another round trip to a vendor, so this reads the files that
    are already on disk and, with --verbose, names the dates to look at.
    """
    root = Path(csv_root) / timeframe
    if not root.exists():
        raise typer.BadParameter(f"no cache at {root}")
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    paths = (
        [root / f"{s}.csv" for s in wanted] if wanted else sorted(root.glob("*.csv"))
    )

    provider = CsvProvider(csv_root)
    far_past = datetime(1990, 1, 1, tzinfo=UTC)
    far_future = datetime(2100, 1, 1, tzinfo=UTC)
    flagged = 0
    for path in paths:
        if not path.exists():
            console.print(f"[red]{path.stem}: not cached[/red]")
            continue
        bars = provider.load(path.stem, far_past, far_future, timeframe)
        report = inspect_bars(bars)
        if problems_only and report.ok and not report.notes():
            continue
        colour = "green" if report.ok else "yellow"
        console.print(f"[{colour}]{report}[/{colour}]")
        if not report.ok:
            flagged += 1
        if verbose:
            for line in _explain(bars, report):
                console.print(f"      {line}")

    console.print()
    console.print(f"{len(paths)} series, {flagged} with something to check")


@app.command()
def preregister(
    fingerprint: str = typer.Argument(..., help="The candidate, by fingerprint."),
    rule: str = typer.Option(
        ...,
        "--rule",
        help="How this candidate was selected, in words. 'The highest-scoring of "
        "305 hypotheses' and 'the only one that cleared t > 2' are different "
        "claims about the multiple-comparisons problem.",
    ),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
) -> None:
    """Declare a candidate for the sealed run, before any sealed bar is read.

    Deliberately here rather than in the sealed entry point. A declaration made
    by the process that is about to read the answer is not a declaration, and
    keeping the two commands in two binaries makes the ordering visible in the
    shell history rather than only in the database.

    This command reads no market data at all, so running it cannot spend the
    seal or taint anything.
    """
    with Registry(db) as registry:
        strategy = registry.get_strategy(fingerprint)
        if strategy is None:
            raise typer.BadParameter(f"no strategy {fingerprint!r} in {db}")

        taint = registry.ancestry_taint(fingerprint)
        console.print(f"[dim]{taint}[/dim]")
        if not taint.clean:
            # Declared anyway would be worse than refused: the record would then
            # carry a pre-registration for a candidate that can never be run,
            # and someone would eventually run it.
            raise typer.BadParameter(
                f"{fingerprint} is disqualified: {len(taint.tainted)} experiment(s) in "
                "its ancestry ran under a tainted seal."
            )
        try:
            declared = registry.preregister(
                fingerprint, selection_rule=rule, seal_digest=current_seal().digest
            )
        except PreregistrationError as exc:
            raise typer.BadParameter(str(exc)) from exc

    console.print(
        f"[green]pre-registered[/green] {strategy.name} [{fingerprint}]\n"
        f"family   {strategy.family}\n"
        f"declared {declared.declared_at}\n"
        f"rule     {declared.selection_rule}\n"
        f"seal     {declared.seal_digest[:16]}"
    )
    # The two sealed runs read different windows, charge different costs and are
    # counted as different denominators, so naming the wrong one here would send
    # somebody to a command that refuses -- or, worse, would have read as the
    # right one before ``option-run`` existed.
    command = "option-run" if strategy.family == OPTION else "run"
    console.print(
        "\nNothing has been read yet. Spend the seal with:\n"
        f"  python -m aqr.cli_sealed {command} {fingerprint}"
    )


@app.command("preregistered")
def preregistered(
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
) -> None:
    """Every declared candidate, and whether its seal has been spent."""
    with Registry(db) as registry:
        rows = registry.preregistrations()
        if not rows:
            console.print("[yellow]nothing pre-registered[/yellow]")
            return
        table = Table(
            "fingerprint", "strategy", "family", "declared", "sealed run", "look", "rule"
        )
        for row in rows:
            strategy = registry.get_strategy(row.fingerprint)
            spent = registry.sealed_run(row.fingerprint)
            look = registry.sealed_look(row.fingerprint)
            table.add_row(
                row.fingerprint,
                strategy.name if strategy else "?",
                strategy.family if strategy else "?",
                row.declared_at[:19],
                spent["sealed_run_at"][:19] if spent else "[green]unspent[/green]",
                str(look) if look else "-",
                row.selection_rule[:50],
            )
        console.print(table)
        # The count is the multiple-comparisons denominator for the sealed
        # window, the way `distinct_hypotheses` is for the search. Printed here
        # because this is the table somebody reads before believing a result.
        #
        # Per window, and the two windows are different data: the equity one is
        # S&P 500 bars past the embargo, the option one is SPY chains past it. A
        # candidate screened against one has not consumed a look at the other,
        # and charging it as though it had would raise a bar for a reading that
        # never touched the data.
        for name in (EQUITY, OPTION):
            total = registry.sealed_looks(family=name)
            if total:
                console.print(
                    f"\nthe sealed {name} window has screened {total} candidate(s); "
                    f"the bar for an alpha at that count is "
                    f"t >= {multiplicity_bar(total):.2f}"
                )


@app.command("seal-check")
def seal_check(
    root: str = typer.Option("", help="Cache root to audit. Default: both known roots."),
) -> None:
    """Prove, from the files on disk, that the research cache holds no embargoed bars.

    The seal detects a peek at run time; this checks the claim the seal rests on
    -- that the rows are not there to be read. It parses timestamps with the csv
    module and never builds a `Bars`, so auditing is not itself the peek it is
    looking for.

    The canary is reported rather than counted as a violation: it is *supposed*
    to sit in the research root holding embargoed rows, because a tripwire
    placed where a peek cannot happen catches nothing. It is reported per
    timeframe, because one armed timeframe must not read as cover for the
    others.
    """
    roots = (
        [Path(root)]
        if root
        else [SP500_RESEARCH_ROOT, SP500_SEALED_ROOT]
    )
    table = Table(box=None, pad_edge=False)
    for column in ("root", "files", "latest bar", "past embargo", "canary"):
        table.add_column(column)

    for path in roots:
        if not path.exists():
            console.print(f"[yellow]{path} does not exist[/yellow]")
            continue
        report = audit_cache_root(path)
        # Named for what it is, not for what it is called: a root is allowed
        # to hold embargoed rows exactly when "sealed" is in its name. Keying
        # on "research" instead marked every unprefixed research root red.
        expected_clean = "sealed" not in path.name
        colour = "green" if report.clean == expected_clean else "red"
        latest = report.latest.date().isoformat() if report.latest else "-"
        offenders = str(len(report.offenders)) if report.offenders else "none"
        canary = (
            f"armed ({', '.join(report.canary_timeframes)})"
            if report.canary_present
            else "-"
        )
        table.add_row(
            f"[{colour}]{path.name}[/{colour}]",
            str(report.files),
            latest,
            offenders,
            canary,
        )
    console.print(table)
    console.print()
    console.print(f"embargo starts {EMBARGO_START.date().isoformat()}")
    console.print(
        "the sealed root is [i]expected[/i] to hold embargoed bars -- it is the only "
        "thing that may, and only in a sealed run"
    )


def _explain(bars: Bars, report: object) -> list[str]:
    """The specific dates behind a report's complaints.

    A count is enough to notice a problem and never enough to resolve one: "1
    overnight move over 40%" is a real earnings gap or an unadjusted split, and
    the only way to tell is to look at the date.
    """
    import numpy as np

    lines: list[str] = []
    for left, right, days in getattr(report, "gaps", []):
        lines.append(f"gap: {left.date()} -> {right.date()} ({days} calendar days)")
    close = np.asarray(bars.close, dtype=float)
    if close.size > 1:
        moves = (close[1:] - close[:-1]) / np.where(close[:-1] > 0, close[:-1], np.nan)
        for i in np.flatnonzero(np.abs(moves) > 0.4):
            lines.append(
                f"move: {bars.timestamps[i].date()} {close[i]:.2f} -> "
                f"{bars.timestamps[i + 1].date()} {close[i + 1]:.2f}  {moves[i]:+.1%}"
            )
    return lines


@app.command()
def validate(
    strategy: Path = typer.Argument(..., help="Path to a strategy YAML file."),
    symbols: str = typer.Option("", help=SYMBOLS_HELP),
    source: str = typer.Option("synthetic", help="synthetic | yahoo | csv"),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT),
) -> None:
    """Parse a strategy and check it against data without evaluating it."""
    spec = load_file(strategy)
    data, _, _membership, _version = _load(
        source, _symbols_for(spec, symbols), start, end, timeframe, csv_root
    )
    primary = next(iter(spec.universe.symbols))
    if primary not in data:
        raise typer.BadParameter(f"{primary} is in the universe but was not loaded")

    console.print(f"[bold]{spec.name}[/bold]  fingerprint {spec.fingerprint()}")
    console.print(f"  parameters: {spec.parameter_count()}")
    console.print(f"  features:   {', '.join(sorted(str(k) for k in spec.features()))}")
    report = validate_against(spec, data[primary])
    style = "green" if report.ok else "red"
    console.print(f"[{style}]{report}[/{style}]")
    raise typer.Exit(0 if report.ok else 1)


@app.command()
def backtest(
    strategy: Path = typer.Argument(...),
    symbols: str = typer.Option("", help=SYMBOLS_HELP),
    source: str = typer.Option("synthetic", help=f"One of: {', '.join(SOURCES)}."),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT),
    universe: str = typer.Option(
        "",
        help=(
            "Point-in-time universe whose membership gates what may be held on "
            f"each session: {', '.join(universe_names())}. The symbols traded "
            "still come from the strategy file -- this restricts them by date, "
            "it does not add to them."
        ),
    ),
    equity: float = typer.Option(100_000.0),
    spread_bps: float = typer.Option(2.0),
    slippage_bps: float = typer.Option(1.0),
    commission: float = typer.Option(0.005, help="Per share."),
    cost_schedule: str = typer.Option("", "--costs", help=COSTS_HELP),
    show_trades: int = typer.Option(0, help="Print the first N trades."),
) -> None:
    """Run one in-sample backtest. In-sample results prove nothing on their own."""
    spec = load_file(strategy)
    # Built before the data is loaded: a mistyped schedule should be refused in
    # the first second, not after six hundred symbols have been read.
    config = BacktestConfig(
        initial_equity=equity,
        costs=_costs(spread_bps, slippage_bps, commission, cost_schedule),
    )
    data, _, membership, _version = _load(
        source, _symbols_for(spec, symbols), start, end, timeframe, csv_root,
        universe=universe,
    )
    result = run_strategy(spec, data, config, membership=membership)
    metrics = compute_metrics(result)

    console.print(f"[bold]{spec.name}[/bold] on {', '.join(result.symbols)}")
    console.print(f"  equity {result.initial_equity:,.0f} -> {result.final_equity:,.0f}")
    console.print(f"  {metrics}")
    console.print(f"  fees {metrics.total_fees:,.0f}  slippage {metrics.total_slippage:,.0f}")
    if result.halted:
        console.print(f"[red]  HALTED: {result.halt_reason}[/red]")
    console.print(
        "[yellow]  in-sample only — run 'aqr evaluate' before believing any of this[/yellow]"
    )

    if show_trades:
        table = Table("symbol", "entry", "exit", "qty", "net P&L", "reason", "bars")
        for trade in result.trades[:show_trades]:
            table.add_row(
                trade.symbol,
                trade.entry_dt.date().isoformat(),
                trade.exit_dt.date().isoformat(),
                f"{trade.quantity:.0f}",
                f"{trade.net_pnl:,.0f}",
                trade.exit_reason,
                str(trade.bars_held),
            )
        console.print(table)


@app.command()
def walkforward(
    strategy: Path = typer.Argument(...),
    symbols: str = typer.Option("", help=SYMBOLS_HELP),
    source: str = typer.Option("synthetic", help=f"One of: {', '.join(SOURCES)}."),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT),
    train_bars: int = typer.Option(
        TRAIN_BARS, help="Bars per training window. See splits.TRAIN_BARS for the geometry."
    ),
    test_bars: int = typer.Option(TEST_BARS, help="Bars per out-of-sample window."),
    anchored: bool = typer.Option(True, help="Growing train windows rather than sliding."),
) -> None:
    """Walk a strategy forward and report out-of-sample performance."""
    spec = load_file(strategy)
    data, _, _membership, _version = _load(
        source, _symbols_for(spec, symbols), start, end, timeframe, csv_root
    )
    report = run_walk_forward(
        spec, data, train_bars=train_bars, test_bars=test_bars, anchored=anchored
    )
    if not report.folds:
        console.print("[red]not enough history for a single fold[/red]")
        raise typer.Exit(1)
    # str() explicitly: rich pretty-prints a dataclass by its repr, which would
    # dump the fold objects instead of the report's own summary.
    console.print(str(report))
    if report.stitched:
        console.print(f"\n  stitched out-of-sample: {report.stitched}")


@app.command()
def evaluate(
    strategy: Path = typer.Argument(...),
    symbols: str = typer.Option("", help=SYMBOLS_HELP),
    source: str = typer.Option("synthetic", help=f"One of: {', '.join(SOURCES)}."),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT),
    universe: str = typer.Option(
        "",
        help=(
            "Point-in-time universe whose membership gates what may be held on "
            f"each session: {', '.join(universe_names())}. The symbols traded "
            "still come from the strategy file -- this restricts them by date, "
            "it does not add to them."
        ),
    ),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
    train_bars: int = typer.Option(TRAIN_BARS),
    test_bars: int = typer.Option(TEST_BARS),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """The full gauntlet: walk-forward, robustness, overfitting, score, record."""
    spec = load_file(strategy)
    data, labels, membership, _version = _load(
        source, _symbols_for(spec, symbols), start, end, timeframe, csv_root,
        universe=universe,
    )
    with Registry(db) as registry:
        outcome = evaluate_candidate(
            spec,
            data,
            regime_labels=labels,
            membership=membership,
            registry=registry,
            train_bars=train_bars,
            test_bars=test_bars,
            dataset_version=_version,
        )
    if as_json:
        console.print_json(json.dumps(outcome.as_dict(), default=str))
    else:
        console.print(outcome.summary())
        for warning in outcome.warnings:
            console.print(f"[yellow]warning: {warning}[/yellow]")
    raise typer.Exit(0 if outcome.verdict in ("ACCEPT", "PAPER") else 1)


@app.command()
def research(
    iterations: int = typer.Option(8, help="Hypotheses to test."),
    # Unlike the other commands there is no strategy file to take a universe
    # from, so this one carries the default itself.
    universe: str = typer.Option(
        "", help=f"Named universe: {', '.join(universe_names())}. Combines with --symbols."
    ),
    symbols: str = typer.Option(DEFAULT_SYMBOLS, help="Comma-separated universe to research."),
    universe_limit: int = typer.Option(0, help="Take only the largest N of a named universe."),
    min_bars: int = typer.Option(
        1500, help="Drop symbols with less history than this. Prevents half-empty backtests."
    ),
    source: str = typer.Option("synthetic", help=f"One of: {', '.join(SOURCES)}."),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    timeframes: str = typer.Option(
        "",
        help=(
            "Comma-separated granularities the proposer may choose from (1D, 1h, 4h). "
            "Overrides --timeframe when set; the first entry is the default a "
            "proposal falls back to."
        ),
    ),
    csv_root: str = typer.Option(DEFAULT_CSV_ROOT),
    db: Path = typer.Option(Path("runs/research.sqlite")),
    provider: str = typer.Option(
        "offline", help="Who proposes: offline | deepseek | anthropic."
    ),
    model: str = typer.Option("", help="Model id. Defaults to the provider's usual one."),
    save_to: str = typer.Option("strategies", help="Where surviving strategies are written."),
) -> None:
    """Run the research loop: propose, test, record, repeat."""
    allowed = tuple(t.strip() for t in timeframes.split(",") if t.strip()) or (timeframe,)
    unknown = [t for t in allowed if t not in ("1D", "1h", "4h")]
    if unknown:
        raise typer.BadParameter(f"unknown timeframe(s): {', '.join(unknown)}; use 1D, 1h, 4h")
    requested = _universe(universe, "" if universe else symbols, universe_limit)

    # Every allowed granularity gets its own load: a candidate is evaluated on
    # the bar size it asked for, and only that one.
    data_by_timeframe: dict[str, dict[str, Bars]] = {}
    labels_by_timeframe: dict[str, dict[str, list[str]]] = {}
    membership: PointInTimeUniverse | None = None
    version = ""
    for tf in allowed:
        data, labels, membership, tf_version = _load(
            source, requested, start, end, tf, csv_root,
            universe=universe, tolerant=True,
        )
        # A symbol that IPO'd in 2021 cannot take part in a 2010 walk-forward.
        # Drop it here and say so, rather than letting it contribute empty
        # folds. min_bars is denominated in daily bars; scale it to this bar
        # size or an hourly run would demand 1500 years of history.
        scaled_min = round(min_bars * bars_per_year(tf) / 252.0)
        short_history = {s: len(b) for s, b in data.items() if len(b) < scaled_min}
        if short_history:
            console.print(
                f"[yellow]dropped {len(short_history)} symbol(s) with under "
                f"{scaled_min} {tf} bars: "
                + ", ".join(f"{s} ({n})" for s, n in sorted(short_history.items()))
                + "[/yellow]"
            )
            data = {s: b for s, b in data.items() if s not in short_history}
            if labels:
                labels = {s: v for s, v in labels.items() if s not in short_history}
        if not data:
            raise typer.BadParameter(
                f"every requested symbol was dropped for lack of {tf} history"
            )
        data_by_timeframe[tf] = data
        labels_by_timeframe[tf] = labels or {}
        if tf == allowed[0]:
            version = tf_version
    # A name missing from one granularity's cache would trade on a different
    # universe per timeframe; the intersection is the honest universe.
    names = sorted(set.intersection(*map(set, data_by_timeframe.values())))
    default_tf = allowed[0]
    console.print(f"researching {len(names)} symbols on {'/'.join(allowed)} bars from {source}")
    proposer = _proposer(provider, model)

    with Registry(db) as registry:
        loop = ResearchLoop(
            data=data_by_timeframe[default_tf],
            registry=registry,
            config=ResearchConfig(
                symbols=names,
                timeframe=default_tf,
                timeframes=allowed,
                iterations=iterations,
                dataset_version=version,
                save_accepted_to=save_to,
            ),
            proposer=proposer,
            regime_labels=labels_by_timeframe[default_tf],
            data_by_timeframe=data_by_timeframe,
            regime_labels_by_timeframe=labels_by_timeframe,
            membership=membership,
        )
        for step in loop.run():
            colour = {"ACCEPT": "green", "PAPER": "cyan", "REVIEW": "yellow"}.get(
                step.verdict, "red"
            )
            console.print(f"[{colour}]{step}[/{colour}]")
        console.print()
        best = loop.best()
        if best and best.outcome:
            console.print(best.outcome.summary())
        console.print(f"\ncumulative backtests in this database: {registry.total_backtests()}")


@app.command()
def holdout(
    status: str = typer.Option("PAPER", help="Which registry state to re-test."),
    symbols: str = typer.Option("", help="Held-out symbols. Default: the rest of --universe."),
    universe: str = typer.Option("nasdaq50", help="Universe to take held-out symbols from."),
    skip: int = typer.Option(
        20, help="How many of the universe the research already used. Those are excluded."
    ),
    source: str = typer.Option("csv", help=f"One of: {', '.join(SOURCES)}."),
    start: str = typer.Option(DEFAULT_START),
    end: str = typer.Option(DEFAULT_END),
    timeframe: str = typer.Option("1D"),
    csv_root: str = typer.Option("data-yahoo"),
    min_bars: int = typer.Option(1500, help="Drop held-out symbols with less history."),
    db: Path = typer.Option(Path("runs/research.sqlite")),
    limit: int = typer.Option(20, help="How many strategies to re-test."),
) -> None:
    """Re-test promoted strategies on symbols the search never saw.

    Every other number this project reports is measured on the data that chose
    the strategy. Walk-forward helps, but the universe and the window were fixed
    before the first hypothesis and every hypothesis since has been tried against
    them; a best-of-N result on one universe is a claim about that universe.

    This is the one test the search cannot have leaked into. A strategy that
    survives it is not proven -- one holdout is one sample -- but a strategy that
    collapses has said something definite, and that is the more common result.
    """
    used = set(resolve_universe(universe, skip)) if skip else set()
    if symbols:
        held = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        held = [s for s in resolve_universe(universe) if s not in used]
    if not held:
        raise typer.BadParameter("no held-out symbols left after --skip")

    leaked = sorted(set(held) & used)
    if leaked:
        raise typer.BadParameter(
            f"these symbols were used by the research and are not held out: {leaked}"
        )

    data, _labels, _membership, version = _load(
        source, held, start, end, timeframe, csv_root, tolerant=True
    )
    short = {s: len(b) for s, b in data.items() if len(b) < min_bars}
    if short:
        console.print(
            f"[yellow]dropped {len(short)} with under {min_bars} bars: "
            + ", ".join(f"{s} ({n})" for s, n in sorted(short.items()))
            + "[/yellow]"
        )
        data = {s: b for s, b in data.items() if s not in short}
    if not data:
        raise typer.BadParameter("every held-out symbol was dropped for lack of history")

    console.print(
        f"holding out {len(data)} symbols the research never saw: {', '.join(sorted(data))}"
    )
    console.print(f"dataset: {version}\n")

    survived = 0
    beat = 0
    tested = 0
    with Registry(db) as registry:
        records = registry.strategies(status=status, limit=limit)  # type: ignore[arg-type]
        if not records:
            raise typer.BadParameter(f"no strategies in state {status!r} in {db}")
        for record in records:
            spec = loads_spec(record.spec_yaml)
            selected = _selected_sharpe(registry, record.fingerprint)
            result = run_holdout(spec, data, selected_sharpe=selected)
            tested += 1
            kept = result.retained
            colour = "green" if kept is not None and kept >= 0.5 else "yellow"
            if not result.traded or result.metrics is not None and result.metrics.sharpe <= 0:
                colour = "red"
            else:
                survived += 1
            if result.beat_benchmark:
                beat += 1
                colour = "green"
            console.print(f"[{colour}]{result}[/{colour}]")

    console.print()
    console.print(
        f"{survived} of {tested} kept a positive Sharpe on symbols they were never "
        f"selected on -- but only [bold]{beat} of {tested}[/bold] beat buy and hold."
    )
    if beat < tested:
        # The headline number on its own is misleading, and misleading in the
        # flattering direction. A long-only rule on these names over this window
        # makes money almost whatever the rule is.
        console.print(
            "[yellow]A positive Sharpe here is not evidence of an edge: holding the "
            "same symbols over the same window was available to anyone, for free, "
            "with no rule at all.[/yellow]"
        )


def _selected_sharpe(registry: Registry, fingerprint: str) -> float | None:
    """The out-of-sample Sharpe recorded when this strategy was promoted."""
    for row in registry.experiments(limit=1, fingerprint=fingerprint):
        raw = row.get("oos_metrics")
        if raw:
            return float(json.loads(raw).get("sharpe", 0.0))
    return None


@app.command("registry")
def registry_cmd(
    db: Path = typer.Option(Path("runs/research.sqlite")),
    status: str = typer.Option("", help="Filter: CANDIDATE, PAPER, LIVE, REJECTED, ..."),
    family: str = typer.Option(
        "", help=f"Filter to one search program: {EQUITY} or {OPTION}."
    ),
    limit: int = typer.Option(20),
) -> None:
    """List known strategies and their lifecycle state.

    The family column is not decoration. An equity fingerprint and an option
    fingerprint are handed off by different commands to different executors, and
    the two spec formats are not interchangeable -- reading one with the other's
    loader produces a rule with no entry condition rather than an error.
    """
    with Registry(db) as reg:
        rows = reg.strategies(status or None, limit, family=family or None)  # type: ignore[arg-type]
        counts = {name: reg.distinct_hypotheses(family=name) for name in (EQUITY, OPTION)}
    if not rows:
        console.print("no strategies recorded yet")
        return
    table = Table("fingerprint", "name", "family", "status", "score", "updated")
    for row in rows:
        table.add_row(
            row.fingerprint,
            row.name,
            row.family,
            row.status,
            f"{row.score:.0f}" if row.score is not None else "-",
            row.updated_at[:19],
        )
    console.print(table)
    console.print(
        f"\ndistinct hypotheses: {counts[EQUITY]} equity, {counts[OPTION]} option "
        "(counted separately -- specs/10 D8: one multiplicity denominator across "
        "both searches would make both bars wrong)"
    )


@app.command()
def experiments(
    db: Path = typer.Option(Path("runs/research.sqlite")),
    family: str = typer.Option(
        "", help=f"Filter to one search program: {EQUITY} or {OPTION}."
    ),
    limit: int = typer.Option(20),
) -> None:
    """The research log, most recent first. Failures included.

    ``cycles`` is filled for an option experiment and empty for an equity one,
    because it is the number the option evaluator gates on: a structure held to
    expiry produces evidence when it closes, so thirty overlapping spreads can
    be eight independent bets (specs/10 D8).
    """
    with Registry(db) as reg:
        rows = reg.memory(limit, family=family or None)
        totals = {name: reg.total_backtests(family=name) for name in (EQUITY, OPTION)}
        distinct = {name: reg.distinct_hypotheses(family=name) for name in (EQUITY, OPTION)}
    if not rows:
        console.print("no experiments recorded yet")
        return
    table = Table(
        "strategy", "family", "verdict", "score", "OOS Sharpe", "trades", "cycles", "note"
    )
    for row in rows:
        sharpe = row.get("oos_sharpe")
        cycles = row.get("oos_cycles")
        table.add_row(
            str(row.get("name")),
            str(row.get("family")),
            str(row.get("verdict")),
            f"{row['score']:.0f}" if row.get("score") is not None else "-",
            f"{sharpe:.2f}" if sharpe is not None else "-",
            str(row.get("oos_trades") or "-"),
            str(cycles) if cycles is not None else "-",
            (row.get("error") or row.get("overfitting") or "")[:60],
        )
    console.print(table)
    console.print(
        f"\ncumulative backtests: {totals[EQUITY]} equity, {totals[OPTION]} option"
    )
    console.print(
        f"multiple-comparisons denominators: {distinct[EQUITY]} equity hypotheses, "
        f"{distinct[OPTION]} option hypotheses of {OPTION_SEARCH_BUDGET} "
        "(never summed -- specs/10 D8)"
    )


@app.command()
def promote(
    fingerprint: str = typer.Argument(..., help="Strategy fingerprint from 'aqr registry'."),
    status: str = typer.Argument(..., help="PAPER, LIVE, DEGRADED, RETIRED, REJECTED"),
    reason: str = typer.Option("", help="Why. Recorded in the status log."),
    db: Path = typer.Option(Path("runs/research.sqlite")),
) -> None:
    """Move a strategy through its lifecycle. Illegal transitions are refused."""
    with Registry(db) as reg:
        try:
            reg.set_status(fingerprint, status.upper(), reason)  # type: ignore[arg-type]
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        record = reg.get_strategy(fingerprint)
    assert record is not None
    console.print(f"[green]{record.name} is now {record.status}[/green]")


@app.command("costs")
def costs_cmd(
    equity: float = typer.Option(100_000.0, help="Account size the book is run at."),
    positions: int = typer.Option(
        110,
        help="Names held, weighted evenly. A 10-name core plus a 100-name sleeve "
        "is 110 — but that book is not even, and its smallest positions cost far "
        "more per order than this average. Pass the count that gives the position "
        "size you actually want priced.",
    ),
    price: float = typer.Option(100.0, help="Representative share price."),
    rebalances_per_year: int = typer.Option(
        50, help="Rebalances a year. Every 5 sessions is about 50."
    ),
) -> None:
    """What an order actually costs, under each named schedule.

    The question the fatal cost gate turns on and that nothing could answer
    before. Two of the charges scale with notional and two do not, so the same
    schedule prices the same strategy differently depending on how many names it
    holds and how large the account is — and ``initial_equity``, which nobody
    thinks of as a cost parameter, is the one that moves it most.

    Run this before trusting a cost-sensitivity verdict on a wide book.
    """
    notional = equity / positions if positions > 0 else 0.0
    console.print(
        f"{positions} positions at ${equity:,.0f} → ${notional:,.0f} per position, "
        f"~{rebalances_per_year} rebalances/yr"
    )
    console.print()

    table = Table("schedule", "fee", "spread+slip", "all-in", "floor binds", "drag/yr")
    for name in sorted(PRESETS):
        order = PRESETS[name].price_order(notional, price)
        # Two orders per round trip, and a rebalance turning over the whole book
        # is the pessimistic reading. Annualised so it can be compared against a
        # return rather than against another basis-point number.
        drag = order.bps / 10_000.0 * 2 * rebalances_per_year
        table.add_row(
            name,
            f"${order.fee:,.2f} ({order.fee_bps:.1f}bp)",
            f"${order.adverse:,.2f}",
            f"{order.bps:.1f}bp",
            "[red]yes[/red]" if order.floor_binds else "-",
            f"{drag:.1%}",
        )
    console.print(table)
    console.print()
    console.print(
        "[dim]drag/yr assumes the whole book turns over at every rebalance, which "
        "is the pessimistic reading. The default schedule is ibkr_fixed; it is "
        "unchanged so that recorded verdicts stay reproducible.[/dim]"
    )


@app.command("target-book")
def target_book_cmd(
    strategy: str = typer.Argument(
        ...,
        help="A registered fingerprint, or a path to the spec file. A fingerprint "
        "is preferred: a path is editable between the sealed run and the handoff "
        "and a fingerprint is not.",
    ),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
    source: str = typer.Option("csv", help=f"One of {', '.join(SOURCES)}."),
    csv_root: str = typer.Option(
        SEALED_SP500_ROOT,
        help="Where the bars are. The sealed cache by default -- it is the only "
        "root that holds sessions up to the present.",
    ),
    universe: str = typer.Option("sp500_pit", help="Point-in-time universe to gate by."),
    timeframe: str = typer.Option("1D"),
    start: str = typer.Option(
        "", help=f"First bar. Default: {TARGET_BOOK_WARMUP_DAYS} days before the end."
    ),
    end: str = typer.Option("", help="Last session to consider. Default: today."),
    out: Path = typer.Option(Path("runs/target_books"), help="Where to write the book."),
) -> None:
    """Write the target book for a validated strategy. It is not placed anywhere.

    There is no ``--dry-run`` because there is nothing to be dry about: no code
    path in this project sends an order to anything. The output is a file, and
    executing it is a different system's job -- one that still needs the
    equity-shaped risk gate, the reconciliation and the kill switch that the
    artefact lists under ``consumer_must_supply``.

    The refusals come first and they are the point. A book is written only for a
    strategy the registry knows, whose seal has been spent, and whose sealed run
    did not refute it. Producing one for an undeclared candidate would read the
    embargoed years for a rule that still has an unspent seal, which is the
    loophole the whole pre-registration protocol exists to close. And because the
    sealed run measured the spec's own timeframe, a ``--timeframe`` that diverges
    from it is refused too: weights built on other bars are not what was
    validated, whatever the artefact would have claimed.
    """
    with Registry(db) as reg:
        spec, record = _spec_for_handoff(reg, strategy)
        fingerprint = spec.fingerprint()

        if timeframe != spec.universe.timeframe:
            # The option picks which cache folder is loaded; the book is stamped
            # with the spec's own timeframe. A divergent pair would stamp "1D"
            # on weights built from other bars and still carry the sealed
            # verdict, which was measured on the spec's timeframe only.
            raise typer.BadParameter(
                f"--timeframe {timeframe} but {record.name} [{fingerprint}] declares "
                f"{spec.universe.timeframe}: the sealed run measured the spec's own "
                "timeframe, so weights built on other bars are not what the seal "
                "was spent on."
            )

        sealed = reg.sealed_run(fingerprint)
        if sealed is None:
            raise typer.BadParameter(
                f"{record.name} [{fingerprint}] has no sealed run. A book handed off "
                "before the seal is spent would read the embargoed years for a rule "
                "whose out-of-sample verdict is still owed. Declare it with "
                "`aqr preregister` and spend it with `python -m aqr.cli_sealed run`."
            )
        measurement = dict(sealed["result"].get("measurement", {}))
        if measurement.get("refuted"):
            raise typer.BadParameter(
                f"{record.name} was refuted by its sealed run on "
                f"{sealed['sealed_run_at'][:19]}: negative alpha, significant in the "
                "wrong direction. There is nothing to hand off."
            )

        declaration = reg.preregistration(fingerprint)
        provenance: dict[str, object] = {
            "database": str(db),
            "status": record.status,
            "score": record.score,
            "hypothesis": record.hypothesis,
            # Two denominators, and they discount different things. The first
            # counts the hypotheses the *search* compared; the second counts the
            # candidates the *sealed window* has screened. A book that carried
            # only the first would understate what its claim rests on.
            "distinct_hypotheses": reg.distinct_hypotheses(),
            "sealed_look": reg.sealed_look(fingerprint),
            "sealed_looks_total": reg.sealed_looks(),
            "sealed_run_at": sealed["sealed_run_at"],
            "sealed_measurement": measurement,
            "sealed_seal": sealed["result"].get("seal", {}),
            "preregistration": (
                {
                    "declared_at": declaration.declared_at,
                    "selection_rule": declaration.selection_rule,
                    "seal_digest": declaration.seal_digest,
                }
                if declaration
                else None
            ),
        }

        last = _parse_date(end) if end else datetime.now(UTC)
        first = (
            _parse_date(start)
            if start
            else datetime.fromtimestamp(
                last.timestamp() - TARGET_BOOK_WARMUP_DAYS * 86_400, tz=UTC
            )
        )
        data, _, membership, dataset_version = _load(
            source,
            list(spec.universe.symbols),
            first.date().isoformat(),
            last.date().isoformat(),
            timeframe,
            csv_root,
            universe=universe,
            tolerant=True,
        )
        loaded_last = max(bars.timestamps[-1].date() for bars in data.values())
        console.print(
            f"{len(data)} of {len(spec.universe.symbols)} symbols, "
            f"{first.date().isoformat()} -> {last.date().isoformat()} requested; "
            f"the data reaches {loaded_last.isoformat()}"
        )
        if loaded_last < last.date():
            # as_of comes from the data, not the request -- name the session the
            # book will actually carry rather than letting the requested window
            # stand in for it.
            console.print(
                f"[yellow]the book will be as of {loaded_last.isoformat()}, the last "
                "session the data holds, not the end that was requested[/yellow]"
            )

        try:
            book = build_target_book(
                spec,
                data,
                generated_at=datetime.now(UTC),
                dataset_version=dataset_version,
                universe=universe,
                provenance=provenance,
                seal=current_seal().certificate(),
                config=BacktestConfig(),
                membership=membership,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        path = out / f"{spec.name}-{fingerprint}-{book.as_of}.json"
        digest = write_book(book, path)
        reg.record_target_book(
            fingerprint,
            as_of=book.as_of,
            path=str(path),
            digest=digest,
            book=book.as_dict(),
        )

    console.print()
    console.print(book.summary())
    console.print()
    console.print(f"[green]{path}[/green]")
    console.print(f"[dim]sha256 {digest[:16]}  recorded against {fingerprint}[/dim]")
    console.print(
        "[dim]weights only. Sizing, reconciliation, caps, an equity risk gate, a "
        "kill switch and a fill journal belong to whatever executes this.[/dim]"
    )


def _spec_for_handoff(reg: Registry, strategy: str) -> tuple[StrategySpec, StrategyRecord]:
    """Resolve the argument to a spec *and* the registry row that justifies it.

    A path is accepted for convenience and then held to the same standard: its
    fingerprint has to be one the registry already knows, or there is no
    hypothesis, no campaign and no sealed run to trace the book back to.
    """
    path = Path(strategy)
    spec = load_file(path) if path.exists() else None
    fingerprint = spec.fingerprint() if spec is not None else strategy

    record = reg.get_strategy(fingerprint)
    if record is None:
        raise typer.BadParameter(
            f"no strategy {fingerprint!r} in the registry. A target book that cannot "
            "be traced back to a recorded hypothesis is a set of weights from "
            "nowhere; run `aqr evaluate` on the spec first."
        )
    return spec or loads_spec(record.spec_yaml), record


@app.command("target-books")
def target_books_cmd(
    fingerprint: str = typer.Argument("", help="Filter to one strategy."),
    db: Path = typer.Option(Path("runs/research.sqlite")),
    limit: int = typer.Option(20),
) -> None:
    """Every book handed off, newest first."""
    with Registry(db) as reg:
        rows = reg.target_books(fingerprint or None, limit)
    if not rows:
        console.print("no target books recorded yet")
        return
    table = Table("as of", "strategy", "fingerprint", "positions", "gross", "sha256", "path")
    for row in rows:
        book = row["book"]
        table.add_row(
            row["as_of"],
            str(book.get("spec_name", "?")),
            row["fingerprint"],
            str(book.get("positions", "-")),
            f"{float(book.get('gross', 0.0)):.4f}",
            row["digest"][:16],
            row["path"],
        )
    console.print(table)


@app.command("options-pull")
def options_pull(
    universe_file: str = typer.Option(
        "data-universes/sp500_pit.json",
        help="Point-in-time universe JSON. Every ticker ever a member, which is "
        "the point: a pull driven by today's members is missing exactly the "
        "names whose absence created the survivorship bias.",
    ),
    symbols: str = typer.Option(
        "SPY,QQQ,IWM",
        help="Extra symbols to keep, on top of the universe file. Defaults to the "
        "liquid index ETFs, which a membership file cannot contain and which are "
        "the densest option chains listed.",
    ),
    root: str = typer.Option(DEFAULT_OPTIONS_ROOT, help="Cache root."),
    table: str = typer.Option(
        "both", help="option_chain, volatility_history, or both."
    ),
    skip_download: bool = typer.Option(
        False,
        help="Split a vendor file already on disk. The download is the hour; the "
        "split is a minute, so a change to the splitting logic must not cost a "
        "second transfer.",
    ),
) -> None:
    """Cache free end-of-day option chains from DoltHub, filtered to the universe.

    Two phases, because the CSV endpoint reports no size, refuses HEAD and
    honours no Range header -- all three confirmed against the live service, so
    a transfer that dies at minute fifty cannot be resumed. Phase one only moves
    bytes, into ``<root>/_raw/<table>.csv.gz``. Phase two splits that local file
    into ``<root>/<table>/<symbol>.csv`` and may be re-run freely.

    This is a network command, like ``pull``, and for the same reason: every
    later run reads the cache, so a research result is reproducible rather than
    dependent on what the vendor is serving today.

    The data is end-of-day only. There is no 1h or 4h option data here, and none
    is available free anywhere -- ``aqr`` researches options on daily bars or
    not at all.
    """
    extra = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    wanted = _symbols_from_universe_file(universe_file, extra)
    if not wanted:
        raise typer.BadParameter("the universe is empty; nothing would be kept")

    tables: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("option_chain", CHAIN_COLUMNS),
        ("volatility_history", VOLATILITY_COLUMNS),
    )
    if table != "both":
        tables = tuple(t for t in tables if t[0] == table)
        if not tables:
            raise typer.BadParameter(
                f"unknown table {table!r}: option_chain, volatility_history or both"
            )

    base = Path(root)
    console.print(f"universe: [bold]{len(wanted)}[/bold] symbols from {universe_file}")

    for name, required in tables:
        raw = base / "_raw" / f"{name}.csv.gz"
        console.print("")
        console.print(f"[bold]{name}[/bold]")

        if skip_download:
            if not raw.exists():
                raise typer.BadParameter(f"--skip-download but {raw} is not there")
            console.print(f"  reusing {raw} ({raw.stat().st_size / 1e6:.0f} MB on disk)")
        else:
            console.print(f"  downloading -> {raw}  (no size is reported; this is the slow half)")

            def progress(read: int, seconds: float) -> None:
                console.print(
                    f"    {read / 1e9:6.2f} GB in {seconds / 60:5.1f} min"
                    f"  ({read / 1e6 / seconds:.1f} MB/s)"
                )

            got = download_table(name, raw, open_stream=_csv_stream, on_progress=progress)
            console.print(
                f"  downloaded {got.bytes_read / 1e9:.2f} GB in {got.seconds / 60:.1f} min"
                f"  ({got.rate_mb_s:.1f} MB/s)"
            )

        out = base / name
        console.print(f"  splitting -> {out}/<symbol>.csv")
        result = split_table(raw, out, symbols=wanted, required_columns=required)
        console.print(
            f"  kept [bold]{result.rows_kept:,}[/bold] of {result.rows_read:,} rows"
            f"  across [bold]{len(result.symbols)}[/bold] symbols"
        )
        console.print(f"  window {result.first_date} .. {result.last_date}")
        if result.malformed_rows:
            console.print(f"  [yellow]{result.malformed_rows} malformed rows skipped[/yellow]")
        skipped = sorted(result.unknown_symbols)
        console.print(
            f"  {len(skipped)} symbols in the file the universe did not name"
            + (f" (e.g. {', '.join(skipped[:6])})" if skipped else "")
        )
        missing = sorted(set(wanted) - result.symbols)
        console.print(
            f"  {len(missing)} universe symbols the vendor has no rows for"
            + (f" (e.g. {', '.join(missing[:6])})" if missing else "")
        )


def _csv_stream(url: str) -> Any:
    """The one place this command reaches a network. Streamed, never buffered.

    Returns the context manager `httpx.stream` gives back rather than a typed
    handle: what `download_table` needs from it is `iter_bytes`, which is the
    whole of its `ByteStream` Protocol, and naming httpx's own response type
    here would make the module import httpx to be type-checked.
    """
    import httpx

    return httpx.stream("GET", url, timeout=120.0, follow_redirects=True)


@app.command("options-embargo")
def options_embargo(
    symbols: str = typer.Option("SPY", help="Symbols to keep. Everything else is deleted."),
    root: str = typer.Option(DEFAULT_OPTIONS_ROOT, help="Research cache root."),
    sealed_root: str = typer.Option(
        DEFAULT_OPTIONS_SEALED_ROOT, help="Sealed cache root: the full history."
    ),
    prune: bool = typer.Option(
        True, help="Delete cached symbols outside --symbols. specs/07 D2 trades one name."
    ),
) -> None:
    """Split the option cache at the embargo and arm the tripwires.

    The research root is truncated on disk at ``EMBARGO_START``; the sealed root
    keeps everything. That physical separation is the lock that survives being
    called from somewhere nobody reviewed -- the rows are simply not there.

    Re-runnable. It reads the sealed root when one exists, so the split can be
    redone after a re-pull without another download.
    """
    keep = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not keep:
        raise typer.BadParameter("keep at least one symbol")

    research, sealed = Path(root), Path(sealed_root)
    console.print(f"embargo: [bold]{EMBARGO_START.date()}[/bold]  keeping {', '.join(keep)}")

    for table, _cols in (("option_chain", None), ("volatility_history", None)):
        console.print(f"[bold]{table}[/bold]")
        for symbol in keep:
            # Prefer the sealed copy: it is the untruncated one, so a re-run
            # after an earlier split does not truncate an already-truncated file.
            source = sealed / table / f"{symbol}.csv"
            if not source.exists():
                source = research / table / f"{symbol}.csv"
            if not source.exists():
                console.print(f"  [yellow]{symbol}: no cache, skipped[/yellow]")
                continue
            result = split_at_embargo(
                source,
                symbol=symbol,
                table=table,
                research_root=research,
                sealed_root=sealed,
            )
            console.print(
                f"  {symbol}: research {result.research_rows:,} rows to {result.research_last}"
                f"  |  sealed {result.sealed_rows:,} rows to {result.sealed_last}"
            )
        armed = write_option_canary(research, table)
        console.print(f"  canary armed: {armed}")

        if prune:
            removed = 0
            for path in sorted((research / table).glob("*.csv")):
                if path.stem in keep or path.stem == CANARY_SYMBOL:
                    continue
                path.unlink()
                removed += 1
            for path in sorted((sealed / table).glob("*.csv")):
                if path.stem in keep:
                    continue
                path.unlink()
                removed += 1
            if removed:
                console.print(f"  pruned {removed} files outside the universe")

    # The two roots are audited against opposite expectations, and saying so is
    # the point. Past-embargo rows in the research root are contamination; in
    # the sealed root they are the entire reason it exists, and reporting them
    # under the same word would teach a reader to ignore the word.
    for label, base, embargoed_rows_expected in (
        ("research", research, False),
        ("sealed", sealed, True),
    ):
        audit = audit_option_root(base)
        if embargoed_rows_expected:
            state = (
                f"holds the reserved window ({', '.join(audit.offenders)})"
                if audit.offenders
                else "[yellow]EMPTY of reserved rows — the sealed window is not there[/yellow]"
            )
        else:
            state = (
                "clean"
                if audit.clean
                else f"[red]CONTAMINATED: {', '.join(audit.offenders)}[/red]"
            )
        console.print("")
        console.print(f"{label:8} {base}")
        console.print(
            f"         {audit.files} files, {audit.rows:,} rows, latest {audit.latest}"
        )
        console.print(
            f"         {state}"
            + (f", canary in {', '.join(audit.canary_tables)}" if audit.canary_present else "")
        )


# --------------------------------------------------------------------------- #
# Options — specs/10-options-research.md
# --------------------------------------------------------------------------- #


def _option_proposer(provider: str, model: str) -> OptionProposer:
    """Build the option proposer for ``--provider``.

    Imported lazily and per-branch, like ``_proposer``: the offline loop never
    needs an LLM SDK installed, and a missing API key surfaces here rather than
    as an exception ten iterations into a campaign that has a budget of twenty.
    """
    from aqr.agent.option_proposer import (
        AnthropicOptionProposer,
        DeepSeekOptionProposer,
        TemplateOptionProposer,
    )

    choice = provider.strip().lower()
    if choice == "offline":
        return TemplateOptionProposer()
    if choice == "deepseek":
        return DeepSeekOptionProposer(model or "deepseek-chat")
    if choice == "anthropic":
        return AnthropicOptionProposer(model or "claude-opus-5")
    raise typer.BadParameter(
        f"unknown provider {provider!r}; use offline, deepseek or anthropic"
    )


@app.command("option-features")
def option_features() -> None:
    """The option research vocabulary: structures, then features.

    Separate from ``aqr features`` because the two vocabularies are separate
    (CLAUDE.md §2b). This one is the union an option entry expression parses
    against -- specs/10 D6's table *and* the unchanged bar registry -- so a rule
    may say ``iv_rank() > 50 and close > sma(200)`` in one expression.
    """
    table = Table("structure", "what it is", title=f"{len(STRUCTURE_CATALOGUE)} structures")
    for kind, note in STRUCTURE_CATALOGUE:
        table.add_row(kind, note)
    console.print(table)
    console.print()
    features_table = Table(
        "feature", "arity", "description", title=f"{len(OPTION_FEATURES)} option features"
    )
    for name in sorted(OPTION_FEATURES):
        spec = OPTION_FEATURES[name]
        features_table.add_row(name, str(spec.arity), spec.doc)
    console.print(features_table)
    console.print()
    console.print(
        f"[dim]plus every feature in `aqr features` ({len(REGISTRY)} of them), read "
        "off the underlying's daily bars. There is no exit vocabulary and its "
        "absence is the design: specs/10 D1 -- a contract is re-quoted on 1-3% of "
        "later sessions, so a stop, a target or a roll cannot be priced.[/dim]"
    )


@app.command("option-research")
def option_research(
    iterations: int = typer.Option(
        8,
        help=f"Hypotheses to test. Ceiling {OPTION_SEARCH_BUDGET}, counted across "
        "the life of the database rather than per run. The ceiling is a guardrail "
        "against a runaway loop; what actually prices a wide search is the "
        "deflation term, reported as `sharpe_inflation` with every verdict.",
    ),
    underlying: str = typer.Option("SPY", help="The one underlying. The cache holds SPY."),
    chain_root: str = typer.Option(DEFAULT_OPTIONS_ROOT, help="Option chain cache root."),
    underlying_root: str = typer.Option(
        DEFAULT_UNDERLYING_ROOT,
        help="Raw-adjusted bar cache for the underlying. Not data-sp500: option "
        "strikes are raw and an adjusted close reports a moneyness the trade "
        "never had (specs/10 D0).",
    ),
    risk_per_trade: float = typer.Option(
        0.02,
        help="Fraction of equity risked against the structure's maximum loss. 2% "
        "rather than 1% because at 1% the median put spread is unaffordable on "
        "most sessions and the run measures the account, not the rule (D8a).",
    ),
    max_concurrent: int = typer.Option(3, help="Structures open at once."),
    equity: float = typer.Option(100_000.0, help="Account the run is sized against."),
    costs: str = typer.Option(
        "IBKR_OPTIONS",
        help=f"Per-contract schedule: {', '.join(sorted(OPTION_PRESETS))}. The spread "
        "is charged from the quotes regardless (D2, D7).",
    ),
    db: Path = typer.Option(Path("runs/research.sqlite")),
    provider: str = typer.Option(
        "offline", help="Who proposes: offline | deepseek | anthropic."
    ),
    model: str = typer.Option("", help="Model id. Defaults to the provider's usual one."),
    save_to: str = typer.Option(
        "strategies/options", help="Where surviving rules are written."
    ),
) -> None:
    """Run the option research loop: propose, test, record, repeat.

    A separate command from ``aqr research`` and a separate budget from it. The
    equity campaign spent 414 hypotheses against a 600-name universe; this one
    is capped at twenty against 71 independent cycles, and the two denominators
    are kept apart in the registry so neither multiplicity bar is computed
    against the other's count.
    """
    try:
        schedule = OPTION_PRESETS[costs.strip().upper()]
    except KeyError as exc:
        raise typer.BadParameter(
            f"unknown option cost schedule {costs!r}; use one of "
            f"{', '.join(sorted(OPTION_PRESETS))}"
        ) from exc

    market, version = research_option_market(
        underlying.upper(), chain_root=chain_root, underlying_root=underlying_root
    )
    console.print(
        f"{len(market.chain.sessions)} chain sessions "
        f"{market.chain.sessions[0]} -> {market.chain.sessions[-1]}, "
        f"{len(market.underlying)} underlying bars, "
        f"{len(market.volatility) if market.volatility else 0} volatility rows"
    )
    console.print(f"[dim]{version}[/dim]")

    try:
        config = OptionResearchConfig(
            underlying=underlying.upper(),
            iterations=iterations,
            risk_per_trade=risk_per_trade,
            max_concurrent=max_concurrent,
            dataset_version=version,
            save_accepted_to=save_to,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    with Registry(db) as registry:
        spent = registry.distinct_hypotheses(family=OPTION)
        console.print(
            f"option hypotheses already recorded: {spent} of {OPTION_SEARCH_BUDGET} "
            f"(the equity search's {registry.distinct_hypotheses(family=EQUITY)} are "
            "counted separately and do not raise this bar)"
        )
        projected = spent + iterations
        if projected >= WIDE_SEARCH_WARNING_AT:
            # Said before the run, not after, because after is when the score is
            # already on the screen. The deflation term is what prices a wide
            # search, and a reader who has not thought about it will misread a
            # high score -- the first real campaign produced a 97/100 rule whose
            # Sharpe deflated to -0.28 at twenty trials.
            console.print(
                f"[yellow]this campaign will take the option search to about "
                f"{projected} distinct hypotheses[/yellow] — the deflation term "
                "scales with that count, so read `sharpe_inflation` in the "
                "overfitting report before believing any score this run produces. "
                "The window holds about 71 independent cycles (specs/10 D8)."
            )
        loop = OptionResearchLoop(
            market=market,
            registry=registry,
            config=config,
            proposer=_option_proposer(provider, model),
            backtest_config=OptionBacktestConfig(initial_equity=equity, costs=schedule),
        )
        for step in loop.run():
            colour = {"ACCEPT": "green", "PAPER": "cyan", "REVIEW": "yellow"}.get(
                step.verdict, "red"
            )
            console.print(f"[{colour}]{step}[/{colour}]")
        console.print()
        best = loop.best()
        if best and best.outcome:
            console.print(best.outcome.summary())
        console.print(
            f"\noption hypotheses in this database: "
            f"{registry.distinct_hypotheses(family=OPTION)} of {OPTION_SEARCH_BUDGET}"
        )


@app.command("option-book")
def option_book_cmd(
    strategy: str = typer.Argument(
        ...,
        help="A registered option fingerprint. Not a path: a file is editable "
        "between the sealed run and the handoff and a fingerprint is not.",
    ),
    db: Path = typer.Option(Path("runs/research.sqlite"), help="Experiment database."),
    chain_root: str = typer.Option(
        DEFAULT_OPTIONS_ROOT,
        help="Option chain cache root. The research root by default -- the book "
        "carries a rule, not strikes, so it does not need the sealed sessions.",
    ),
    underlying_root: str = typer.Option(DEFAULT_UNDERLYING_ROOT),
    equity: float = typer.Option(100_000.0, help="Account the evidence run is sized against."),
    out: Path = typer.Option(Path("runs/option_books"), help="Where to write the book."),
) -> None:
    """Write the option book for a validated rule. It is not placed anywhere.

    There is no ``--dry-run`` because there is nothing to be dry about: no code
    path in this project sends an order to anything. The output is a file, and
    executing it is a different system's job -- one that still needs the live
    chain, the delta selection, the options-shaped risk gate, the kill switch
    and the fill journal the artefact lists under ``consumer_must_supply``.

    The refusals come first and they are the point, and they are the same three
    ``target-book`` makes: the registry must know the rule, its seal must have
    been spent, and its sealed run must not have refuted it. Producing one for
    an undeclared candidate would hand off a rule whose out-of-sample verdict is
    still owed, which is the loophole the pre-registration protocol exists to
    close.
    """
    with Registry(db) as reg:
        record = reg.get_strategy(strategy)
        if record is None:
            raise typer.BadParameter(
                f"no strategy {strategy!r} in the registry. An option book that cannot "
                "be traced back to a recorded hypothesis is a rule from nowhere; run "
                "`aqr option-research` first."
            )
        if record.family != OPTION:
            raise typer.BadParameter(
                f"{record.name} [{strategy}] is an {record.family} strategy. Its "
                "handoff is `aqr target-book`, which writes weights; an option book "
                "carries a structure and a delta and the two are not interchangeable."
            )
        spec = loads_option_spec(record.spec_yaml)

        sealed = reg.sealed_run(strategy)
        if sealed is None:
            raise typer.BadParameter(
                f"{record.name} [{strategy}] has no sealed run. A book handed off "
                "before the seal is spent would trade a rule whose out-of-sample "
                "verdict is still owed. Declare it with `aqr preregister` and spend "
                "it with `python -m aqr.cli_sealed option-run`."
            )
        measurement = dict(sealed["result"].get("measurement", {}))
        if measurement.get("refuted"):
            raise typer.BadParameter(
                f"{record.name} was refuted by its sealed run on "
                f"{sealed['sealed_run_at'][:19]}: negative alpha, significant in the "
                "wrong direction. There is nothing to hand off."
            )

        declaration = reg.preregistration(strategy)
        provenance: dict[str, object] = {
            "database": str(db),
            "status": record.status,
            "score": record.score,
            "hypothesis": record.hypothesis,
            # The option search's own denominators, never the combined ones
            # (specs/10 D8). A book that carried the equity campaign's 414 would
            # overstate what this rule survived by a factor of twenty.
            "distinct_option_hypotheses": reg.distinct_hypotheses(family=OPTION),
            "sealed_look": reg.sealed_look(strategy),
            "sealed_looks_total": reg.sealed_looks(family=OPTION),
            "sealed_run_at": sealed["sealed_run_at"],
            "sealed_measurement": measurement,
            "sealed_seal": sealed["result"].get("seal", {}),
            "sealed_window_can_refute_not_confirm": (
                "about 25 independent 28-DTE cycles: entitled to say this stopped "
                "working, never entitled to say it works (specs/10 D8)"
            ),
            "preregistration": (
                {
                    "declared_at": declaration.declared_at,
                    "selection_rule": declaration.selection_rule,
                    "seal_digest": declaration.seal_digest,
                }
                if declaration
                else None
            ),
        }

        market, version = research_option_market(
            spec.underlying, chain_root=chain_root, underlying_root=underlying_root
        )
        try:
            book = build_option_book(
                spec,
                market,
                generated_at=datetime.now(UTC),
                dataset_version=version,
                provenance=provenance,
                seal=current_seal().certificate(),
                config=OptionBacktestConfig(initial_equity=equity),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        path = out / f"{spec.name}-{spec.fingerprint()}-{book.as_of}.json"
        digest = write_option_book(book, path)
        reg.record_target_book(
            strategy,
            as_of=book.as_of,
            path=str(path),
            digest=digest,
            book=book.as_dict(),
            family=OPTION,
        )

    console.print()
    console.print(book.summary())
    console.print()
    console.print(f"[green]{path}[/green]")
    console.print(f"[dim]sha256 {digest[:16]}  recorded against {strategy}[/dim]")
    console.print(
        "[dim]the rule only, deliberately without strikes: a strike named from "
        "yesterday's close is wrong by today's open. Resolving it against a live "
        "chain, sizing, an options risk gate, a kill switch and a fill journal "
        "belong to whatever executes this.[/dim]"
    )


@app.command("option-books")
def option_books_cmd(
    fingerprint: str = typer.Argument("", help="Filter to one rule."),
    db: Path = typer.Option(Path("runs/research.sqlite")),
    limit: int = typer.Option(20),
) -> None:
    """Every option book handed off, newest first."""
    with Registry(db) as reg:
        rows = reg.target_books(fingerprint or None, limit, family=OPTION)
    if not rows:
        console.print("no option books recorded yet")
        return
    table = Table("as of", "rule", "fingerprint", "structure", "DTE", "delta", "sha256", "path")
    for row in rows:
        book = row["book"]
        rule = book.get("rule", {})
        table.add_row(
            row["as_of"],
            str(book.get("spec_name", "?")),
            row["fingerprint"],
            str(rule.get("structure", "-")),
            str(rule.get("dte", {}).get("target", "-")),
            str(rule.get("anchor", {}).get("delta", "-")),
            row["digest"][:16],
            row["path"],
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
