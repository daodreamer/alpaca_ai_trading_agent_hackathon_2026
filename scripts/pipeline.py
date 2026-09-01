#!/usr/bin/env python3
"""The chain, driven from one place — specs/09 D0.

Two chains, one shape. The equity sleeve:

```
aqr research → walk-forward → pre-register → sealed run   (occasionally)
                                                  ↓
                              aqr target-book → a JSON file   (every session)
                                                  ↓
                              alphagate equity-run → orders   (every session)
```

and the options sleeve, which is the same four steps against a different
artefact — an *option book*, which carries a rule rather than a vector of
weights, because the executor rebuilds the structure from live quotes and a
strike named from yesterday's close is wrong by today's open:

```
aqr option-research → pre-register → cli_sealed option-run   (occasionally)
                                                  ↓
                              aqr option-book → a JSON file   (every session)
                                                  ↓
                              alphagate iv-seed → the rank the rule reads
                                                  ↓
                              alphagate run → orders          (every session)
```

The `iv-seed` step has no equity counterpart and is not an optimisation. The
researched rule's entry is `iv_rank() < 15`, and `iv_rank` needs a year of
implied-volatility history that Alpaca will not serve without a signed OPRA
agreement (`agent/iv_store.py`). Without the seed the rule is not *false*, it is
*undecidable*, and the agent stands aside every cycle — which looks exactly like
a quiet market from the outside. So it runs before every session.

**This script imports neither project.** It runs their command-line interfaces
as subprocesses, which is the only coupling specs/09 D0 permits: the researcher
holds no money and places no orders, AlphaGate holds both, and an import in
either direction would make one project's invariants the other's problem. The
seam between them is the target-book file; this is just the thing that runs the
two commands either side of it in order.

Three stages, and the reason they are separate commands rather than one
button:

**`refresh`** rewrites the sealed cache with full history through yesterday.
Network, minutes. It runs with `--force`, so it *replaces* each symbol's file
rather than appending — which is why the start date is the cache's own origin
and not a window sized to the strategy's warm-up. See `CACHE_START`.

It runs `python -m aqr.cli_sealed pull`, **not** `aqr pull`, and the difference
is the whole embargo. `aqr pull` clamps every request at 2024-09-01 and has no
escape hatch — the researcher's own comment says so: "a flag on the search
binary that could read the sealed years would make the phase separation a
convention again". Reading the present is a different binary, and that is what
keeps "search" and "read the embargoed years" apart. The first version of this
script called the search binary, which silently pulled a cache that stopped two
years ago.

**`book`** re-runs the *validated* strategy over the refreshed cache and writes
the weights it holds today. Deterministic given the cache. It refuses outright
unless the registry knows the strategy, its seal has been spent, and the sealed
window did not refute it.

**`trade`** hands that file to AlphaGate, which prices it, gates every order and
places what survives.

The stage that is deliberately *not* here is `research`. A campaign burns looks
against the sealed window — the multiplicity denominator every claim in this
repository is deflated by — and a nightly cron that quietly screened another
seven candidates would invalidate the `t` printed on the dashboard without
anyone noticing. Running one is a decision a person makes, and `README.md`
documents the commands.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCHER = ROOT / "ai_quant_researcher"
BACKEND = ROOT / "backend"

DEFAULT_UNIVERSE = "sp500_pit"
DEFAULT_VOLATILITY_CSV = "data-options-sealed/volatility_history/SPY.csv"
"""The vendor implied-volatility table `iv-seed` reads, inside the researcher.

A path handed to AlphaGate's own CLI, which parses the CSV itself. Named here
rather than defaulted inside `alphagate` so that the one place that knows both
projects' layouts is this script — which is the only place allowed to."""

IV_WINDOW_DAYS = 365
"""How much history to seed, and it is load-bearing rather than a default.

`options/volatility.py` ranks against the whole stored history, and the
researched rule meant the vendor's own one-year range. Seed seven years and
`iv_rank` ranks against a window containing March 2020 — a different number
under the same name, which is the substitution specs/07 D3 refuses everywhere
else."""
DEFAULT_CACHE = "data-sp500-sealed"
"""The only root that holds sessions up to the present.

`data-sp500` is physically truncated at the embargo and must stay that way —
it is the cache the *search* reads, and the whole seal rests on it containing
nothing past 2024-09-01. A book is built from the sealed root because a book is
about today, and producing one taints the seal by construction. That is the
correct record rather than a defect: the taint says "this process read past the
embargo", which is the entire job here."""

CACHE_START = "2016-01-01"
"""The first bar the sealed cache holds, and the first bar every refresh asks for.

**Not a shorter window, and this cost seven years of history to learn.** The
refresh runs with `--force`, which rewrites each symbol's file rather than
appending to it — so a narrowed `--start` does not fetch *less*, it *replaces*
the cache with less. The first version asked for 1200 days on the grounds that a
book build cannot see further back than the strategy's warm-up, which is true and
irrelevant: the file it overwrote is also what a future sealed run, an audit and
`aqr seal-check` read.

Matches the date the sealed cache was originally built with, so a refresh
extends it forward and changes nothing behind."""


def run(command: list[str], *, cwd: Path, dry: bool) -> int:
    """Run one stage, echoing what it is about to do.

    Echoed because these are the commands a person types by hand when something
    goes wrong, and a driver that hides them is a driver you have to read the
    source of at 09:25.
    """
    printable = " ".join(command)
    print(f"\n$ ({cwd.name}) {printable}", flush=True)
    if dry:
        return 0
    # No shell, and every element of `command` is built above from constants and
    # validated flags. The fingerprint is the only value reaching here from
    # configuration, and the book loader re-checks it against the pin anyway.
    return subprocess.run(command, cwd=cwd, check=False).returncode  # noqa: S603


def strategy_fingerprint(explicit: str | None) -> str:
    """The pinned strategy, from the flag or from `.env.local`.

    Read out of the same variable AlphaGate reads, rather than passed between
    stages, so the book that is built and the book that is executed cannot
    disagree about which strategy this is. A mismatch would be caught at load —
    that is what the pin is for — but catching it here costs nothing and saves a
    pull.
    """
    return _pinned(
        explicit,
        "ALPHAGATE_STRATEGY_FINGERPRINT",
        "the equity strategy the research validated (specs/09 D1)",
    )


def _pinned(explicit: str | None, variable: str, what: str) -> str:
    """A fingerprint from the flag, the environment, or `.env.local` — in that order.

    Read out of the same variable AlphaGate reads, rather than passed between
    stages, so the book that is built and the book that is executed cannot
    disagree about which rule this is. A mismatch would be caught at load — that
    is what the pin is for — but catching it here costs nothing and saves a pull.
    """
    if explicit:
        return explicit
    env = os.environ.get(variable)
    if env:
        return env
    path = ROOT / ".env.local"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == variable:
                found = value.strip().strip("\"'")
                if found:
                    return found
    raise SystemExit(
        f"nothing pinned for {what}. Set {variable} in .env.local, or pass the "
        "matching --fingerprint flag. There is deliberately no default: the pin "
        "is what makes 'only the rule the researcher validated' checkable."
    )


def option_fingerprint(explicit: str | None) -> str:
    """The pinned option rule, from the flag or from `.env.local`.

    The sibling of `strategy_fingerprint`, and a separate variable rather than a
    shared one because the two sleeves execute two different rules validated
    against two different sealed windows. One pin covering both would make
    "which rule was live" unanswerable the moment they diverged, which they
    already have.
    """
    return _pinned(
        explicit,
        "ALPHAGATE_OPTION_FINGERPRINT",
        "the option rule the research validated (specs/07 D1)",
    )


def previous_session(today: date) -> date:
    """The last date a book can describe: yesterday, or Friday over a weekend.

    Approximate on purpose — it is the *end* of a pull range, and asking for a
    holiday costs an empty response rather than a wrong answer. The exchange
    calendar is the broker's to know, and AlphaGate asks it (`read_clock`)
    before it trades.
    """
    yesterday = today - timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)
    return yesterday


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_refresh(args: argparse.Namespace) -> int:
    """Pull bars up to the last session into the sealed cache.

    Through `aqr.cli_sealed`, which is the only entry point permitted past the
    embargo. `--universe` and `--source` are not passed: that binary pulls the
    point-in-time membership file it is given and fetches from Alpaca, because
    those are the only choices a sealed pull has.
    """
    end = args.end or previous_session(datetime.now(UTC).date()).isoformat()
    start = args.start or CACHE_START
    command = [
        "uv", "run", "python", "-m", "aqr.cli_sealed", "pull",
        "--start", start,
        "--end", end,
        "--csv-root", args.cache,
        "--force",
    ]
    return run(command, cwd=RESEARCHER, dry=args.dry_run)


def stage_book(args: argparse.Namespace) -> int:
    """Re-run the validated strategy and write today's target book.

    `--force` is not offered and there is nothing to be dry about: this writes a
    file and sends it nowhere. `aqr target-book` refuses outright unless the
    seal has been spent and the sealed run did not refute the rule, which is the
    loophole pre-registration exists to close.
    """
    end = args.end or previous_session(datetime.now(UTC).date()).isoformat()
    command = [
        "uv", "run", "aqr", "target-book", strategy_fingerprint(args.fingerprint),
        "--source", "csv",
        "--csv-root", args.cache,
        "--universe", args.universe,
        "--end", end,
    ]
    # Runs even under `--dry-run`: this writes a file and sends it nowhere, and
    # a dry pipeline that skipped it would plan against yesterday's book, which
    # is not the thing being rehearsed.
    return run(command, cwd=RESEARCHER, dry=False)


def stage_trade(args: argparse.Namespace) -> int:
    """Hand the book to AlphaGate.

    `equity-plan` when `--dry-run`, `equity-run` otherwise. Not one command with
    a flag, because the two are different acts and the dashboard records them as
    different stages — a journal that had to be read as "submitted, but not
    really" would be worth less than no journal.
    """
    if args.dry_run:
        command = ["uv", "run", "python", "-m", "alphagate", "equity-plan"]
    else:
        command = ["uv", "run", "python", "-m", "alphagate", "equity-run"]
    if args.fingerprint:
        command += ["--fingerprint", args.fingerprint]
    return run(command, cwd=BACKEND, dry=False)


def stage_option_book(args: argparse.Namespace) -> int:
    """Write today's option book for the pinned option rule.

    Refuses for the same reasons `target-book` does — unspent seal, refuted
    sealed run, wrong registry status — and writes the *rule*, deliberately
    without strikes. Runs even under `--dry-run`, like `book` above and for the
    same reason: it writes a file and sends it nowhere, and a rehearsal against
    yesterday's book is not a rehearsal of today.
    """
    command = [
        "uv", "run", "aqr", "option-book", option_fingerprint(args.option_fingerprint),
    ]
    return run(command, cwd=RESEARCHER, dry=False)


def stage_iv_seed(args: argparse.Namespace) -> int:
    """Top up the implied-volatility history the option rule's entry reads.

    Idempotent per session, so running it every day is correct rather than
    merely harmless. Runs under `--dry-run` too: it reads a CSV and appends to a
    local file, places nothing, and skipping it would make the dry run rehearse
    a rule that cannot be decided -- which is not the failure being rehearsed.
    """
    command = [
        "uv", "run", "python", "-m", "alphagate", "iv-seed",
        "--from", str(RESEARCHER / DEFAULT_VOLATILITY_CSV),
        "--symbol", args.underlying,
        "--days", str(IV_WINDOW_DAYS),
    ]
    return run(command, cwd=BACKEND, dry=False)


def stage_options_trade(args: argparse.Namespace) -> int:
    """Hand the option book to AlphaGate.

    `once` when `--dry-run`, `run` otherwise. The asymmetry is deliberate and is
    documented in CLAUDE.md section 6: `once` defaults to NOT placing because it
    is the debugging command, and `run` defaults to placing because running a
    whole session is the thing that is meant to trade. Two commands rather than
    one flag, so the journal records two different acts.
    """
    if args.dry_run:
        command = ["uv", "run", "python", "-m", "alphagate", "once"]
    else:
        command = ["uv", "run", "python", "-m", "alphagate", "run"]
    return run(command, cwd=BACKEND, dry=False)


STAGES = {
    "refresh": stage_refresh,
    "book": stage_book,
    "trade": stage_trade,
    "option-book": stage_option_book,
    "iv-seed": stage_iv_seed,
    "options-trade": stage_options_trade,
}

EQUITY_CHAIN = ("refresh", "book", "trade")
OPTIONS_CHAIN = ("option-book", "iv-seed", "options-trade")
"""The two sleeves' stages, named so `--only` can select one.

`refresh` is in the equity chain alone because it pulls *stock* bars. The option
chain cache is refreshed by `aqr options-pull`, which is hours of transfer from
a public database and is not a per-session step; it is run by hand when the
vendor publishes, and `README.md` documents it."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Refresh the caches, rebuild both books, and trade them. "
            "Runs two projects' CLIs as subprocesses and imports neither."
        ),
    )
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="STAGE",
        help=(
            f"which of {', '.join(STAGES)} to run, in order. "
            "Default: both sleeves, equity first."
        ),
    )
    parser.add_argument("--fingerprint", default=None, help="equity strategy pin override")
    parser.add_argument(
        "--option-fingerprint", default=None, help="option rule pin override"
    )
    parser.add_argument(
        "--only",
        choices=("equity", "options"),
        default=None,
        help="run one sleeve's chain instead of both",
    )
    parser.add_argument(
        "--underlying", default="SPY", help="the option sleeve's one underlying"
    )
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--source", default="alpaca")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None, help="last session; default yesterday")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip the pull, still build the book, and plan without submitting",
    )
    args = parser.parse_args(argv)

    if args.stages and args.only:
        parser.error("--only selects a chain; naming stages selects them explicitly")
    if args.only == "equity":
        wanted = list(EQUITY_CHAIN)
    elif args.only == "options":
        wanted = list(OPTIONS_CHAIN)
    else:
        wanted = args.stages or [*EQUITY_CHAIN, *OPTIONS_CHAIN]
    unknown = [name for name in wanted if name not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    print(f"AlphaGate pipeline — {' → '.join(wanted)}")
    if args.dry_run:
        print("dry run: no bars pulled, no orders placed")
    print(
        "the two sleeves are budgeted apart (specs/03 D6): a failure in one "
        "stops the run before the other trades on a stale book"
    )

    for name in wanted:
        code = STAGES[name](args)
        if code != 0:
            # Stop rather than continue. A book built on a failed pull is a book
            # about last week, and trading one is worse than not trading.
            print(f"\n! {name} exited {code}; stopping.", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
