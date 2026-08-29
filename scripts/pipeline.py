#!/usr/bin/env python3
"""The chain, driven from one place — specs/09 D0.

```
aqr research → walk-forward → pre-register → sealed run   (occasionally)
                                                  ↓
                              aqr target-book → a JSON file   (every session)
                                                  ↓
                              alphagate equity-run → orders   (every session)
```

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
    if explicit:
        return explicit
    env = os.environ.get("ALPHAGATE_STRATEGY_FINGERPRINT")
    if env:
        return env
    path = ROOT / ".env.local"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ALPHAGATE_STRATEGY_FINGERPRINT":
                found = value.strip().strip("\"'")
                if found:
                    return found
    raise SystemExit(
        "no strategy pinned. Set ALPHAGATE_STRATEGY_FINGERPRINT in .env.local, "
        "or pass --fingerprint. There is deliberately no default: the pin is "
        "what makes 'only the strategy the researcher validated' checkable."
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


STAGES = {
    "refresh": stage_refresh,
    "book": stage_book,
    "trade": stage_trade,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Refresh the cache, rebuild the target book, and trade it. "
            "Runs two projects' CLIs as subprocesses and imports neither."
        ),
    )
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="STAGE",
        help=f"which of {', '.join(STAGES)} to run, in order. Default: all three.",
    )
    parser.add_argument("--fingerprint", default=None)
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

    wanted = args.stages or list(STAGES)
    unknown = [name for name in wanted if name not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    print(f"AlphaGate pipeline — {' → '.join(wanted)}")
    if args.dry_run:
        print("dry run: no bars pulled, no orders placed")

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
