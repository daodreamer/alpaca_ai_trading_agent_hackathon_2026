"""Compress the research registry for version control, and restore it.

The registry is evidence: it holds `distinct_hypotheses()`, the
multiple-comparisons denominator, and the sealed runs, which are one-shot by
construction and cannot be re-created. So it is committed rather than ignored.

It is committed compressed because of its shape. The database reached 51 MB and
47 MB of that is a single column -- `experiments.robustness`, which stores JSON
diagnostics per experiment. JSON compresses about five to one, sqlite does not
delta-compress between commits, and 51 MB is already past the size GitHub warns
at. Every uncompressed commit would add another whole copy to the pack.

`VACUUM` runs first so a repacked archive does not carry freed pages, and the
copy it vacuums is a temporary one -- this never writes to the live database,
which may be open in another process.

    python scripts/pack_registry.py             # .sqlite -> .sqlite.gz
    python scripts/pack_registry.py --unpack    # .sqlite.gz -> .sqlite

Imports neither project. Packing bytes needs nothing from `aqr`, and
`backend/tests/test_boundaries.py` holds that line for everything under
`scripts/`.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "ai_quant_researcher" / "runs" / "research.sqlite"
ARCHIVE = LIVE.with_suffix(".sqlite.gz")

# gzip stamps mtime into its header, so an unchanged database would still
# produce a different archive on every run and show up as a spurious diff.
# Zero is the "no timestamp" value the format reserves for exactly this.
NO_MTIME = 0


def pack() -> int:
    if not LIVE.is_file():
        print(f"no database at {LIVE}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "research.sqlite"
        shutil.copy2(LIVE, staged)
        # `with sqlite3.connect(...)` commits the transaction and leaves the
        # connection open, which on Windows keeps a handle on the file and
        # makes the temporary directory undeletable. Close it explicitly.
        conn = sqlite3.connect(staged)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        with (
            open(staged, "rb") as raw,
            gzip.GzipFile(ARCHIVE, "wb", compresslevel=9, mtime=NO_MTIME) as out,
        ):
            shutil.copyfileobj(raw, out)
    print(
        f"{LIVE.name} {LIVE.stat().st_size / 1e6:.1f} MB"
        f"  ->  {ARCHIVE.name} {ARCHIVE.stat().st_size / 1e6:.1f} MB"
    )
    return 0


def unpack(force: bool) -> int:
    if not ARCHIVE.is_file():
        print(f"no archive at {ARCHIVE}", file=sys.stderr)
        return 1
    if LIVE.is_file() and not force:
        # Overwriting a live registry would discard experiments that were run
        # since the archive was packed, and those are the expensive half.
        print(
            f"{LIVE.name} already exists; pass --force to overwrite it",
            file=sys.stderr,
        )
        return 1
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ARCHIVE, "rb") as raw, open(LIVE, "wb") as out:
        shutil.copyfileobj(raw, out)
    print(f"{ARCHIVE.name}  ->  {LIVE.name} {LIVE.stat().st_size / 1e6:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--unpack",
        action="store_true",
        help="restore runs/research.sqlite from the committed archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --unpack, overwrite an existing database",
    )
    args = parser.parse_args()
    return unpack(args.force) if args.unpack else pack()


if __name__ == "__main__":
    raise SystemExit(main())
