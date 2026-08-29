"""How far along are the intraday pulls, and how fast.

The sealed CLI prints only the series that FAIL the quality check, so its log
looks like a list of failures even when the run is healthy. Files on disk are
the honest progress signal; this reads those.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AQR = REPO / "ai_quant_researcher"
UNIVERSE = AQR / "data-universes" / "sp500_pit.json"
ROOTS = (("research", "data-sp500"), ("sealed", "data-sp500-sealed"))


def _universe_size() -> int:
    raw = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    symbols: set[str] = set()
    for value in raw.get("counts", {}).values() if isinstance(raw, dict) else []:
        if isinstance(value, list):
            symbols.update(value)
    return len(symbols)


def report(timeframe: str) -> None:
    now = dt.datetime.now()
    for label, root in ROOTS:
        d = AQR / root / timeframe
        if not d.is_dir():
            print(f"{label:9s} {timeframe:3s}  (no cache yet)")
            continue
        files = list(d.glob("*.csv"))
        recent = [
            f
            for f in files
            if dt.datetime.fromtimestamp(f.stat().st_mtime) > now - dt.timedelta(minutes=30)
        ]
        mb = sum(f.stat().st_size for f in files) / 1e6
        line = f"{label:9s} {timeframe:3s}  {len(files):4d} files  {mb:7.1f} MB"
        if recent:
            newest = max(f.stat().st_mtime for f in files)
            rate = len(recent) / 30
            line += (
                f"  |  {len(recent)} in last 30m = {rate:.2f}/min"
                f"  last write {dt.datetime.fromtimestamp(newest):%H:%M:%S}"
            )
        else:
            line += "  |  idle (nothing written in 30m)"
        print(line)


if __name__ == "__main__":
    for tf in sys.argv[1:] or ["1h", "4h"]:
        report(tf)
