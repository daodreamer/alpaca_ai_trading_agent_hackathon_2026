"""Pull S&P 500 PIT intraday bars into research and sealed caches.

Research cache (``data-sp500/``) is truncated at the embargo by ``aqr pull``.
Sealed cache (``data-sp500-sealed/``) holds the full history including the last
two years for out-of-sample validation via ``aqr-sealed pull``.

Alpaca has no 4-hour bar; after 1h is cached, this script aggregates it into 4h
with the same CSV layout as every other timeframe.

Minute bars are deliberately out of scope: 682 tickers x 10 years of 1m is tens
of GB and days of wall clock, which this hackathon does not have. 5m and 1m stay
selectable via --timeframe for a narrower universe, but never run by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AQR = REPO / "ai_quant_researcher"
UNIVERSE_FILE = "data-universes/sp500_pit.json"
RESEARCH_ROOT = str(AQR / "data-sp500")
SEALED_ROOT = str(AQR / "data-sp500-sealed")
START = "2016-01-01"
# Past the embargo so ResearchProvider clamps at 2024-09-01.
RESEARCH_END = "2026-08-29"
# The sealed cache exists to validate on the embargoed years, so it needs the
# embargo window plus enough warm-up to have every feature hot on the first
# sealed session -- not the research cache's start.
#
# `aqr-sealed run` asks for EMBARGO_START - WARMUP_DAYS(1200) = 2021-05-20, but
# that constant is CALENDAR days sized for 1D bars. The longest lookback in the
# DSL, rs_rank(126), is 126 *bars*: ~126 sessions on 1D, but only ~18 sessions
# on 1h. Three months of 1h bars is ~450 bars -- over three times what the
# longest lookback needs -- so 2024-06-01 is warm well before the embargo.
# CsvProvider clips a start earlier than the file holds instead of failing, so
# the run's 2021-05-20 request is satisfied by whatever exists.
#
# This is timeframe-specific and deliberately NOT a change to WARMUP_DAYS: that
# constant is still right for the 1D cache.
SEALED_START = "2024-06-01"
ALPACA_TIMEFRAMES = ("1h",)
OPT_IN_TIMEFRAMES = ("5m", "1m")
CANARY_TIMEFRAMES = ("1D", "1h", "4h")


def _run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {stamp} ===\n")
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=AQR, stdout=fh, stderr=subprocess.STDOUT, check=False)
        fh.write(f"\nexit {proc.returncode}\n")
    return proc.returncode


def pull_research(timeframe: str, *, force: bool) -> int:
    cmd = [
        "uv",
        "run",
        "aqr",
        "pull",
        "--source",
        "alpaca",
        "--universe-file",
        UNIVERSE_FILE,
        "--csv-root",
        RESEARCH_ROOT,
        "--timeframe",
        timeframe,
        "--start",
        START,
        "--end",
        RESEARCH_END,
        "--feed",
        "sip",
    ]
    if force:
        cmd.append("--force")
    return _run(cmd, AQR / "runs" / f"pull-research-{timeframe}.log")


def pull_sealed(timeframe: str, *, force: bool) -> int:
    cmd = [
        "uv",
        "run",
        "aqr-sealed",
        "pull",
        "--timeframe",
        timeframe,
        "--start",
        SEALED_START,
        "--csv-root",
        SEALED_ROOT,
        "--feed",
        "sip",
    ]
    if force:
        cmd.append("--force")
    return _run(cmd, AQR / "runs" / f"pull-sealed-{timeframe}.log")


def resample_4h(root: str) -> int:
    """Build 4h bars from cached 1h CSVs."""
    sys.path.insert(0, str(AQR / "src"))
    from aqr.data.providers import CsvProvider
    from aqr.data.resample import aggregate_bars
    from aqr.seal import CANARY_SYMBOL

    provider = CsvProvider(root)
    source_dir = provider.root / "1h"
    if not source_dir.is_dir():
        print(f"no 1h cache at {source_dir}", file=sys.stderr)
        return 1

    written = 0
    for path in sorted(source_dir.glob("*.csv")):
        symbol = path.stem
        if symbol == CANARY_SYMBOL:
            # The tripwire is written by write_canaries, not resampled, and
            # loading it here would taint this process's seal for nothing.
            continue
        bars = provider.load(
            symbol,
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2030, 1, 1, tzinfo=UTC),
            "1h",
        )
        resampled = aggregate_bars(bars, target_timeframe="4h", factor=4)
        if len(resampled):
            provider.write(resampled)
            written += 1
    print(f"{root}/4h: {written} symbols resampled from 1h")
    return 0


def write_canaries() -> int:
    """Arm the tripwire in every research-root timeframe.

    The canary lives in the research cache and nowhere else -- never the
    sealed root, where embargoed rows are expected. Idempotent, so it runs
    after every research pull and can be re-run on its own.
    """
    sys.path.insert(0, str(AQR / "src"))
    from aqr.data.embargo import write_canary

    for timeframe in CANARY_TIMEFRAMES:
        path = write_canary(RESEARCH_ROOT, timeframe)
        print(f"canary armed: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframe",
        choices=[*ALPACA_TIMEFRAMES, *OPT_IN_TIMEFRAMES, "4h", "all"],
        default="all",
        help='Which timeframe to pull. "all" is 1h plus the 4h it is resampled '
        "into; 5m and 1m are opt-in only.",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch symbols already cached.")
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--sealed-only", action="store_true")
    parser.add_argument("--resample-only", action="store_true", help="Only build 4h from 1h.")
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="Only re-arm the research-root canaries.",
    )
    args = parser.parse_args()

    if args.canary_only:
        return write_canaries()

    if args.resample_only:
        code = resample_4h(RESEARCH_ROOT)
        return code or resample_4h(SEALED_ROOT)

    timeframes = list(ALPACA_TIMEFRAMES) if args.timeframe == "all" else [args.timeframe]
    if args.timeframe == "4h":
        return resample_4h(RESEARCH_ROOT) or resample_4h(SEALED_ROOT)

    for tf in timeframes:
        if not args.sealed_only:
            code = pull_research(tf, force=args.force)
            if code:
                return code
        if not args.research_only:
            code = pull_sealed(tf, force=args.force)
            if code:
                return code
        if tf == "1h":
            code = resample_4h(RESEARCH_ROOT)
            if code:
                return code
            code = resample_4h(SEALED_ROOT)
            if code:
                return code
    if not args.sealed_only:
        return write_canaries()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
