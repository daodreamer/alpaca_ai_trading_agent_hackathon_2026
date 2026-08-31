"""Print the sealed window three ways: the strategy, the universe, and SPY.

The sealed measurement already carries a benchmark, and it is the equal-weight
return of the whole point-in-time universe. That is the right thing to regress
residual alpha against -- it is the exposure the strategy actually ran, over the
same names -- but it is not what a reader means by "the market". The index is
cap-weighted, 2024-09 to 2026-08 was led by its largest names, and the gap
between the two is not a rounding difference.

So this adds a third curve, for reporting only. It feeds no verdict, updates no
registry row, and is not an input to anything: `SealedMeasurement` keeps
regressing against the equal-weight series, because swapping in a cap-weighted
proxy would change what `alpha` and `beta` mean in every record already stored.

**On the seal.** This process reads bars past the 2024-09-01 embargo, so the
ambient seal taints, and the script prints that rather than hiding it. Taint is
the honest outcome here: the bit means "this process saw the embargoed years",
which is true, and what makes it harmless is that nothing here can search,
propose, score or promote. It mints no `SealToken` -- only `aqr.data.embargo`
and `aqr.cli_sealed` may, enforced over `src/` by
`test_no_module_outside_the_embargo_layer_constructs_a_seal_token`, and a
script that minted one would be a hole those tests do not scan for.

**On the bar size.** The comparison is drawn at whatever timeframe the strategy
was measured on, read from its own most recent experiment rather than assumed. A
1h rule against a daily SPY would put two sampling rates in one table: both
Sharpes are annualised, from different `periods_per_year`, so they would look
comparable and would not be. `--timeframe` overrides it for a deliberate
cross-rate look.

    python scripts/report_benchmark.py
    python scripts/report_benchmark.py --fingerprint 96cbc95ab6f09a60
    python scripts/report_benchmark.py --timeframe 1h

Fetch the series first, once per timeframe:

    cd ai_quant_researcher && uv run aqr-sealed pull \\
        --universe-file data-universes/benchmark_spy.json \\
        --csv-root data-benchmark --timeframe 1D \\
        --start 2016-01-01 --end 2026-08-29
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESEARCHER = REPO / "ai_quant_researcher"
sys.path.insert(0, str(RESEARCHER / "src"))

from aqr.data.providers import CsvProvider  # noqa: E402
from aqr.seal import current as current_seal  # noqa: E402
from aqr.validation.holdout import buy_and_hold  # noqa: E402

REGISTRY = RESEARCHER / "runs" / "research.sqlite"
BENCHMARK_ROOT = RESEARCHER / "data-benchmark"
SYMBOL = "SPY"


def _sealed_measurement(fingerprint: str | None) -> dict:
    if not REGISTRY.is_file():
        raise SystemExit(
            f"no registry at {REGISTRY}\n"
            "restore it with: python scripts/pack_registry.py --unpack"
        )
    conn = sqlite3.connect(f"file:{REGISTRY}?mode=ro", uri=True)
    try:
        if fingerprint:
            row = conn.execute(
                "SELECT name, fingerprint, sealed_result FROM strategies "
                "WHERE fingerprint = ? AND sealed_result IS NOT NULL",
                (fingerprint,),
            ).fetchone()
        else:
            # Most recent sealed run, which is the one the pin points at.
            row = conn.execute(
                "SELECT name, fingerprint, sealed_result FROM strategies "
                "WHERE sealed_result IS NOT NULL ORDER BY sealed_run_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise SystemExit(
                f"no sealed run recorded for {fingerprint or '(any strategy)'}"
            )
        name, digest, blob = row
        # The bar size the rule was measured on, taken from its own most recent
        # experiment rather than assumed. A 1h rule compared against a daily SPY
        # would be two different sampling rates in one table: the Sharpes are
        # annualised from different `periods_per_year`, so the numbers would look
        # comparable and would not be.
        timeframe = conn.execute(
            "SELECT timeframe FROM experiments WHERE fingerprint = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (digest,),
        ).fetchone()
    finally:
        conn.close()
    stored = json.loads(blob)
    stored["_name"] = name
    stored["_fingerprint"] = digest
    stored["_timeframe"] = timeframe[0] if timeframe else "1D"
    return stored


def _spy_over(first: datetime, last: datetime, timeframe: str) -> tuple[object, int]:
    series = BENCHMARK_ROOT / timeframe / f"{SYMBOL}.csv"
    if not series.is_file():
        raise SystemExit(
            f"no {SYMBOL} {timeframe} bars at {series}\n"
            "pull them with (from ai_quant_researcher/):\n"
            "  uv run aqr-sealed pull --universe-file data-universes/benchmark_spy.json"
            f" \\\n    --csv-root data-benchmark --timeframe {timeframe}"
            " --start 2016-01-01 --end 2026-08-29"
        )
    bars = CsvProvider(BENCHMARK_ROOT).load(SYMBOL, first, last, timeframe)
    metrics = buy_and_hold({SYMBOL: bars})
    if metrics is None:
        raise SystemExit(f"{SYMBOL} has too few {timeframe} bars between {first} and {last}")
    return metrics, len(bars)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fingerprint",
        help="which sealed run to report on (default: the most recent)",
    )
    parser.add_argument(
        "--timeframe",
        help="override the bar size; defaults to the one the strategy was measured on",
    )
    args = parser.parse_args()

    stored = _sealed_measurement(args.fingerprint)
    timeframe = args.timeframe or stored["_timeframe"]
    m = stored["measurement"]
    first = datetime.fromisoformat(m["first_session"]).astimezone(UTC)
    last = datetime.fromisoformat(m["last_session"]).astimezone(UTC)

    spy, spy_bars = _spy_over(first, last, timeframe)

    print(
        f"sealed window {first.date()} -> {last.date()}  "
        f"({m['observations']} sessions, {timeframe} bars)"
    )
    print(f"{stored['_name']} [{stored['_fingerprint']}]")
    print()
    print(f"{'':<26}{'return':>10}{'sharpe':>9}{'maxDD':>9}")
    print(
        f"{'strategy':<26}{m['strategy_return']:>9.2%}"
        f"{m['strategy_sharpe']:>9.2f}{m['max_drawdown']:>9.2%}"
    )
    print(
        f"{'universe, equal weight':<26}{m['benchmark_return']:>9.2%}"
        f"{m['benchmark_sharpe']:>9.2f}{'':>9}"
    )
    print(
        f"{'SPY, buy and hold':<26}{spy.total_return:>9.2%}"
        f"{spy.sharpe:>9.2f}{-spy.max_drawdown:>9.2%}"
    )
    print()
    print(
        "The residual alpha, beta and t in the sealed record regress against the "
        "equal-weight\nuniverse, not against SPY. This third row is for reading, "
        "not for scoring."
    )
    # `max_drawdown` is stored negative on the measurement and positive on
    # `Metrics`; both are printed with the same sign above, so say which is which
    # rather than leaving a reader to trust that they were reconciled.
    print(
        f"SPY: {spy_bars} {timeframe} bars from {BENCHMARK_ROOT.name}/{timeframe}, "
        f"drawdown shown negated to match the row above it."
    )
    seal = current_seal().certificate()
    print(
        f"seal: phase {seal['phase']}, tainted {str(seal['tainted']).lower()} "
        "-- expected, this process reads past the embargo and scores nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
