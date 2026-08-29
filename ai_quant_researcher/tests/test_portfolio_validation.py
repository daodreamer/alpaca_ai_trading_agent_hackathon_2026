"""What a portfolio spec has to survive before any compute is spent on it.

The validator asked one question of every strategy: is ``entry`` a condition,
and does it ever fire. A portfolio spec has no ``entry``, so the first check
rejected every one of them before the pipeline began -- correctly, in the sense
that the rule it applied was the rule it had, and uselessly, in the sense that
the rule did not apply.

The portfolio equivalents are not cosmetic renames. "Does the entry ever fire"
becomes two separate questions, and skipping either one lets a dead rule consume
a full walk-forward before saying nothing:

* **Is the ranking ever defined?** A ``rank_by`` that is NaN everywhere -- a
  window longer than the history, a feature that never warms up -- produces an
  empty book on every rebalance and an equity curve that is a flat line at the
  starting capital. That is not a strategy returning zero; it is a strategy that
  never ran.

* **Does the ranking distinguish anything?** A ``rank_by`` that returns the same
  value for every name on every bar sorts the universe alphabetically. It will
  produce trades, an equity curve, and a perfectly respectable-looking result
  that is a fact about the symbol names.
"""

from __future__ import annotations

import numpy as np

from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict
from aqr.dsl.validator import validate, validate_against
from aqr.features.cross_section import CrossSection

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
N = 400
T0 = 1_400_000_000


def _bars(symbol: str, drift: float, n: int = N) -> Bars:
    t = np.arange(T0, T0 + n * 86_400, 86_400, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1e6),
    )


def _universe(n: int = N) -> dict[str, Bars]:
    return {s: _bars(s, 0.0003 * (i + 1), n) for i, s in enumerate(SYMBOLS)}


def _spec(**over: object):
    body: dict[str, object] = {
        "name": "xs_probe",
        "mode": "portfolio",
        "rank_by": "roc(20)",
        "hold": 2,
        "rebalance_every": 10,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


def _check(spec, data: dict[str, Bars] | None = None):
    data = data if data is not None else _universe()
    return validate_against(spec, data["AAA"], CrossSection(data))


# --------------------------------------------------------------------------


def test_a_portfolio_spec_is_not_rejected_for_having_no_entry() -> None:
    """The bug this file exists for: every portfolio spec failed at the first
    check, before the pipeline had run anything."""
    report = validate(_spec())
    assert report.ok, report.errors


def test_a_signal_spec_still_needs_an_entry_condition() -> None:
    """The old rule is unchanged where it applies."""
    report = validate(
        spec_from_dict(
            {
                "strategy": {
                    "name": "trigger",
                    "entry": "close > ema(20)",
                    "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
                }
            }
        )
    )
    assert report.ok, report.errors


def test_a_ranking_needing_more_history_than_exists_is_an_error() -> None:
    """Caught by the warm-up check before the ranking is ever evaluated, which
    is the cheaper place to catch it."""
    report = _check(_spec(rank_by="roc(5000)"))
    assert not report.ok
    assert any("warm up" in e for e in report.errors), report.errors


def test_a_ranking_that_is_never_defined_is_an_error() -> None:
    """NaN everywhere means an empty book on every rebalance, and an equity
    curve that is a flat line at the starting capital -- which reads like a
    strategy that lost nothing rather than one that never ran.

    A name that never trades is the realistic way to get here: ``rvol`` divides
    by an average volume of zero, so the ranking warms up fine and is undefined
    forever after.
    """
    data = _universe()
    silent = {
        symbol: Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=bars.event_time,
            open=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=np.zeros(len(bars)),
        )
        for symbol, bars in data.items()
    }
    report = _check(_spec(rank_by="rvol(20)"), silent)
    assert not report.ok, "a ranking that is never defined must not reach a backtest"
    # Which check catches it is not the contract. The warm-up check gets there
    # first, because a feature that is NaN forever never warms up -- and the
    # portfolio branch stays as the backstop for a ranking that warms up and
    # then goes undefined, which no current feature does but one could.


def test_a_ranking_that_cannot_distinguish_anything_is_an_error() -> None:
    """A constant ranking sorts the universe by symbol name. It produces trades,
    a curve, and a result that is a fact about the alphabet."""
    report = _check(_spec(rank_by="1.0"))
    assert not report.ok
    assert any("constant" in e or "distinguish" in e for e in report.errors), report.errors


def test_a_screen_that_never_passes_is_an_error() -> None:
    report = _check(_spec(screen="close > 1000000000"))
    assert not report.ok
    assert any("screen" in e for e in report.errors), report.errors


def test_holding_more_names_than_the_universe_has_is_a_warning() -> None:
    """Not an error: the book simply holds everything, which is a legitimate
    degenerate case worth flagging rather than blocking. It is also exactly the
    benchmark, and a strategy that is the benchmark has no alpha to find."""
    report = _check(_spec(hold=50))
    assert report.ok
    assert any("universe" in w for w in report.warnings), report.warnings


def test_a_healthy_portfolio_spec_passes() -> None:
    report = _check(_spec())
    assert report.ok, report.errors
