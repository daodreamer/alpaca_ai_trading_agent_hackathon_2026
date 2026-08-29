"""The same hole, in the event-driven engine.

`run_portfolio` froze a delisted holding in the book forever. `run_backtest` has
the same failure in a milder form, and the mildness is why it went unnoticed: the
position *is* eventually closed, at the symbol's own final close, because the
end-of-run sweep falls back to the last bar index. The trade record looks
correct, the P&L reconciles, and every existing test stays green.

The mechanism is an asymmetry. A symbol absent at a step is skipped for every
decision -- `bar_of_step.get(step)` is `None`, so the stop, the target, the
holding limit and the exit signal are never evaluated again -- while `last_price`
still carries its final mark forward. That is right for a holiday and wrong for a
terminal bar, and the engine cannot tell them apart.

**What is and is not observable, because the difference matters.** The position
sits in the book at a frozen mark for the rest of the run, and it is tempting to
call that a corrupted equity curve. It is not measurable as one: a position
marked at a constant is numerically identical to the cash it would have become,
so the curve, the exposure (computed from trade timestamps that were already
right) and the drawdown all come out the same. Two consequences do survive into
the output, and they are what the tests below pin:

* **The slot is never released.** The position occupies one of `max_positions`
  for the rest of the run, so a book limited to a few names leaks capacity
  toward exactly the names that delist -- which are not a random sample.
* **The exit is mis-attributed.** It is reported as `end_of_data`, which is true
  of the symbol and false of the run, so an exit-reason breakdown cannot
  distinguish "the backtest ended" from "this company stopped existing".
"""

from __future__ import annotations

import numpy as np

from aqr.backtest.engine import BacktestConfig, run_backtest
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
N = 300
LAST = 150
T0 = 1_400_000_000


def _bars(symbol: str, n: int = N, drift: float = 0.0008) -> Bars:
    t = np.arange(T0, T0 + n * 86_400, 86_400, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        volume=np.full(n, 1e6),
    )


def _data(last: int = LAST) -> dict[str, Bars]:
    data = {s: _bars(s, drift=0.0002) for s in SYMBOLS}
    data["AAA"] = _bars("AAA", n=last)
    return data


def _spec(**over: object):
    body: dict[str, object] = {
        "name": "always_in",
        # Fires on every bar; the stop is far away and the holding limit long,
        # so nothing closes a position except the code under test.
        "entry": "close > 0",
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "max_positions": 4,
        "exit": {
            "stop_loss": {"type": "percent", "multiplier": 0.9},
            "take_profit": {"type": "none"},
            "max_holding_bars": 100000,
        },
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def _run(data: dict[str, Bars] | None = None, **over: object):
    return run_backtest(_spec(**over), data if data is not None else _data(), CONFIG)


# --------------------------------------------------------------------------


def test_the_exit_timestamp_was_already_right() -> None:
    """Recorded because it is why this went unnoticed.

    The end-of-run sweep falls back to the symbol's last bar index, so the trade
    is stamped with AAA's own final bar and the P&L reconciles. Nothing in the
    trade record looks wrong; the damage is entirely in the 150 steps during
    which the position was live and unexitable.
    """
    result = _run()
    exits = [t for t in result.trades if t.symbol == "AAA"]
    assert exits, "AAA was never traded"
    assert exits[-1].exit_time == int(_data()["AAA"].event_time[-1])


def test_the_exit_says_why_it_happened() -> None:
    """`end_of_data` is true of the symbol and false of the run. A reason that
    describes the wrong event cannot be counted in an exit-reason breakdown."""
    result = _run()
    exits = [t for t in result.trades if t.symbol == "AAA"]
    assert exits
    assert exits[-1].exit_reason == "delisted", exits[-1].exit_reason


def test_the_slot_is_released() -> None:
    """The capacity leak. With one slot, a delisted name holds it for the rest of
    the run and no other symbol is ever traded -- and the names that delist are
    not a random sample."""
    result = _run(_data(), max_positions=1)
    traded = {t.symbol for t in result.trades}
    assert len(traded) > 1, f"only {traded} was ever traded"


def test_a_universe_where_nothing_delists_is_unaffected() -> None:
    """Every existing result was measured on equal-length series. The change must
    be invisible there."""
    intact = {s: _bars(s, drift=0.0002) for s in SYMBOLS}
    result = run_backtest(_spec(), intact, CONFIG)
    assert not [t for t in result.trades if t.exit_reason == "delisted"]
