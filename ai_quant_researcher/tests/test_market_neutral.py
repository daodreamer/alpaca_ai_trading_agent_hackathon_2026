"""Strategies that can be long and short at the same time.

Every strategy so far picked one direction for the whole run. That was enough to
express "short when X" but not "long the strong names and short the weak ones",
which is the form the cross-sectional features were built for and the only form
whose result is not dominated by market beta.

It matters because of a measured finding, not a preference. Fourteen long-only
strategies reached PAPER and all fourteen lost to buy-and-hold: on NASDAQ names
over 2010-2026 a long-only rule makes money almost whatever the rule is, so its
Sharpe is mostly the market's. A rule that is long and short in similar size has
no such crutch -- and no such excuse.

Two properties carry it:

**Position-level direction.** ``long`` was decided once per run and threaded
through sizing, stop placement, intrabar exit resolution and borrow cost. Each of
those now asks the position, not the strategy. A stop above the entry is correct
for a short and catastrophic for a long.

**A short leg is not the long leg negated.** The spec carries a separate
``short_entry`` expression. Mirroring the long rule would express a hypothesis
nobody proposed -- that the same condition predicts up moves and its negation
predicts down ones -- which is exactly the symmetry equities do not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import run_backtest
from aqr.data.bars import Bars
from aqr.dsl.loader import loads

NEUTRAL = """
strategy:
  name: cross_sectional_neutral_v1
  hypothesis: Strength persists and weakness persists; own the spread, not the market.
  direction: market_neutral
  universe:
    symbols: [SPY, QQQ, IWM]
    timeframe: 1D
  entry: rs_rank(60) > 0.66
  short_entry: rs_rank(60) < 0.34
  exit:
    stop_loss: {type: atr, multiplier: 2.5, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20
  sizing:
    risk_per_trade: 0.005
    max_position_pct: 0.25
  max_positions: 4
"""

LONG_ONLY = NEUTRAL.replace("  direction: market_neutral\n", "").replace(
    "  short_entry: rs_rank(60) < 0.34\n", ""
)


class TestSpec:
    def test_market_neutral_is_a_direction(self) -> None:
        spec = loads(NEUTRAL)
        assert spec.direction == "market_neutral"
        assert spec.short_entry == "rs_rank(60) < 0.34"

    def test_a_neutral_strategy_without_a_short_leg_is_rejected(self) -> None:
        """Otherwise it is a long-only strategy wearing the label, and the label
        is what a reader trusts."""
        broken = NEUTRAL.replace("  short_entry: rs_rank(60) < 0.34\n", "")
        with pytest.raises(ValueError, match="short_entry"):
            loads(broken)

    def test_a_short_leg_on_a_one_directional_strategy_is_rejected(self) -> None:
        confused = LONG_ONLY.replace(
            "  entry: rs_rank(60) > 0.66",
            "  direction: long\n  entry: rs_rank(60) > 0.66\n  short_entry: rs_rank(60) < 0.34",
        )
        with pytest.raises(ValueError, match="short_entry"):
            loads(confused)

    def test_the_short_leg_is_part_of_the_fingerprint(self) -> None:
        a = loads(NEUTRAL)
        b = loads(NEUTRAL.replace("rs_rank(60) < 0.34", "rs_rank(60) < 0.20"))
        assert a.fingerprint() != b.fingerprint()

    def test_the_short_leg_features_are_counted(self) -> None:
        spec = loads(NEUTRAL.replace("rs_rank(60) < 0.34", "rsi(21) < 30"))
        names = {key.name for key in spec.features()}
        assert "rsi" in names

    def test_the_short_leg_counts_toward_the_parameter_budget(self) -> None:
        one_leg = loads(LONG_ONLY)
        two_legs = loads(NEUTRAL)
        assert two_legs.parameter_count() > one_leg.parameter_count()


class TestTrading:
    def test_it_takes_positions_on_both_sides(self, universe: dict[str, Bars]) -> None:
        result = run_backtest(loads(NEUTRAL), universe)
        directions = {t.direction for t in result.trades}
        assert directions == {"long", "short"}, f"only traded {directions}"

    def test_a_short_position_stops_out_above_its_entry(
        self, universe: dict[str, Bars]
    ) -> None:
        """The direction of a stop is the one thing that cannot be got wrong
        halfway. A stop below the entry on a short is not a stop."""
        result = run_backtest(loads(NEUTRAL), universe)
        shorts = [t for t in result.trades if t.direction == "short"]
        assert shorts
        stopped = [t for t in shorts if t.exit_reason.startswith("stop")]
        assert stopped
        assert all(t.exit_price > t.entry_price for t in stopped)

    def test_a_long_position_stops_out_below_its_entry(
        self, universe: dict[str, Bars]
    ) -> None:
        result = run_backtest(loads(NEUTRAL), universe)
        longs = [t for t in result.trades if t.direction == "long"]
        stopped = [t for t in longs if t.exit_reason.startswith("stop")]
        assert stopped
        assert all(t.exit_price < t.entry_price for t in stopped)

    def test_a_short_pays_borrow_and_a_long_does_not(
        self, universe: dict[str, Bars]
    ) -> None:
        result = run_backtest(loads(NEUTRAL), universe)
        held = [t for t in result.trades if t.bars_held > 3]
        shorts = [t for t in held if t.direction == "short"]
        assert shorts, "no short held long enough to accrue a borrow fee"
        assert all(t.fees > 0 for t in shorts)

    def test_a_falling_symbol_makes_money_on_the_short_side(self) -> None:
        spec = loads(NEUTRAL.replace('[SPY, QQQ, IWM]', '[SPY, QQQ, IWM, DOWN]'))
        result = run_backtest(spec, _diverging())
        shorts = [t for t in result.trades if t.direction == "short" and t.symbol == "DOWN"]
        assert shorts
        assert sum(t.net_pnl for t in shorts) > 0


class TestExposure:
    def test_both_legs_share_the_position_limit(self, universe: dict[str, Bars]) -> None:
        spec = loads(NEUTRAL.replace("max_positions: 4", "max_positions: 2"))
        result = run_backtest(spec, universe)
        # Reconstruct concurrent holdings from the trade list.
        events: list[tuple[int, int]] = []
        for trade in result.trades:
            events.append((trade.entry_time, 1))
            events.append((trade.exit_time, -1))
        events.sort()
        open_now = peak = 0
        for _, delta in events:
            open_now += delta
            peak = max(peak, open_now)
        assert peak <= 2, f"held {peak} positions against a limit of 2"

    def test_one_symbol_is_never_long_and_short_at_once(
        self, universe: dict[str, Bars]
    ) -> None:
        result = run_backtest(loads(NEUTRAL), universe)
        by_symbol: dict[str, list[tuple[int, int, str]]] = {}
        for t in result.trades:
            by_symbol.setdefault(t.symbol, []).append((t.entry_time, t.exit_time, t.direction))
        for spans in by_symbol.values():
            spans.sort()
            for (_, exit_a, _), (entry_b, _, _) in zip(spans, spans[1:], strict=False):
                assert entry_b >= exit_a, "overlapping positions in one symbol"


class TestBackwardCompatibility:
    def test_a_long_only_strategy_is_unchanged(self, universe: dict[str, Bars]) -> None:
        result = run_backtest(loads(LONG_ONLY), universe)
        assert {t.direction for t in result.trades} == {"long"}

    def test_a_short_only_strategy_is_unchanged(self, universe: dict[str, Bars]) -> None:
        short_only = LONG_ONLY.replace(
            "  entry: rs_rank(60) > 0.66", "  direction: short\n  entry: rs_rank(60) < 0.34"
        )
        result = run_backtest(loads(short_only), universe)
        assert {t.direction for t in result.trades} == {"short"}


def _diverging() -> dict[str, Bars]:
    """One symbol grinding up, one grinding down, one flat."""
    from datetime import UTC, datetime, timedelta

    n = 900
    base = datetime(2015, 1, 1, tzinfo=UTC)
    stamps = np.array(
        [int((base + timedelta(days=i)).timestamp()) for i in range(n)], dtype=np.int64
    )
    rng = np.random.default_rng(5)

    def series(symbol: str, drift: float) -> Bars:
        steps = drift + rng.normal(0.0, 0.006, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        span = np.abs(steps) * close + close * 0.002
        return Bars(
            symbol=symbol,
            timeframe="1D",
            event_time=stamps,
            open=close - steps * close * 0.5,
            high=np.maximum(close, close - steps * close * 0.5) + span,
            low=np.minimum(close, close - steps * close * 0.5) - span,
            close=close,
            volume=np.full(n, 5e6),
        )

    return {
        "SPY": series("SPY", 0.0012),
        "QQQ": series("QQQ", 0.0),
        "IWM": series("IWM", -0.0002),
        "DOWN": series("DOWN", -0.0015),
    }


class TestValidatorAgreesWithTheSchema:
    """Two places checked ``direction`` and only one was updated.

    The model proposed a market-neutral strategy on the first iteration of the
    next campaign and it was thrown away with "direction must be 'long' or
    'short'" -- from a validator nobody remembered was there. A vocabulary is
    only extended when every gate agrees it was.
    """

    def test_the_validator_accepts_every_direction_the_schema_does(self) -> None:
        from typing import get_args

        from aqr.dsl.schema import Direction
        from aqr.dsl.validator import validate

        bodies = {
            "long": LONG_ONLY,
            "short": LONG_ONLY.replace(
                "  entry: rs_rank(60) > 0.66",
                "  direction: short\n  entry: rs_rank(60) < 0.34",
            ),
            "market_neutral": NEUTRAL,
        }
        for direction in get_args(Direction):
            report = validate(loads(bodies[direction]))
            assert report.ok, f"{direction}: {report.errors}"

    def test_the_validator_still_refuses_an_unknown_direction(self) -> None:
        from aqr.dsl.validator import validate

        spec = loads(LONG_ONLY)
        object.__setattr__(spec, "direction", "sideways")
        assert not validate(spec).ok
