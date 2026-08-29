"""The backtester.

The one rule this module exists to enforce:

    A decision made from bar ``t`` is filled at bar ``t + 1``'s open.

Signals are computed from the close of bar ``t``. Nothing that happens at
``t + 1`` or later can influence them, because features are vectorised over the
whole series up front and then read only at index ``t``. The fill happens at the
*next* bar's open with adverse slippage. There is no configuration flag that
relaxes this, which is the point -- a look-ahead switch is a look-ahead bug with
a rationale attached.

Intrabar exits follow the pessimistic convention: if a bar's range contains both
the stop and the target, the stop is taken. Without knowing the tick sequence,
assuming the favourable one is how backtests learn to lie.

Gaps are honoured. If a bar opens through the stop, the fill is the open, not
the stop price. Stops are not guarantees, and a backtest that pretends otherwise
under-reports exactly the risk it was built to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from aqr.backtest.costs import CostModel
from aqr.data.bars import Bars, bars_per_year
from aqr.dsl.expr import evaluate
from aqr.dsl.schema import StrategySpec
from aqr.features.cross_section import CrossSection
from aqr.features.engine import FeatureFrame, FeatureKey

__all__ = ["BacktestConfig", "BacktestResult", "Trade", "run_backtest"]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel)
    max_portfolio_drawdown: float = 0.25
    """Hard stop for the whole run. Breaching it flattens and halts trading --
    a real allocator would pull the strategy, so the backtest must too."""
    allow_fractional_shares: bool = False

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not 0 < self.max_portfolio_drawdown <= 1:
            raise ValueError("max_portfolio_drawdown must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    direction: str
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage: float
    bars_held: int
    exit_reason: str
    mae: float
    """Maximum adverse excursion, as a fraction of entry price."""
    mfe: float
    """Maximum favourable excursion, as a fraction of entry price."""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        notional = self.entry_price * self.quantity
        return self.net_pnl / notional if notional else 0.0

    @property
    def entry_dt(self) -> datetime:
        return datetime.fromtimestamp(self.entry_time, tz=UTC)

    @property
    def exit_dt(self) -> datetime:
        return datetime.fromtimestamp(self.exit_time, tz=UTC)


@dataclass(slots=True)
class _Position:
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    entry_time: int
    entry_step: int
    stop: float
    target: float | None
    entry_fees: float
    entry_slippage: float
    worst: float
    best: float


@dataclass(slots=True)
class BacktestResult:
    spec_fingerprint: str
    strategy: str
    symbols: tuple[str, ...]
    timeline: np.ndarray
    equity: np.ndarray
    trades: list[Trade]
    warmup_bars: int
    halted: bool = False
    halt_reason: str = ""

    @property
    def initial_equity(self) -> float:
        return float(self.equity[0]) if self.equity.size else 0.0

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1]) if self.equity.size else 0.0

    @property
    def timestamps(self) -> list[datetime]:
        return [datetime.fromtimestamp(int(t), tz=UTC) for t in self.timeline]


# --------------------------------------------------------------------------- #
# Signal preparation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _SymbolPlan:
    """Everything about one symbol, precomputed and index-aligned."""

    bars: Bars
    entry: np.ndarray
    short_entry: np.ndarray | None
    exit_signal: np.ndarray | None
    stop_distance: np.ndarray
    step_of_bar: np.ndarray
    bar_of_step: dict[int, int]
    warmup: int


def _prepare(
    spec: StrategySpec,
    bars: Bars,
    timeline: np.ndarray,
    cross_section: CrossSection | None = None,
) -> _SymbolPlan:
    frame = FeatureFrame(bars, cross_section)
    warmup = frame.warmup(spec.features())

    entry = np.asarray(evaluate(spec.entry_ast, frame), dtype=bool)
    if spec.regime_ast is not None:
        entry &= np.asarray(evaluate(spec.regime_ast, frame), dtype=bool)
    entry[:warmup] = False

    short_entry: np.ndarray | None = None
    if spec.short_entry_ast is not None:
        short_entry = np.asarray(evaluate(spec.short_entry_ast, frame), dtype=bool)
        if spec.regime_ast is not None:
            # The regime gates *all* entries, both legs. A market-neutral rule
            # whose short leg ignored the regime filter would be two strategies
            # sharing a name.
            short_entry &= np.asarray(evaluate(spec.regime_ast, frame), dtype=bool)
        short_entry[:warmup] = False
        # A bar that satisfies both legs is not a signal, it is a contradiction.
        # Taking one arbitrarily would make the result depend on evaluation
        # order; taking neither is the only reading the rule supports.
        both = entry & short_entry
        entry = entry & ~both
        short_entry = short_entry & ~both

    exit_signal: np.ndarray | None = None
    if spec.exit_ast is not None:
        exit_signal = np.asarray(evaluate(spec.exit_ast, frame), dtype=bool)
        exit_signal[:warmup] = False

    stop = spec.exit.stop_loss
    if stop.type == "atr":
        atr = frame.get(FeatureKey("atr", (float(stop.period),)))
        distance = atr * stop.multiplier
    else:
        distance = np.asarray(bars.close, dtype=np.float64) * stop.multiplier
    distance = np.where(np.isnan(distance) | (distance <= 0), np.nan, distance)

    step_of_bar = np.searchsorted(timeline, bars.event_time)
    bar_of_step = {int(step): i for i, step in enumerate(step_of_bar)}
    return _SymbolPlan(
        bars=bars,
        entry=entry,
        short_entry=short_entry,
        exit_signal=exit_signal,
        stop_distance=distance,
        step_of_bar=step_of_bar,
        bar_of_step=bar_of_step,
        warmup=warmup,
    )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def run_backtest(
    spec: StrategySpec,
    data: dict[str, Bars] | Bars,
    config: BacktestConfig | None = None,
    *,
    peers: dict[str, Bars] | None = None,
) -> BacktestResult:
    """Simulate ``spec`` over ``data``.

    ``data`` may be a single :class:`Bars` or a mapping of symbol to bars. Only
    symbols named in ``spec.universe`` are traded; passing extras is not an
    error, so one loaded dataset can serve many strategies.

    ``peers`` is the market that peer-relative features rank against. It
    defaults to the traded universe, which is what makes a result reproducible
    from the spec alone -- whatever else happens to be loaded cannot change
    the answer. Pass it explicitly to hold the market definition fixed while
    trading a subset of it, which is the only way to ask a cross-sectional
    rule whether its edge lives on one particular ticker.
    """
    config = config or BacktestConfig()
    if isinstance(data, Bars):
        data = {data.symbol: data}
    universe = [s for s in spec.universe.symbols if s in data]
    missing = [s for s in spec.universe.symbols if s not in data]
    if not universe:
        raise ValueError(
            f"{spec.name}: no data for any symbol in the universe {list(spec.universe.symbols)}"
        )
    if missing:
        # Silently trading a subset would make results incomparable across runs.
        raise ValueError(f"{spec.name}: missing bars for {missing}")

    timeline = np.unique(np.concatenate([data[s].event_time for s in universe]))
    # Built from the traded universe, once, and shared by every symbol's
    # frame. A peer set that changed between symbols would mean each name was
    # ranked against a different market.
    cross_section = CrossSection(peers if peers else {s: data[s] for s in universe})
    plans = {s: _prepare(spec, data[s], timeline, cross_section) for s in universe}
    warmup = max(p.warmup for p in plans.values())

    # Direction belongs to the position, not the run. A market-neutral strategy
    # holds both at once, and every place that used a single run-level `long`
    # -- sizing, stop placement, intrabar exit resolution, borrow cost -- now
    # asks the position. A stop above the entry is right for a short and
    # catastrophic for a long.
    default_long = spec.direction != "short"
    costs = config.costs

    cash = config.initial_equity
    equity_curve = np.empty(timeline.size, dtype=np.float64)
    positions: dict[str, _Position] = {}
    # (symbol, bar index that produced the signal, direction)
    pending: list[tuple[str, int, str]] = []
    trades: list[Trade] = []
    peak = config.initial_equity
    halted = False
    halt_reason = ""

    last_price: dict[str, float] = {}

    def mark(step: int) -> float:
        """Equity = cash plus the signed market value of what is held.

        A symbol with no bar at this step is carried at its last known close;
        marking a stale position to its entry price would silently reverse any
        open profit or loss on every gap in the calendar.
        """
        value = cash
        for pos in positions.values():
            plan = plans[pos.symbol]
            i = plan.bar_of_step.get(step)
            if i is not None:
                price = float(plan.bars.close[i])
                last_price[pos.symbol] = price
            else:
                price = last_price.get(pos.symbol, pos.entry_price)
            value += (1.0 if pos.direction == "long" else -1.0) * pos.quantity * price
        return value

    for step in range(timeline.size):
        # --- 1. exits, so a freed slot can be reused on the same bar --------- #
        for symbol in list(positions):
            plan = plans[symbol]
            i = plan.bar_of_step.get(step)
            if i is None:
                continue
            pos = positions[symbol]
            if i <= pos.entry_step:
                continue  # never exit on the entry bar; we filled at its open
            pos_long = pos.direction == "long"

            open_ = float(plan.bars.open[i])
            high = float(plan.bars.high[i])
            low = float(plan.bars.low[i])
            close = float(plan.bars.close[i])

            excursion_low = (low - pos.entry_price) / pos.entry_price
            excursion_high = (high - pos.entry_price) / pos.entry_price
            pos.worst = min(pos.worst, excursion_low if pos_long else -excursion_high)
            pos.best = max(pos.best, excursion_high if pos_long else -excursion_low)

            # An exit decided at the previous close, and the holding limit,
            # both execute at this open -- which happens before any intrabar
            # stop could trigger. Checking them first keeps the reported reason
            # honest even when the same bar would also have hit the stop.
            held = i - pos.entry_step
            quoted: float | None = None
            reason = ""
            if plan.exit_signal is not None and bool(plan.exit_signal[i - 1]):
                quoted, reason = open_, "signal"
            elif held >= spec.exit.max_holding_bars:
                quoted, reason = open_, "max_holding"
            else:
                quoted, reason = _exit_quote(pos, pos_long, open_, high, low)

            if quoted is None:
                # Nothing in the rule closes the position on this bar. If it is
                # this symbol's *last* bar while the timeline continues, the
                # position has to be closed anyway: after it, every check above
                # is skipped -- ``bar_of_step`` has no entry -- so the stop, the
                # target and the holding limit can never fire again. The
                # position would sit in the book holding a slot it can never
                # give up, and a book limited to a few names would leak its
                # capacity toward exactly the names that delist.
                #
                # ``step != timeline.size - 1`` is the end-of-data guard: on the
                # last step of the run every symbol is on its final bar, and
                # liquidating there would relabel every ordinary open position
                # as a delisting.
                if i == len(plan.bars) - 1 and step != timeline.size - 1:
                    quoted, reason = close, "delisted"
                else:
                    continue

            cash, trade = _close(pos, plan, i, float(quoted), reason, pos_long, costs, cash)
            trades.append(trade)
            last_price[symbol] = close
            del positions[symbol]
            # A queued order for a symbol that has stopped trading can never
            # fill, and carrying it to the end of the run leaves a dead symbol
            # in the pending list.
            if reason == "delisted":
                pending = [entry for entry in pending if entry[0] != symbol]

        # --- 2. fills for orders queued at the previous bar ------------------ #
        still_pending: list[tuple[str, int, str]] = []
        for symbol, signal_bar, direction in pending:
            plan = plans[symbol]
            i = plan.bar_of_step.get(step)
            if i is None or i != signal_bar + 1:
                # The next bar for this symbol has not arrived yet. Hold the
                # order rather than filling it late at a stale price.
                if i is None:
                    still_pending.append((symbol, signal_bar, direction))
                continue
            if halted or symbol in positions or len(positions) >= spec.max_positions:
                continue

            distance = float(plan.stop_distance[signal_bar])
            if not np.isfinite(distance) or distance <= 0:
                continue

            quoted = float(plan.bars.open[i])
            if quoted <= 0:
                continue
            take_long = direction == "long"
            sign = 1.0 if take_long else -1.0
            fill = costs.fill_price(quoted, "buy" if take_long else "sell")

            # Sizing must not read this bar's close: we are standing at its
            # open. The last settled equity is the most recent knowable number.
            equity_ref = float(equity_curve[step - 1]) if step else config.initial_equity
            quantity = _size(
                spec, config, equity_ref, cash, fill, distance, plan, i, take_long, costs
            )
            if quantity <= 0:
                continue

            fee = costs.commission(quantity, fill)
            slip = costs.slippage_cost(quantity, quoted)
            cash -= sign * quantity * fill + fee
            stop_price = fill - sign * distance
            target = None
            tp = spec.exit.take_profit
            if tp.type == "risk_reward":
                target = fill + sign * distance * tp.ratio
            elif tp.type == "percent":
                target = fill * (1.0 + sign * tp.ratio)

            positions[symbol] = _Position(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=fill,
                entry_time=int(plan.bars.event_time[i]),
                entry_step=i,
                stop=stop_price,
                target=target,
                entry_fees=fee,
                entry_slippage=slip,
                worst=0.0,
                best=0.0,
            )
        pending = still_pending

        # --- 3. decisions from this bar's close, filled next bar ------------- #
        if not halted:
            for symbol, plan in plans.items():
                i = plan.bar_of_step.get(step)
                if i is None or i + 1 >= len(plan.bars):
                    continue
                if symbol in positions or any(p[0] == symbol for p in pending):
                    continue
                if bool(plan.entry[i]):
                    pending.append((symbol, i, "long" if default_long else "short"))
                elif plan.short_entry is not None and bool(plan.short_entry[i]):
                    pending.append((symbol, i, "short"))

        # --- 4. mark to market and enforce the portfolio drawdown limit ------ #
        equity = mark(step)
        equity_curve[step] = equity
        peak = max(peak, equity)
        if not halted and peak > 0 and (peak - equity) / peak > config.max_portfolio_drawdown:
            halted = True
            halt_reason = (
                f"portfolio drawdown exceeded {config.max_portfolio_drawdown:.0%} at step {step}"
            )
            pending = []
            for symbol in list(positions):
                plan = plans[symbol]
                i = plan.bar_of_step.get(step)
                if i is None:
                    continue
                cash, trade = _close(
                    positions[symbol],
                    plan,
                    i,
                    float(plan.bars.close[i]),
                    "halt",
                    positions[symbol].direction == "long",
                    costs,
                    cash,
                )
                trades.append(trade)
                del positions[symbol]
            equity_curve[step] = mark(step)

    # Close whatever is still open at the final bar, so the equity curve and the
    # trade list describe the same run.
    last = timeline.size - 1
    for symbol in list(positions):
        plan = plans[symbol]
        i = plan.bar_of_step.get(last, len(plan.bars) - 1)
        cash, trade = _close(
            positions[symbol],
            plan,
            i,
            float(plan.bars.close[i]),
            "end_of_data",
            positions[symbol].direction == "long",
            costs,
            cash,
        )
        trades.append(trade)
        del positions[symbol]
    if timeline.size:
        equity_curve[last] = cash

    trades.sort(key=lambda t: (t.exit_time, t.symbol))
    return BacktestResult(
        spec_fingerprint=spec.fingerprint(),
        strategy=spec.name,
        symbols=tuple(universe),
        timeline=timeline,
        equity=equity_curve,
        trades=trades,
        warmup_bars=warmup,
        halted=halted,
        halt_reason=halt_reason,
    )


def _exit_quote(
    pos: _Position, long: bool, open_: float, high: float, low: float
) -> tuple[float | None, str]:
    """Stop and target resolution for one bar, pessimistic by construction."""
    if long:
        gapped_through_stop = open_ <= pos.stop
        if gapped_through_stop:
            return open_, "stop_gap"
        if low <= pos.stop:
            return pos.stop, "stop"
        if pos.target is not None:
            if open_ >= pos.target:
                return open_, "target_gap"
            if high >= pos.target:
                return pos.target, "target"
        return None, ""

    if open_ >= pos.stop:
        return open_, "stop_gap"
    if high >= pos.stop:
        return pos.stop, "stop"
    if pos.target is not None:
        if open_ <= pos.target:
            return open_, "target_gap"
        if low <= pos.target:
            return pos.target, "target"
    return None, ""


def _close(
    pos: _Position,
    plan: _SymbolPlan,
    bar: int,
    quoted: float,
    reason: str,
    long: bool,
    costs: CostModel,
    cash: float,
) -> tuple[float, Trade]:
    fill = costs.fill_price(quoted, "sell" if long else "buy")
    fee = costs.commission(pos.quantity, fill)
    slip = costs.slippage_cost(pos.quantity, quoted)
    sign = 1.0 if long else -1.0
    gross = sign * pos.quantity * (fill - pos.entry_price)

    # A short borrows the stock, and borrowing is not free. Charged on the entry
    # notional over the trading time actually held: a short backtest that skips
    # this reports a return nobody could have earned.
    held_bars = max(bar - pos.entry_step, 0)
    if not long and held_bars:
        years = held_bars / bars_per_year(plan.bars.timeframe)
        fee += costs.borrow_cost(pos.quantity, pos.entry_price, years)

    cash = cash + sign * pos.quantity * fill - fee
    trade = Trade(
        symbol=pos.symbol,
        direction=pos.direction,
        entry_time=pos.entry_time,
        entry_price=pos.entry_price,
        exit_time=int(plan.bars.event_time[bar]),
        exit_price=fill,
        quantity=pos.quantity,
        gross_pnl=gross,
        fees=pos.entry_fees + fee,
        slippage=pos.entry_slippage + slip,
        bars_held=bar - pos.entry_step,
        exit_reason=reason,
        mae=pos.worst,
        mfe=pos.best,
    )
    return cash, trade


def _size(
    spec: StrategySpec,
    config: BacktestConfig,
    equity: float,
    cash: float,
    fill: float,
    stop_distance: float,
    plan: _SymbolPlan,
    bar: int,
    long: bool,
    costs: CostModel,
) -> float:
    """Risk-first sizing, then every ceiling that could shrink it.

    The order matters. Risk sets the intent; concentration, cash and liquidity
    can only reduce it. Nothing here is allowed to increase size.
    """
    if equity <= 0 or fill <= 0 or stop_distance <= 0:
        return 0.0

    sizing = spec.sizing
    if sizing.type == "fixed_notional":
        quantity = (equity * sizing.max_position_pct) / fill
    else:
        quantity = (equity * sizing.risk_per_trade) / stop_distance

    quantity = min(quantity, (equity * sizing.max_position_pct) / fill)
    if long:
        # Shorts release cash rather than consuming it; the concentration cap
        # above is what bounds them.
        quantity = min(quantity, max(cash, 0.0) / fill)
    quantity = min(quantity, costs.max_quantity(float(plan.bars.volume[bar])))

    if not config.allow_fractional_shares:
        quantity = float(np.floor(quantity))
    return max(quantity, 0.0)
