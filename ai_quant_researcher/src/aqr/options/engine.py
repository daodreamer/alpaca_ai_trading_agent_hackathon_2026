"""The options backtester — specs/10-options-research.md D1, D2, D3.

The rule this module exists to enforce, and the reason it is not the equity
engine with a different instrument:

    A decision made from the bar that closed *before* the fill session's
    reference bar is filled at the fill session's quotes, and the position is
    then held to expiry and settled against the underlying's close.

Three things make that harder than it sounds on this data, and each one is a
design decision rather than an implementation detail.

**The session grid is not the bar grid.** 82 of 753 chain sessions have no
same-day SPY bar — 2019's snapshots are dated Saturday, and the rest are market
holidays. A session's *reference* bar is the most recent close at or before it,
refused beyond four days; the *decision* bar is the one before that. Deciding
from Friday and filling on a Saturday snapshot that reflects Friday's close
would be trading at the close on the close: no future information, but no time
to act either.

**There is no exit to simulate.** A specific contract is re-quoted on 1–3% of
later sessions (D0), so a stop, a target or a managed roll cannot be priced.
Positions are held to expiry. This is a limit of the data, it is the strongest
constraint in the system, and it is not a parameter.

**Settlement can cross the embargo.** A structure opened on the last research
session expires a month inside the reserved window, and reading the close there
would taint the run that arranged not to see it. So the boundary is checked at
*selection* time and the entry is refused (D3), rather than the run being
truncated afterwards when the decision has already been made.

The money and the model are kept apart (D1a): every number in a trade's P&L
comes from a quote, and the Black-Scholes mark exists only to give the equity
curve the beta and the drawdown that cash accounting destroys.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestResult, Trade
from aqr.data.bars import Bars
from aqr.dsl.expr import evaluate
from aqr.features.engine import FeatureFrame
from aqr.options.chain import ChainIndex, NoSuchContract, Quote, Right, SessionChain
from aqr.options.costs import IBKR_OPTIONS, OptionCostModel
from aqr.options.spec import OptionSpec, StructureSpec
from aqr.options.structure import Leg, Side, Structure
from aqr.seal import EMBARGO_START

__all__ = [
    "MULTIPLIER",
    "OptionBacktestConfig",
    "OptionBacktestResult",
    "OptionTrade",
    "build_structure",
    "run_option_backtest",
]

MULTIPLIER = 100.0
"""Shares per contract. Applied once, here — prices everywhere else are per share."""

DAYS_PER_YEAR = 365.0
"""Calendar days, for the mark's time-to-expiry. An option decays on weekends."""


@dataclass(frozen=True, slots=True)
class OptionBacktestConfig:
    initial_equity: float = 100_000.0
    costs: OptionCostModel = IBKR_OPTIONS
    rate: float = 0.0
    """Risk-free rate for the mark only (D1a). Never touches a realised number."""
    max_reference_staleness_days: int = 4
    """How far back a session may reach for its underlying close before the
    reference price stops being a price and becomes a guess. Four covers every
    holiday and weekend gap in the cache; nothing legitimate needs five."""
    settle_before: date | None = None
    """D3's boundary. ``None`` means the embargo, which is the right default:
    an entry that would settle inside the reserved window is refused."""

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.max_reference_staleness_days < 0:
            raise ValueError("max_reference_staleness_days must be >= 0")

    @property
    def boundary(self) -> date:
        return self.settle_before or EMBARGO_START.date()


@dataclass(frozen=True, slots=True)
class OptionTrade:
    """One structure, opened at a session's quotes and settled at expiry.

    Everything here is a quoted number or arithmetic on one. The mark never
    reaches this type.
    """

    entry_session: date
    """The chain session the fill came from."""
    entry_reference: date
    """The trading day the cash flow lands on -- the session itself, or the most
    recent close before it when the session is not a trading day."""
    decision_bar: date
    """The close the entry condition was evaluated on. Strictly earlier than
    ``entry_reference``, which is the whole of D2."""
    expiration: date
    kind: str
    contracts: int
    strikes: tuple[float, ...]
    entry_cash: float
    """Per share, signed. Positive is a credit."""
    settlement_cash: float
    """Per share, signed."""
    max_loss: float
    """Per share, positive. The denominator sizing divided by."""
    crossed_spread: float
    """Per share: what crossing cost against the mid, reported as slippage."""
    fees: float
    """Dollars, entry commission plus any settlement fee."""
    underlying_at_entry: float
    underlying_at_expiry: float
    mae: float
    """Worst marked P&L while open, as a fraction of capital at risk."""
    mfe: float

    @property
    def pnl_per_share(self) -> float:
        return self.entry_cash + self.settlement_cash

    @property
    def gross_pnl(self) -> float:
        return self.pnl_per_share * MULTIPLIER * self.contracts

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def capital_at_risk(self) -> float:
        return self.max_loss * MULTIPLIER * self.contracts

    @property
    def return_on_risk(self) -> float:
        return self.net_pnl / self.capital_at_risk if self.capital_at_risk else 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "return_on_risk": self.return_on_risk,
        }


@dataclass(slots=True)
class OptionBacktestResult(BacktestResult):
    """A :class:`BacktestResult`, so every downstream consumer needs no change.

    ``PortfolioResult`` is the precedent. ``trades`` carries the generic
    :class:`Trade` view that ``metrics`` and the walk-forward read;
    ``option_trades`` carries what actually happened, which a generic trade
    cannot express -- four legs, a width, a settlement price.
    """

    option_trades: list[OptionTrade] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)
    """Why an entry did not happen, in order. A rule that never trades and a
    rule whose every selection was refused look identical in the metrics, and
    telling them apart is most of debugging a spec."""


# --------------------------------------------------------------------------- #
# Building a structure from a session's chain
# --------------------------------------------------------------------------- #

_ANCHOR_SIDE: dict[str, tuple[Right, Side]] = {
    "long_call": ("call", "buy"),
    "long_put": ("put", "buy"),
    "put_credit_spread": ("put", "sell"),
    "call_credit_spread": ("call", "sell"),
    "put_debit_spread": ("put", "buy"),
    "call_debit_spread": ("call", "buy"),
}
"""Which leg the delta names, and whether it is bought or sold.

Derived from the kind rather than declared in the spec: letting a rule say it
sells the anchor of a debit spread would let it name a structure whose maximum
loss is not the one its kind implies, and sizing would divide by the wrong
number.
"""


def build_structure(spec: StructureSpec, chain: SessionChain) -> Structure:
    """Resolve a rule into an actual position on one session's ladder.

    Raises :class:`NoSuchContract` when the ladder does not carry what the rule
    named, and ``ValueError`` when the legs it does carry cannot form a valid
    structure. Both mean "no entry today"; neither is repaired.
    """
    if spec.type == "iron_condor":
        return Structure(
            kind="iron_condor",
            legs=(
                *_spread_legs(chain, spec, right="put"),
                *_spread_legs(chain, spec, right="call"),
            ),
        )

    right, side = _ANCHOR_SIDE[spec.type]
    anchor = chain.select(
        right=right,
        dte_target=spec.dte.target,
        dte_tolerance=spec.dte.tolerance,
        delta_target=spec.anchor.delta,
        delta_tolerance=spec.anchor.tolerance,
        sellable=side == "sell",
    )
    legs = [Leg(quote=anchor, side=side)]
    if spec.width_delta or spec.width_points:
        wing = _wing(chain, anchor, spec.width_delta, spec.width_points, spec.anchor.tolerance)
        other: Side = "buy" if side == "sell" else "sell"
        if other == "sell" and not wing.sellable:
            raise NoSuchContract(
                f"{chain.session}: the {wing.strike:g} {right} wing has a zero bid"
            )
        legs.append(Leg(quote=wing, side=other))
    return Structure(kind=spec.type, legs=tuple(legs))


def _spread_legs(chain: SessionChain, spec: StructureSpec, *, right: Right) -> tuple[Leg, Leg]:
    """One credit spread of a condor: sell the anchor, buy the wing."""
    anchor = spec.anchor if right == "put" else spec.call_side_anchor
    by_delta = spec.width_delta if right == "put" else spec.call_side_width_delta
    by_points = spec.width_points if right == "put" else spec.call_side_width_points
    short = chain.select(
        right=right,
        dte_target=spec.dte.target,
        dte_tolerance=spec.dte.tolerance,
        delta_target=anchor.delta,
        delta_tolerance=anchor.tolerance,
        sellable=True,
    )
    wing = _wing(chain, short, by_delta, by_points, anchor.tolerance)
    return Leg(quote=short, side="sell"), Leg(quote=wing, side="buy")


def _wing(
    chain: SessionChain,
    short: Quote,
    by_delta: float | None,
    by_points: float | None,
    tolerance: float,
) -> Quote:
    """The protective leg, by whichever form the spec named.

    ``width_delta`` is the one that works on this cache: against a 16-delta
    short put it resolves on 98% of sessions where an exact 10-point width
    resolves on 23%. ``width_points`` stays available for a rule that means a
    literal distance, and it refuses rather than widening quietly.
    """
    if by_delta is not None:
        return chain.select_wing_by_delta(
            short, delta_target=by_delta, delta_tolerance=tolerance
        )
    assert by_points is not None  # StructureSpec requires exactly one
    return chain.select_wing(short, width_points=by_points)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Open:
    """A position between its entry and its expiry."""

    structure: Structure
    contracts: int
    entry_index: int
    expiry_index: int
    entry_session: date
    entry_reference: date
    decision_bar: date
    entry_fee: float


def run_option_backtest(
    spec: OptionSpec,
    chain: ChainIndex,
    underlying: Bars,
    config: OptionBacktestConfig | None = None,
) -> OptionBacktestResult:
    """Simulate ``spec`` over ``chain``, settling against ``underlying``."""
    config = config or OptionBacktestConfig()
    days = [datetime.fromtimestamp(int(t), tz=UTC).date() for t in underlying.event_time]
    if not days:
        raise ValueError(f"{underlying.symbol}: no bars to settle against")
    closes = np.asarray(underlying.close, dtype=np.float64)
    day_index = {day: i for i, day in enumerate(days)}

    frame = FeatureFrame(underlying)
    warmup = frame.warmup(spec.features())
    signal = np.asarray(evaluate(spec.entry_ast, frame), dtype=bool)
    signal[:warmup] = False

    opened: list[_Open] = []
    skipped: list[str] = []
    realised = config.initial_equity
    last_entry_at: int | None = None
    settled_by_index: dict[int, float] = {}

    for position, session in enumerate(chain.sessions):
        reference = _reference_index(days, session, config.max_reference_staleness_days)
        if reference is None:
            skipped.append(f"{session}: no underlying close within "
                           f"{config.max_reference_staleness_days} days")
            continue
        if reference == 0:
            continue  # nothing closed before it; the decision bar would not exist
        decision = reference - 1
        if not bool(signal[decision]):
            continue
        if last_entry_at is not None and position - last_entry_at < (
            spec.cadence.min_sessions_between_entries
        ):
            continue
        live = sum(1 for open_ in opened if days[open_.expiry_index] > session)
        if live >= spec.sizing.max_concurrent:
            continue

        try:
            structure = build_structure(spec.structure, chain[session])
        except (NoSuchContract, ValueError) as exc:
            skipped.append(f"{session}: {exc}")
            continue

        if structure.expiration >= config.boundary:
            skipped.append(
                f"{session}: refused, the structure expires {structure.expiration}, on or "
                f"after {config.boundary} -- settling it would read the reserved window"
            )
            continue
        expiry_index = day_index.get(structure.expiration)
        if expiry_index is None:
            skipped.append(
                f"{session}: no underlying close on expiry {structure.expiration}"
            )
            continue

        # Realised equity only: sizing must not lever up on unrealised marks,
        # which are a model's opinion (D1a) and not money that has arrived.
        realised = config.initial_equity + sum(
            pnl for at, pnl in settled_by_index.items() if at <= reference
        )
        risk_per_contract = structure.max_loss * MULTIPLIER
        contracts = int((realised * spec.sizing.risk_per_trade) // risk_per_contract)
        if contracts < 1:
            skipped.append(
                f"{session}: {spec.sizing.risk_per_trade:.1%} of {realised:,.0f} does not "
                f"cover one contract at {risk_per_contract:,.0f} of risk"
            )
            continue

        fee = config.costs.entry_fee(legs=len(structure.legs), contracts=contracts)
        opened.append(
            _Open(
                structure=structure,
                contracts=contracts,
                entry_index=reference,
                expiry_index=expiry_index,
                entry_session=session,
                entry_reference=days[reference],
                decision_bar=days[decision],
                entry_fee=fee,
            )
        )
        settled = _settlement_pnl(opened[-1], closes[expiry_index], config.costs)
        settled_by_index[expiry_index] = settled_by_index.get(expiry_index, 0.0) + settled
        last_entry_at = position

    equity, option_trades = _curve_and_trades(opened, days, closes, config)
    trades = [_as_trade(spec, option, underlying, days) for option in option_trades]
    return OptionBacktestResult(
        spec_fingerprint=spec.fingerprint(),
        strategy=spec.name,
        symbols=(spec.underlying,),
        timeline=np.asarray(underlying.event_time, dtype=np.int64),
        equity=equity,
        trades=trades,
        warmup_bars=warmup,
        option_trades=option_trades,
        skipped_reasons=skipped,
    )


def _reference_index(days: list[date], session: date, staleness: int) -> int | None:
    """The most recent close at or before ``session``, or ``None`` if too stale.

    82 of 753 sessions in the cache take this path: 2019's Saturday snapshots
    and the market holidays. Every one of them resolves within four days.
    """
    at = bisect_right(days, session) - 1
    if at < 0:
        return None
    return at if (session - days[at]).days <= staleness else None


def _settlement_pnl(open_: _Open, close: float, costs: OptionCostModel) -> float:
    value = open_.structure.settlement_value(float(close))
    itm = sum(
        1
        for leg in open_.structure.legs
        if (leg.quote.right == "call" and close > leg.quote.strike)
        or (leg.quote.right == "put" and close < leg.quote.strike)
    )
    fee = costs.settlement_fee(itm_legs=itm, contracts=open_.contracts)
    entry = open_.structure.entry_cash * MULTIPLIER * open_.contracts
    return entry + value * MULTIPLIER * open_.contracts - open_.entry_fee - fee


def _curve_and_trades(
    opened: list[_Open],
    days: list[date],
    closes: np.ndarray,
    config: OptionBacktestConfig,
) -> tuple[np.ndarray, list[OptionTrade]]:
    """Cash on the trading-day grid, plus the marked value of what is open.

    Cash steps at the entry and at the settlement, both from quoted numbers. The
    mark fills in between, and at expiry it equals the settlement exactly, so
    the curve does not jump on the settlement date for a reason nobody can point
    at.
    """
    cash = np.full(len(days), config.initial_equity, dtype=np.float64)
    marked = np.zeros(len(days), dtype=np.float64)
    trades: list[OptionTrade] = []

    for open_ in opened:
        structure = open_.structure
        scale = MULTIPLIER * open_.contracts
        entry_cash = structure.entry_cash * scale - open_.entry_fee
        cash[open_.entry_index :] += entry_cash

        close_at_expiry = float(closes[open_.expiry_index])
        settlement = structure.settlement_value(close_at_expiry)
        itm = sum(
            1
            for leg in structure.legs
            if (leg.quote.right == "call" and close_at_expiry > leg.quote.strike)
            or (leg.quote.right == "put" and close_at_expiry < leg.quote.strike)
        )
        exit_fee = config.costs.settlement_fee(itm_legs=itm, contracts=open_.contracts)
        cash[open_.expiry_index :] += settlement * scale - exit_fee

        at_risk = structure.max_loss * scale
        worst = 0.0
        best = 0.0
        for i in range(open_.entry_index, open_.expiry_index):
            years = (days[open_.expiry_index] - days[i]).days / DAYS_PER_YEAR
            value = structure.mark(spot=float(closes[i]), years=years, rate=config.rate)
            marked[i] += value * scale
            unrealised = (structure.entry_cash + value) * scale
            worst = min(worst, unrealised)
            best = max(best, unrealised)

        crossed = sum(
            (leg.quote.ask - leg.quote.bid) / 2.0 for leg in structure.legs
        )
        trades.append(
            OptionTrade(
                entry_session=open_.entry_session,
                entry_reference=open_.entry_reference,
                decision_bar=open_.decision_bar,
                expiration=days[open_.expiry_index],
                kind=structure.kind,
                contracts=open_.contracts,
                strikes=structure.strikes,
                entry_cash=structure.entry_cash,
                settlement_cash=settlement,
                max_loss=structure.max_loss,
                crossed_spread=crossed,
                fees=open_.entry_fee + exit_fee,
                underlying_at_entry=float(closes[open_.entry_index]),
                underlying_at_expiry=close_at_expiry,
                mae=worst / at_risk if at_risk else 0.0,
                mfe=best / at_risk if at_risk else 0.0,
            )
        )
    return cash + marked, trades


def _as_trade(spec: OptionSpec, trade: OptionTrade, underlying: Bars, days: list[date]) -> Trade:
    """The generic view ``metrics`` and the walk-forward read.

    ``entry_price`` is the **capital at risk per share**, not the premium. A
    credit spread's premium is a terrible denominator -- 1.00 collected against
    9.00 risked would report a 100% return on a maximum loss -- and every ratio
    in ``metrics`` divides by this number.
    """
    quantity = MULTIPLIER * trade.contracts
    entry_price = trade.max_loss
    exit_price = trade.max_loss + trade.pnl_per_share
    entry_at = _stamp(underlying, days.index(trade.entry_reference))
    exit_at = _stamp(underlying, days.index(trade.expiration))
    return Trade(
        symbol=spec.underlying,
        direction="short" if trade.entry_cash > 0 else "long",
        entry_time=entry_at,
        entry_price=entry_price,
        exit_time=exit_at,
        exit_price=exit_price,
        quantity=quantity,
        gross_pnl=trade.gross_pnl,
        fees=trade.fees,
        slippage=trade.crossed_spread * quantity,
        bars_held=days.index(trade.expiration) - days.index(trade.entry_reference),
        exit_reason="expiry",
        mae=trade.mae,
        mfe=trade.mfe,
    )


def _stamp(underlying: Bars, index: int) -> int:
    return int(underlying.event_time[index])
