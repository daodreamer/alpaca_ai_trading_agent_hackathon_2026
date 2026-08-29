"""Transaction costs.

Section 10 of the architecture lists what a backtest must not pretend away.
This module is where the pretending is prevented. The defaults are deliberately
pessimistic for liquid US equities -- a strategy that only works at zero cost
should die in research, not in production.

Every cost is charged *against* the trade: slippage widens the price you pay and
narrows the price you receive, never the reverse.

**The per-order floor makes cost a function of breadth, and that has to be
visible.** Two of these charges scale with notional (``spread_bps``,
``slippage_bps``) and two do not (``commission_per_share``, ``min_commission``).
The ones that do not are why the same rule, on the same bars, under the same
model, is priced differently depending on how many names it holds and how large
the account is:

```
the promoted strategy, 10 core + ~100 sleeve names, rebalanced every 5 sessions
      equity     CAGR   Sharpe   frictionless   Sharpe retained
     100,000   15.94%     1.15           1.58              73%
   1,000,000   20.50%     1.43           1.58              91%
  10,000,000   20.85%     1.46           1.58              92%
```

Nothing about the strategy changed across those rows. At $100k a sleeve position
is $192, and a $1.00 order floor on $192 is **52 basis points** -- against the
3bp the spread and slippage charge. The floor bound on 1262 of 2317 core
round-trips. Cost retention is a *fatal* gate in the evaluator, so this is not a
cosmetic difference: it decides verdicts, and it decides them on
``BacktestConfig.initial_equity``, a number nobody thinks of as a cost
parameter.

That is not a reason to make the model cheaper. A trader with $100k and a
hundred-name book at a broker that charges a dollar an order really would pay
52bp, and a model that hid it would be lying in the flattering direction. It is
a reason to (a) name which broker's schedule is being modelled rather than
leaving it implied, (b) let a caller ask what an order of a given size actually
costs, and (c) record the model alongside the verdict it produced.

:data:`IBKR_FIXED` is the historical default, preserved exactly, so every verdict
already in the registry stays reproducible. :data:`ALPACA_EQUITIES` is the
venue this project's bars come from and whose paper account the target book is
built for; it is commission-free, so only the turnover-scaled charges remain.
Neither is "the right one" -- which one is right depends on where the book is
actually executed, and that is not this project's to decide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "ALPACA_EQUITIES",
    "IBKR_FIXED",
    "PRESETS",
    "ZERO_COST",
    "CostModel",
    "OrderCost",
    "preset",
]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Commission, spread and slippage.

    ``commission_per_share``  cents-per-share broker fee.
    ``commission_bps``        proportional fee, in basis points of notional.
    ``min_commission``        per-order floor.
    ``spread_bps``            half-spread paid on entry and on exit.
    ``slippage_bps``          market impact beyond the spread.
    ``participation_cap``     max fraction of a bar's volume one order may take.
                              Orders above it are truncated, which is how a
                              backtest learns that a strategy does not scale.
    ``borrow_bps_per_year``   stock-loan fee charged on short positions, pro-rated
                              over the holding period. 50bp is a reasonable
                              easy-to-borrow large-cap rate; hard-to-borrow names
                              run to hundreds or thousands of basis points. A
                              short backtest that omits this is not a backtest of
                              a short.
    """

    commission_per_share: float = 0.005
    commission_bps: float = 0.0
    min_commission: float = 1.0
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    participation_cap: float = 0.05
    borrow_bps_per_year: float = 50.0

    @property
    def _adverse(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10_000.0

    def fill_price(self, quoted: float, side: str) -> float:
        """The price actually paid or received.

        ``side`` is the direction of the order: ``buy`` pays up, ``sell`` gives
        up. This is applied to every fill including stop-outs, because a stop is
        a market order and market orders slip.
        """
        if quoted <= 0:
            raise ValueError(f"quoted price must be positive, got {quoted}")
        if side == "buy":
            return quoted * (1.0 + self._adverse)
        if side == "sell":
            return quoted * (1.0 - self._adverse)
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def commission(self, quantity: float, price: float) -> float:
        """Broker fee for one order. Zero quantity costs nothing."""
        if quantity <= 0:
            return 0.0
        fee = quantity * self.commission_per_share
        fee += quantity * price * self.commission_bps / 10_000.0
        return max(fee, self.min_commission)

    def max_quantity(self, bar_volume: float) -> float:
        """Liquidity ceiling for one order against one bar's volume."""
        if bar_volume <= 0:
            return 0.0
        return bar_volume * self.participation_cap

    def slippage_cost(self, quantity: float, quoted: float) -> float:
        """Slippage in currency, recorded separately so it can be attributed."""
        return abs(quantity) * quoted * self._adverse

    def borrow_cost(self, quantity: float, price: float, years: float) -> float:
        """Stock-loan fee for holding a short for ``years`` of trading time."""
        if quantity <= 0 or price <= 0 or years <= 0:
            return 0.0
        return abs(quantity) * price * (self.borrow_bps_per_year / 10_000.0) * years

    # -- calibration -------------------------------------------------------

    def price_order(self, notional: float, price: float) -> OrderCost:
        """What one order of this size actually costs, broken into its parts.

        The question the model could not answer before, and the one that decides
        whether a schedule fits a book: a charge that is 3bp on a $30,000 order
        is 52bp on a $192 one, and only the second number tells you the strategy
        is uneconomic at that size.
        """
        if notional <= 0 or price <= 0:
            return OrderCost(notional=max(notional, 0.0), shares=0.0, fee=0.0, adverse=0.0)
        shares = notional / price
        fee = self.commission(shares, price)
        per_share = shares * self.commission_per_share + shares * price * self.commission_bps / 1e4
        return OrderCost(
            notional=notional,
            shares=shares,
            fee=fee,
            adverse=notional * self._adverse,
            floor_binds=fee > per_share + 1e-12,
        )

    def as_dict(self) -> dict[str, Any]:
        """The schedule, recorded alongside the verdict it produced.

        Two runs under different cost models are not comparable, and comparing
        them anyway is how a change of broker gets attributed to a change of
        strategy — the same reason ``dataset_version`` is written with every
        experiment.
        """
        return dict(sorted(asdict(self).items()))


@dataclass(frozen=True, slots=True)
class OrderCost:
    """One order priced under one schedule, with the fixed part separated out."""

    notional: float
    shares: float
    fee: float
    """Broker commission, floor included."""
    adverse: float
    """Spread and slippage. The part that scales with size."""
    floor_binds: bool = False
    """Whether ``min_commission`` exceeded the per-share fee for this order."""

    @property
    def total(self) -> float:
        return self.fee + self.adverse

    @property
    def bps(self) -> float:
        """All-in cost of this order in basis points of its notional."""
        return self.total / self.notional * 10_000.0 if self.notional > 0 else 0.0

    @property
    def fee_bps(self) -> float:
        return self.fee / self.notional * 10_000.0 if self.notional > 0 else 0.0


ZERO_COST = CostModel(
    commission_per_share=0.0,
    commission_bps=0.0,
    min_commission=0.0,
    spread_bps=0.0,
    slippage_bps=0.0,
    participation_cap=1.0,
    borrow_bps_per_year=0.0,
)
"""Frictionless model. For unit tests and for measuring how much cost matters --
never for evaluating a strategy."""

IBKR_FIXED = CostModel()
"""The historical default, named rather than implied.

$0.005/share with a $1.00 order floor is Interactive Brokers' fixed retail tier.
It is preserved exactly as ``CostModel()``'s defaults so that every verdict
already in the registry stays reproducible — recalibrating by editing the
defaults would silently reinterpret 324 recorded experiments.

Its floor is what makes cost depend on breadth and account size; see the module
docstring for the measurement. That is a real property of this schedule, not a
modelling error, and it is the reason a hundred-name book needs either a larger
account or a different broker."""

ALPACA_EQUITIES = CostModel(
    commission_per_share=0.0,
    commission_bps=0.0,
    min_commission=0.0,
    spread_bps=2.0,
    slippage_bps=1.0,
)
"""Commission-free US equities: the venue these bars come from.

Alpaca charges no commission on US equity orders, so nothing here is fixed per
order and cost scales with turnover alone — which is what makes it the schedule
a breadth strategy can actually be judged under. The spread and slippage
assumptions are unchanged from :data:`IBKR_FIXED`, because they are properties of
the market rather than of the broker, and they are where the pessimism belongs.

Not the default. Making it the default would improve every recorded result at a
stroke, and a cost model that gets cheaper the year a strategy needs it to is not
a cost model."""

PRESETS: dict[str, CostModel] = {
    "ibkr_fixed": IBKR_FIXED,
    "alpaca": ALPACA_EQUITIES,
    "zero": ZERO_COST,
}


def preset(name: str) -> CostModel:
    """Look up a named schedule. Unknown names raise rather than defaulting.

    Defaulting would mean a typo silently priced a run under a different broker
    than the one asked for, which is the failure this whole module exists to make
    impossible to have quietly.
    """
    key = name.strip().lower()
    if key not in PRESETS:
        raise ValueError(f"unknown cost preset {name!r}; use one of {', '.join(sorted(PRESETS))}")
    return PRESETS[key]
