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
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestResult, Trade
from aqr.data.bars import Bars
from aqr.dsl.expr import evaluate
from aqr.options.chain import ChainIndex, NoSuchContract, Quote, Right, SessionChain, SkipReason
from aqr.options.costs import IBKR_OPTIONS, OptionCostModel
from aqr.options.features import OptionFeatureFrame, VolatilityHistory
from aqr.options.spec import OptionSpec, StructureSpec
from aqr.options.structure import Leg, Side, Structure
from aqr.seal import EMBARGO_START

__all__ = [
    "MULTIPLIER",
    "OptionBacktestConfig",
    "OptionBacktestResult",
    "OptionTrade",
    "SkipCensus",
    "SkippedEntry",
    "affordability_bound_fraction",
    "build_structure",
    "run_option_backtest",
]

MULTIPLIER = 100.0
"""Shares per contract. Applied once, here — prices everywhere else are per share."""

DAYS_PER_YEAR = 365.0
"""Calendar days, for the mark's time-to-expiry. An option decays on weekends."""

GREEK_CONSISTENCY_TOLERANCE = 0.05
"""How far :meth:`~aqr.options.chain.SessionChain.delta_implied_spot` may
disagree with the session's reference close before the session is refused —
D2b.

Four sessions in the research cache (2021-11-12, -11-17, -11-19, -11-22) have
greeks the vendor computed against an underlying price about 10% below the
real one: prices are correct (D2a's parity check passes on them), but a
contract's delta, and therefore what selecting "16 delta" actually buys, is
not. Measured across all 753 sessions, ``delta_implied_spot() / reference
close`` has p1 0.995, p5 0.999, p50 1.000, p95 1.001, p99 1.002, max 1.040 —
and the four bad sessions sit at 0.898-0.901. The separation is total, so any
threshold between about 1% and 9% draws the same line; 5% is the middle of
that band and leaves no margin for a session that is merely noisy to be
mistaken for one that is wrong.

This is silent by construction, same as the failure D2a catches: nothing
raises, the backtest runs to completion, and a "16-delta" structure is priced
and sized around a strike that was never 16 delta against the real spot.
"""


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


@dataclass(frozen=True, slots=True)
class SkippedEntry:
    """One session the engine declined to enter, with why -- specs/10 D8a.

    Carries the session date as a first-class field rather than leaving it
    embedded only in ``reason``'s prose, because a consumer that wants to
    bucket skips by calendar window (``options/walkforward.py``'s fold
    slicing, the way it already buckets ``OptionTrade.entry_session``) needs
    the date typed, not parsed back out of a formatted string.
    """

    session: date
    category: SkipReason
    reason: str


@dataclass(frozen=True, slots=True)
class SkipCensus:
    """Why a run did not trade, counted rather than left to English -- D8a.

    Measured on the real SPY cache, ``put_credit_spread`` / 28 DTE / anchor
    0.16 / width_delta 0.06 / cadence 5 / ``max_concurrent`` 3, at two
    sizings:

    ```
    risk_per_trade   trades   independent cycles   skipped   affordability
          1%            31            21             598          578
          5%           148            57              13            0
    ```

    At 1% of a $100,000 account (D5's own default), the risk budget is
    $1,000 and this structure's median maximum loss is $892 per contract --
    right at the edge, so most sessions cannot afford even one contract and
    the entry is skipped for a reason that has nothing to do with whether the
    rule is any good. **578 of 598 skips at 1% are affordability, not the
    market.** The same pathology ``backtest/costs.py``'s module docstring
    documents for the equity per-order floor: "cost retention ... decides
    verdicts, and it decides them on ``BacktestConfig.initial_equity``, a
    number nobody thinks of as a cost parameter." This census is what makes
    that fact visible here instead of being rediscovered by hand every time a
    rule's cycle count looks suspiciously low.

    ``skipped_reasons`` on :class:`OptionBacktestResult` stays a flat list of
    formatted strings -- unchanged, and still the thing to read for the
    detail of one particular session. This is the thing to read for anything
    that has to *reason* about why a run did not trade, without parsing
    English to do it. Both are built from the same
    :class:`SkippedEntry` list in :func:`run_option_backtest`, so the two
    views cannot disagree about what happened on a given session.
    """

    affordability: int = 0
    """``risk_per_trade`` of realised equity could not cover one contract at
    the structure's own maximum loss. A property of the account and the
    sizing rule, never of whether the structure was a good idea."""
    no_leg_or_wing: int = 0
    """The ladder that session did not offer a leg (or a sellable wing) at
    the named delta -- ``NoSuchContract`` from ``chain.py``'s ``select``,
    ``select_wing`` or ``select_wing_by_delta``, or a ``ValueError`` from
    ``structure.py`` when the legs selected cannot form a valid structure.
    Both mean the same thing from the rule's point of view: the day's ladder
    did not have what was asked for."""
    no_expiry_in_band: int = 0
    """No listed expiry sat within ``dte.tolerance`` of ``dte.target`` --
    ``NoSuchContract`` from ``chain.py``'s ``_expiry_for``. Distinct from
    ``no_leg_or_wing`` because the two point at different fixes: a rule
    starved by this one needs a wider DTE tolerance or a different target; a
    rule starved by that one needs a wider delta tolerance."""
    greek_consistency: int = 0
    """D2b's guard: the session's greeks were computed against an
    underlying price the delta-implied spot disagrees with by more than
    :data:`GREEK_CONSISTENCY_TOLERANCE`. Four sessions in the research cache
    (2021-11-12, -17, -19, -22) take this path; the guard exists so a
    mislabelled "16 delta" is never traded rather than repaired."""
    embargo_refusal: int = 0
    """D3: the structure's expiry falls on or after the embargo boundary,
    so filling it would settle against reserved data. 9 of 753 research
    sessions at 28 DTE take this path.

    Named ``embargo_refusal`` rather than the shorter ``embargo`` on
    purpose: ``tests/test_seal.py``'s
    ``test_the_embargo_is_a_constant_not_a_parameter_of_the_search`` scans
    every call in this codebase for an ``embargo=`` keyword argument, on the
    theory that the only reason to pass one is to override
    ``EMBARGO_START`` -- exactly the loophole that guard exists to close. A
    dataclass field is not that loophole, but its name is indistinguishable
    from one to a scanner that only looks at keyword spelling, and
    ``SkipCensus(embargo=...)`` would have tripped it on every call site
    that builds this census. The false positive is cheaper to avoid by
    renaming the field than by teaching the guard to special-case this
    module."""
    stale_underlying: int = 0
    """No trading-day close within ``max_reference_staleness_days`` of the
    session -- either the *entry* reference (82 of 753 sessions: 2019's
    Saturday snapshots and market holidays, all within the bound) or the
    *settlement* reference on the structure's own expiry (742 of 742
    settleable expiries have one, so this branch is a defensive refusal
    rather than a measured cost)."""

    @property
    def total(self) -> int:
        return (
            self.affordability
            + self.no_leg_or_wing
            + self.no_expiry_in_band
            + self.greek_consistency
            + self.embargo_refusal
            + self.stale_underlying
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self) | {"total": self.total}

    @classmethod
    def from_entries(cls, entries: list[SkippedEntry]) -> SkipCensus:
        counts = Counter(entry.category for entry in entries)
        return cls(
            affordability=counts.get("affordability", 0),
            no_leg_or_wing=counts.get("no_leg_or_wing", 0),
            no_expiry_in_band=counts.get("no_expiry_in_band", 0),
            greek_consistency=counts.get("greek_consistency", 0),
            embargo_refusal=counts.get("embargo_refusal", 0),
            stale_underlying=counts.get("stale_underlying", 0),
        )


def affordability_bound_fraction(census: SkipCensus, num_trades: int) -> float:
    """D8a's definition of "affordability-bound," the one place it is made.

    Affordability skips as a fraction of the sessions where the engine got
    far enough to *ask* whether a contract was affordable -- every session
    that reached the sizing step, whether it went on to open a trade or was
    turned away for being too small. That denominator is exactly
    ``census.affordability + num_trades``: everything else in
    :class:`SkipCensus` (no leg, no expiry, the greek guard, the embargo, a
    stale reference) is refused *before* sizing is ever computed, so those
    sessions never had an affordability opinion to contribute one way or the
    other, and folding them into the denominator would dilute the fraction
    with sessions that were never in play for this question.

    Not "affordability skips over all skips": a rule that mostly gets no
    contract because its DTE band is too narrow would report a small
    fraction under that definition even though every session it *did* reach
    sizing on was unaffordable, which is the opposite of what a reader needs
    to know before trusting a cycle count. This definition asks only about
    the sessions where the account, and nothing else, decided the outcome.

    Measured on the real cache at ``risk_per_trade=0.01`` for
    ``put_credit_spread`` / 28 DTE / 0.16 anchor / 0.06 width: 578
    affordability skips against 31 trades is ``578 / (578 + 31) = 94.9%`` --
    the run in :class:`SkipCensus`'s own docstring, and the reason
    :data:`~aqr.evaluator.score.AFFORDABILITY_BOUND_THRESHOLD` exists.
    """
    reached_sizing = census.affordability + num_trades
    return census.affordability / reached_sizing if reached_sizing else 0.0


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
    skips: list[SkippedEntry] = field(default_factory=list)
    """The structured form of ``skipped_reasons``, one :class:`SkippedEntry`
    per line, in the same order. What ``skip_census`` is built from, and what
    a fold-level report (``options/walkforward.py``) buckets by session the
    same way it already buckets ``option_trades``."""
    skip_census: SkipCensus = field(default_factory=SkipCensus)
    """``SkipCensus.from_entries(skips)``, computed once here so every reader
    of this result sees the same counts ``skips`` itself would produce."""


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
                f"{chain.session}: the {wing.strike:g} {right} wing has a zero bid",
                category="no_leg_or_wing",
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
    *,
    volatility: VolatilityHistory | None = None,
) -> OptionBacktestResult:
    """Simulate ``spec`` over ``chain``, settling against ``underlying``.

    ``volatility`` is ``volatility_history``, already indexed (specs/10 D6).
    Optional and keyword-only: an entry that never names an option feature --
    every equity-shaped spec this engine ran before D6 existed -- needs none of
    it, and every existing positional call site (``run_option_backtest(spec,
    chain, bars, config)``) is unaffected by this parameter's addition.
    """
    config = config or OptionBacktestConfig()
    days = [datetime.fromtimestamp(int(t), tz=UTC).date() for t in underlying.event_time]
    if not days:
        raise ValueError(f"{underlying.symbol}: no bars to settle against")
    closes = np.asarray(underlying.close, dtype=np.float64)
    day_index = {day: i for i, day in enumerate(days)}

    frame = OptionFeatureFrame(underlying, volatility=volatility, chain=chain)
    warmup = frame.warmup(spec.features())
    signal = np.asarray(evaluate(spec.entry_ast, frame), dtype=bool)
    signal[:warmup] = False

    opened: list[_Open] = []
    skipped: list[SkippedEntry] = []

    def _skip(session: date, category: SkipReason, reason: str) -> None:
        # The one place a skip enters the record, so ``skipped_reasons`` (the
        # formatted detail) and ``skip_census`` (the count D8a gates on) are
        # built from the same list and cannot drift apart -- every other skip
        # in this loop calls this instead of appending to a bare string list.
        skipped.append(SkippedEntry(session=session, category=category, reason=reason))

    realised = config.initial_equity
    last_entry_at: int | None = None
    settled_by_index: dict[int, float] = {}

    for position, session in enumerate(chain.sessions):
        reference = _reference_index(days, session, config.max_reference_staleness_days)
        if reference is None:
            _skip(
                session,
                "stale_underlying",
                f"{session}: no underlying close within "
                f"{config.max_reference_staleness_days} days",
            )
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

        session_chain = chain[session]
        implied_spot = session_chain.delta_implied_spot()
        if implied_spot is not None:
            reference_close = closes[reference]
            ratio = implied_spot / reference_close
            if abs(ratio - 1.0) > GREEK_CONSISTENCY_TOLERANCE:
                _skip(
                    session,
                    "greek_consistency",
                    f"{session}: refused, delta-implied ATM strike {implied_spot:g} is "
                    f"{ratio:.1%} of the reference close {reference_close:,.2f} -- the "
                    f"chain's greeks look computed against the wrong underlying price (D2b)",
                )
                continue

        try:
            structure = build_structure(spec.structure, session_chain)
        except (NoSuchContract, ValueError) as exc:
            # A plain ValueError (structure.py's own leg-count/strike
            # invariants) carries no ``category`` attribute; it means the
            # same thing a ``NoSuchContract`` defaults to -- the day's ladder
            # did not yield a valid structure -- so it is categorised the
            # same way rather than left uncounted.
            category: SkipReason = getattr(exc, "category", "no_leg_or_wing")
            _skip(session, category, f"{session}: {exc}")
            continue

        if structure.expiration >= config.boundary:
            _skip(
                session,
                "embargo_refusal",
                f"{session}: refused, the structure expires {structure.expiration}, on or "
                f"after {config.boundary} -- settling it would read the reserved window",
            )
            continue
        expiry_index = day_index.get(structure.expiration)
        if expiry_index is None:
            # Same failure mode as the entry-reference staleness check above,
            # at the other end of the position: no bar exists to settle
            # against. specs/10 D0 measures this at 0 of 742 settleable
            # expiries in the research cache, so it is a defensive refusal
            # rather than a cost this census expects to see move.
            _skip(
                session,
                "stale_underlying",
                f"{session}: no underlying close on expiry {structure.expiration}",
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
            _skip(
                session,
                "affordability",
                f"{session}: {spec.sizing.risk_per_trade:.1%} of {realised:,.0f} does not "
                f"cover one contract at {risk_per_contract:,.0f} of risk",
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
        skipped_reasons=[entry.reason for entry in skipped],
        skips=skipped,
        skip_census=SkipCensus.from_entries(skipped),
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
