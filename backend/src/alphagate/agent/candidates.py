"""Building the menu — specs/05 D1 step 3.

Pure. Takes a chain snapshot and produces a bounded list of validated,
priced, already-sized `Candidate`s. No I/O, no clock: `as_of` is an argument,
as everywhere else in this system.

This module is where three of specs/05's guarantees are actually implemented:

**The model never writes a symbol.** It receives an indexed list of structures
that already exist and are already legal. Nothing downstream parses model text
into a contract, because there is no such text.

**Stale quotes drop candidates before the model sees them** (D6). Not after —
a candidate shown and then rejected for staleness is a candidate the model spent
its reasoning on, and a race we would have to explain.

**Zero-quantity candidates never appear** (D4). If the risk budget cannot fund
one unit, the structure is not an option, so it is not offered as one.

**Nor do candidates the delta budget cannot admit.** Found by watching the live
agent: the menu ranks by return on risk, the highest return is always the strike
closest to the money, and that is also the highest delta — so index 0 was
systematically the one candidate the Gate would refuse. The model chose it,
correctly by its own lights, and was vetoed every cycle. A menu whose ranking
fights the Gate is the same mistake as showing a structure the account cannot
size, and it gets the same answer: drop it before the model sees it.

Wide spreads are filtered here at a *looser* threshold than the Gate's, so a
marginal-liquidity structure can still appear and still be refused — this filter
keeps the menu short, the Gate's `liquidity` check refuses the trade, and making
them equal would mean the Gate never fires on this path.

But the threshold was only half the problem, and the other half took a live run
to see. **Ranking by raw return on risk put the widest spread first.** The
closest-to-the-money strike has the best return *and* the worst market, so index
0 was systematically the one the Gate would refuse; the model chose it, and was
vetoed, every cycle. A loose filter plus a ranking that fights the Gate is not a
control demonstrating itself, it is a pipeline that never trades.

So the ranking is now **net of the crossing cost**: the spread you must pay to
get in is subtracted from the return you are ranking on. That is the financially
correct metric anyway — a 30% return that costs 6% to enter is not a better
trade than a 26% return that costs 1% — and it stops the ordering from
adversarially selecting the Gate's refusals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from alphagate.agent.model import Candidate
from alphagate.agent.sizing import size_for
from alphagate.core.errors import InvariantViolation
from alphagate.options import (
    Leg,
    OptionContract,
    OptionQuote,
    OptionStructure,
    Right,
    Side,
    StructureKind,
    StructureRisk,
    compute_risk,
)
from alphagate.risk import RiskLimits

__all__ = [
    "MENU_LIMIT",
    "SHORTLIST_SPREAD_PCT",
    "build_candidates",
    "net_return_on_risk",
    "spreads_by_delta",
    "summarise_menu",
    "vertical_credit_spreads",
]

MENU_LIMIT: Final = 12
"""specs/05 D3: typically 6 to 12. A longer menu is not a better menu — it is a
longer prompt with more ways to be wrong, and the tail of it is never chosen."""

SHORTLIST_SPREAD_PCT: Final = Decimal("0.08")
"""Looser than the Gate's 5%. See the module docstring."""


def vertical_credit_spreads(
    quotes: Mapping[OptionContract, OptionQuote],
    *,
    right: Right,
    width: Decimal,
    as_of: datetime,
) -> list[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]]:
    """Every constructible `width`-wide credit spread in a quote set.

    A put credit spread sells the higher strike; a call credit spread sells the
    lower one. Getting that backwards builds a debit spread and labels it a
    credit, which `OptionStructure` would refuse — so the direction is asserted
    by construction rather than by comment.
    """
    by_strike = {
        contract.strike: contract
        for contract in quotes
        if contract.right is right
    }
    built: list[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]] = []
    for strike, short_contract in sorted(by_strike.items()):
        long_strike = strike - width if right is Right.PUT else strike + width
        long_contract = by_strike.get(long_strike)
        if long_contract is None:
            continue
        try:
            structure = OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (Leg(short_contract, Side.SELL), Leg(long_contract, Side.BUY)),
            )
        except InvariantViolation:
            # A shape the domain refuses is not a candidate. specs/02 D3 already
            # decided which shapes exist; this loop does not get to disagree.
            continue
        built.append(
            (
                structure,
                {
                    short_contract: quotes[short_contract],
                    long_contract: quotes[long_contract],
                },
            )
        )
    return built


def spreads_by_delta(
    quotes: Mapping[OptionContract, OptionQuote],
    *,
    right: Right,
    anchor_delta: float,
    anchor_tolerance: float,
    width_delta: float,
    as_of: datetime,
) -> list[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]]:
    """Credit spreads whose wings are chosen by delta rather than by points.

    specs/07 D1's rule names both legs by delta: sell the strike nearest
    `anchor_delta`, buy protection nearest `width_delta` further out. A live
    chain reshapes every session — the strike nearest 0.16 delta today is not
    the strike nearest it tomorrow — so this is the resolution step the option
    book's own `consumer_must_supply` asks for (`agent/option_book.py`), and it
    is a second function rather than a `width` argument bolted onto
    `vertical_credit_spreads` above: that one scans every strike for a *fixed*
    points width, and there is no fixed width here to scan for.

    Resolved once per expiry in the window, not once for the whole chain. A
    14-DTE anchor and a 24-DTE anchor do not share a strike, and folding them
    into one search would pick whichever expiry's skew happened to be closest
    to the target — the nearest-to-target strike for neither.

    **Absence is not zero.** A contract with no `greeks` on its quote is not a
    candidate for either leg (specs/02 D2): a strike nobody can rank by delta
    must not be silently ranked at delta zero, which would make it look like
    the money strike and win every tie against a strike that is actually
    measured.

    **No fallback to a points wing.** If nothing beyond the anchor clears
    `width_delta`, that expiry produces no structure. A chain thin enough that
    no wing exists is a chain this rule was not researched to trade, and
    substituting a points width would execute a different structure under the
    name of this one — exactly what `option_book.py` refuses at load time for a
    book that names `width_points` at all.
    """
    by_expiry: dict[date, list[OptionContract]] = {}
    for contract in quotes:
        if contract.right is right:
            by_expiry.setdefault(contract.expiry, []).append(contract)

    built: list[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]] = []
    for expiry in sorted(by_expiry):
        contracts = by_expiry[expiry]
        short_contract = _nearest_by_delta(
            contracts, quotes, target=anchor_delta, tolerance=anchor_tolerance
        )
        if short_contract is None:
            continue
        wings = [c for c in contracts if _further_otm(c, short_contract, right)]
        long_contract = _nearest_by_delta(wings, quotes, target=width_delta, tolerance=None)
        if long_contract is None:
            continue
        try:
            structure = OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (Leg(short_contract, Side.SELL), Leg(long_contract, Side.BUY)),
            )
        except InvariantViolation:
            # A shape the domain refuses is not a candidate — same discipline as
            # `vertical_credit_spreads` above.
            continue
        built.append(
            (
                structure,
                {
                    short_contract: quotes[short_contract],
                    long_contract: quotes[long_contract],
                },
            )
        )
    return built


def _nearest_by_delta(
    contracts: Sequence[OptionContract],
    quotes: Mapping[OptionContract, OptionQuote],
    *,
    target: float,
    tolerance: float | None,
) -> OptionContract | None:
    """The contract whose `|delta|` is closest to `target`, or `None`.

    Ties broken by strike rather than left to dict order, so the choice is
    total and reproducible (specs/05 D7) — two contracts equidistant from the
    target must not pick differently between the run that journals a cycle and
    the run that replays it.
    """
    best: OptionContract | None = None
    best_key: tuple[float, int] | None = None
    for contract in contracts:
        quote = quotes.get(contract)
        if quote is None or quote.greeks is None:
            continue  # absence is not zero — specs/02 D2
        distance = abs(abs(quote.greeks.delta) - target)
        if tolerance is not None and distance > tolerance:
            continue
        key = (distance, contract.strike_thousandths)
        if best_key is None or key < best_key:
            best_key, best = key, contract
    return best


def _further_otm(candidate: OptionContract, anchor: OptionContract, right: Right) -> bool:
    """Whether `candidate` sits further out of the money than `anchor`.

    A put moves further OTM as its strike falls; a call, as its strike rises.
    Getting this backwards would let the "protective" leg sit inside the
    anchor, which is the same not-defined-risk shape `option_book.py` already
    refuses at the book level (CLAUDE.md §3.6) — this is that rule applied to a
    live chain instead of a sealed one.
    """
    if right is Right.PUT:
        return candidate.strike < anchor.strike
    return candidate.strike > anchor.strike


def build_candidates(
    structures: Iterable[tuple[OptionStructure, Mapping[OptionContract, OptionQuote]]],
    *,
    limits: RiskLimits,
    equity: Decimal,
    as_of: datetime,
    book_delta: float = 0.0,
    max_spread_pct: Decimal = SHORTLIST_SPREAD_PCT,
    limit: int = MENU_LIMIT,
) -> tuple[Candidate, ...]:
    """Price, filter, size, rank and number the menu.

    `book_delta` is the portfolio's current net delta, so a candidate is judged
    on the exposure it would leave rather than the exposure it carries. Passing
    the default of zero is right only for an empty book; passing a stale one
    would show a menu that the Gate then refuses, which is the failure this
    parameter exists to prevent.

    Ranked by return on risk, descending, then by expiry and short strike so the
    order is total and therefore reproducible — two candidates with identical
    return-on-risk must not swap places between runs (specs/05 D7).
    """
    priced: list[tuple[OptionStructure, StructureRisk]] = []
    for structure, quotes in structures:
        try:
            risk = compute_risk(structure, quotes, as_of)
        except InvariantViolation:
            continue

        if risk.quote_age_seconds > limits.max_quote_age:
            continue  # D6: dropped before the model sees it, not after
        if risk.worst_spread_pct > max_spread_pct:
            continue
        if risk.days_to_expiry < limits.dte_range[0] or risk.days_to_expiry > limits.dte_range[1]:
            continue
        quantity = size_for(risk, limits, equity)
        if quantity <= 0:
            continue  # D4: never shown what it cannot legally trade
        if not _within_delta_budget(risk, limits, equity, book_delta, quantity):
            continue  # the menu must not rank the Gate's refusals to the top
        priced.append((structure, risk))

    priced.sort(key=_rank)
    return tuple(
        Candidate(
            index=position,
            structure=structure,
            risk=risk,
            quantity=size_for(risk, limits, equity),
        )
        for position, (structure, risk) in enumerate(priced[:limit])
    )


def _within_delta_budget(
    risk: StructureRisk,
    limits: RiskLimits,
    equity: Decimal,
    book_delta: float,
    quantity: int,
) -> bool:
    """Would this candidate, at its own size, still fit the delta band?

    Unknown greeks **pass** here and are left to the Gate. That is deliberate:
    `known_greeks` refusing an opening trade whose exposure nobody can state is
    a veto the journal should show, and silently filtering it out here would
    hide a data-quality problem behind an empty menu.
    """
    if risk.net_greeks is None:
        return True
    low, high = limits.scaled_delta_band(equity)
    after = book_delta + risk.net_greeks.delta * quantity
    return low <= after <= high


def net_return_on_risk(risk: StructureRisk) -> Decimal:
    """Return on risk, less the bid/ask you have to cross to get in.

    The crossing cost is charged as the full width of the worst leg's market,
    against the credit taken in: you give up roughly half the spread entering
    and half again leaving, so a round trip is about one whole spread. Charging
    it once, up front, is the conservative reading and it is the one that stops
    a wide market from looking like a good trade.
    """
    if risk.max_loss <= 0:  # pragma: no cover - specs/02 D4 guards this
        return Decimal(0)
    crossing = abs(risk.net_premium) * risk.worst_spread_pct
    return (risk.max_profit - crossing) / risk.max_loss


def _rank(entry: tuple[OptionStructure, StructureRisk]) -> tuple[Decimal, str, int]:
    """Best net return on risk first; ties broken by something total and stable.

    The tiebreakers matter as much as the metric: indices are the model's whole
    vocabulary, so two candidates that score identically must not swap places
    between the run that produced a journal line and the run that replays it.
    """
    structure, risk = entry
    return (
        -net_return_on_risk(risk),
        structure.expiry.isoformat(),
        min(leg.contract.strike_thousandths for leg in structure.legs),
    )


def summarise_menu(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    """The list handed to the model. Indices are positions, and they are stable."""
    return [candidate.summarise() for candidate in candidates]
