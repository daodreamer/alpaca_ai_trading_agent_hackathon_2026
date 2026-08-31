"""The option chain, indexed and queryable — specs/10-options-research.md D5.

``data/option_embargo.py`` produces an :class:`~aqr.data.option_embargo.OptionChain`:
the vendor's rows, unparsed, with a sensor that reports every session it holds
to the seal. It deliberately stops there, because a chain row's meaning lives in
the combination of expiry, strike and right, and the container had no business
picking a layout before anything was known about the consumer.

This is the consumer. It turns those strings into :class:`Quote` objects and
answers the one question the engine asks: *given this session, which contract
does "the 16-delta put at 28 DTE" name?*

Two properties matter more than anything else here.

**Selection refuses rather than approximates.** The cache carries about 24
strikes per expiry, sampled from a ladder that lists hundreds, so a rule can
easily name a contract that is not there. A selector that returned the nearest
available row instead would turn a rule about 16-delta puts into a rule about
whatever the ladder happened to contain that day, and the backtest would report
the second while the spec described the first. Every miss raises.

**Every tie has a stated winner.** Expiry first, by distance from the DTE
target, ties to the earlier expiry. Then the leg, by distance from the delta
target, ties to the lower strike. Choosing expiry and strike jointly would let a
better delta on a worse expiry win, which makes ``dte`` advisory; and a tie
resolved by row order would make the result depend on how the vendor sorted its
file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

__all__ = [
    "ChainIndex",
    "DELTA_IMPLIED_ATM_MAX_DISTANCE",
    "NoSuchContract",
    "Quote",
    "Right",
    "SessionChain",
    "SkipReason",
]

Right = Literal["call", "put"]

SkipReason = Literal[
    "affordability",
    "no_leg_or_wing",
    "no_expiry_in_band",
    "greek_consistency",
    "embargo_refusal",
    "stale_underlying",
]
"""specs/10-options-research.md D8a's skip census, in one place so
``options/engine.py``'s ``SkipCensus`` and this module's own raise sites
cannot drift apart on what a category means. Defined here rather than in
``engine.py`` because ``NoSuchContract`` is what carries the category for
``no_leg_or_wing`` and ``no_expiry_in_band``; the other four --
``affordability``, ``greek_consistency``, ``embargo_refusal`` and
``stale_underlying`` -- are decided in the engine's own session loop, which
already imports this module, so importing one Literal the other direction
would be circular."""

DEFAULT_DELTA_TOLERANCE = 0.06
"""How far from the named delta a contract may sit and still answer to the name.

0.06 around a 0.16 target spans roughly 0.10–0.22, which the measured ladder
supplies on 742 of 753 sessions (specs/10 D0). Wider would let a 0.30-delta leg
answer to "16 delta", which is a different trade with roughly twice the risk."""

_STRIKE_EPSILON = 1e-6
"""Strikes arrive as decimal strings with two places. Compared as floats, so a
width match needs a tolerance rather than equality."""

_DELTA_PLACES = 6
"""Decimal places the delta *distance* is compared at when ranking candidates.

The vendor quotes delta to four places, so two candidates whose distances differ
at the seventeenth are not two candidates -- they are one distance and its
binary representation error. Without this rounding the documented tie-break
(prefer the nearer strike) can never fire, because a floating-point tie
essentially never occurs, and the rule would be decoration.
"""

DELTA_IMPLIED_ATM_MAX_DISTANCE = 0.15
"""How far a call's delta may sit from 0.50 and still count toward
:meth:`SessionChain.delta_implied_spot`.

This is deliberately looser than :data:`DEFAULT_DELTA_TOLERANCE`: that
tolerance guards a leg an order is about to be built from and must refuse an
approximation; this one only has to recognise "near the money" well enough to
notice when it isn't, and a thin session's nearest-to-0.50 call can sit a
little further out without the estimate becoming useless. A session whose
whole call ladder falls outside this band returns ``None`` rather than a
number computed from calls that were never near the money to begin with.
"""

_TOLERANCE_EPSILON = 1e-9
"""Slack on the tolerance comparison, so the boundary means what it says.

``abs(0.08 - 0.06)`` is ``0.020000000000000004`` in binary floating point, so a
contract sitting exactly on a stated tolerance of ``0.02`` would be rejected
while its mirror image at ``0.04`` was accepted. That is deterministic and
therefore not a correctness bug, but it makes ``tolerance: 0.02`` mean something
a spec author cannot predict, and the afternoon spent finding out is not
research.
"""


class NoSuchContract(LookupError):
    """The rule named a contract this session's ladder does not contain.

    A ``LookupError`` rather than a ``ValueError``: nothing is wrong with the
    request, and the same request on the next session may well succeed. The
    engine catches this and skips the entry; it does not repair it.

    ``category`` is D8a's skip census, attached at the raise site rather than
    inferred later from the message text. ``options/engine.py`` catches every
    ``NoSuchContract`` and ``ValueError`` a structure build can throw in one
    ``except`` clause (it has to: a spread's short leg, its wing, and the
    structure's own leg-count invariant fail in three different modules), and
    a census built by pattern-matching that clause's exception *message*
    would silently miscount the day a message was reworded for clarity. The
    category travels with the exception instead, so the two can never
    disagree about what happened -- an unset category defaults to
    ``"no_leg_or_wing"``, which is what every raise site in this module and
    the plain ``ValueError``s ``options/structure.py`` raises when the
    selected legs cannot form a valid structure both are: the day's ladder
    did not offer what the rule asked for.
    """

    def __init__(self, message: str, *, category: SkipReason = "no_leg_or_wing") -> None:
        super().__init__(message)
        self.category: SkipReason = category


@dataclass(frozen=True, slots=True)
class Quote:
    """One contract, as of one session's close.

    ``delta`` keeps the vendor's sign — negative for puts, positive for calls —
    because that is what it means, and a magnitude stored in a field called
    delta would be wrong in every downstream greek calculation. Matching against
    a target is done on the magnitude; see :meth:`SessionChain.select`.
    """

    expiration: date
    strike: float
    right: Right
    bid: float
    ask: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def __post_init__(self) -> None:
        if self.ask < self.bid:
            raise ValueError(
                f"crossed quote: ask {self.ask} below bid {self.bid} for "
                f"{self.right} {self.strike} {self.expiration}"
            )
        if self.bid < 0:
            raise ValueError(f"negative bid {self.bid}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def sellable(self) -> bool:
        """A zero bid is not a price a short leg can be opened at.

        4.7% of the cache has one (specs/10 D0). Selling into it fills at zero,
        which is not a trade; it is a row that looks like one.
        """
        return self.bid > 0.0

    @property
    def relative_spread(self) -> float | None:
        """``(ask - bid) / mid``, or ``None`` when there is no market at all."""
        mid = self.mid
        return (self.ask - self.bid) / mid if mid > 0 else None

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


@dataclass(frozen=True, slots=True)
class SessionChain:
    """Every quote the vendor carried for one underlying on one session."""

    session: date
    quotes: tuple[Quote, ...]

    def dte(self, expiration: date) -> int:
        return (expiration - self.session).days

    def expiries(self, right: Right | None = None) -> tuple[date, ...]:
        pool = self.quotes if right is None else [q for q in self.quotes if q.right == right]
        return tuple(sorted({q.expiration for q in pool}))

    def delta_implied_spot(self) -> float | None:
        """An estimate of this session's underlying price, from the calls'
        own greeks rather than from a bar — specs/10 D2b.

        A call's delta crosses 0.50 at the money, so the listed strike whose
        delta sits nearest 0.50 is close to spot for a normal chain. That
        makes it a second, independent opinion on what the underlying cost,
        computable from the same rows :meth:`select` reads and nothing else —
        which is what makes it useful as a check on the greeks themselves
        rather than on the strikes or the quoted prices, both of which D2a
        already cross-checks against the bar cache and the put-call parity
        identity.

        The strike is taken per expiry, then the **median** across the
        session's expiries is returned, so one thin or noisy expiry cannot
        swing the estimate. Measured against the reference close across all
        753 research sessions: p50 lands exactly on it, p99 within 0.21%,
        p1 within 0.52% — except on four sessions in November 2021 (2021-11-12,
        -17, -19, -22) where this ratio drops to 0.898-0.901, because the
        vendor computed every greek that day against a spot roughly 10% below
        the real one. The separation between the two populations is total,
        which is what makes a single threshold a sound check rather than a
        guess (:data:`aqr.options.engine.GREEK_CONSISTENCY_TOLERANCE`).

        ``None`` when no expiry offers a call within
        :data:`DELTA_IMPLIED_ATM_MAX_DISTANCE` of 0.50 delta -- a chain too
        thin to answer the question, not a session to guess about.
        """
        per_expiry: list[float] = []
        for expiry in self.expiries("call"):
            calls = [q for q in self.quotes if q.right == "call" and q.expiration == expiry]
            nearest = min(calls, key=lambda q: abs(q.abs_delta - 0.5))
            if abs(nearest.abs_delta - 0.5) <= DELTA_IMPLIED_ATM_MAX_DISTANCE:
                per_expiry.append(nearest.strike)
        if not per_expiry:
            return None
        per_expiry.sort()
        mid = len(per_expiry) // 2
        if len(per_expiry) % 2:
            return per_expiry[mid]
        return (per_expiry[mid - 1] + per_expiry[mid]) / 2.0

    def select(
        self,
        *,
        right: Right,
        dte_target: int,
        dte_tolerance: int,
        delta_target: float,
        delta_tolerance: float = DEFAULT_DELTA_TOLERANCE,
        sellable: bool = False,
    ) -> Quote:
        """The one contract ``right``/``dte``/``delta`` names, or a refusal.

        ``sellable=True`` for a leg that will be sold: a zero bid is excluded
        from the candidates rather than selected and then rejected, so a
        tradable strike further from the target still wins.
        """
        if delta_target < 0:
            raise ValueError(
                f"delta_target is a magnitude, got {delta_target}. A put's delta is "
                f"negative in the data; the rule names 0.16 for either right."
            )
        expiry = self._expiry_for(right, dte_target, dte_tolerance)
        pool = [q for q in self.quotes if q.right == right and q.expiration == expiry]
        unsellable = 0
        if sellable:
            before = len(pool)
            pool = [q for q in pool if q.sellable]
            unsellable = before - len(pool)
        near = [
            q
            for q in pool
            if abs(q.abs_delta - delta_target) <= delta_tolerance + _TOLERANCE_EPSILON
        ]
        if not near:
            extra = (
                f"; {unsellable} candidate(s) were dropped for a zero bid"
                if unsellable
                else ""
            )
            raise NoSuchContract(
                f"{self.session}: no {right} within {delta_tolerance:.2f} of delta "
                f"{delta_target:.2f} expiring {expiry} ({self.dte(expiry)} DTE){extra}",
                category="no_leg_or_wing",
            )
        return min(
            near, key=lambda q: (round(abs(q.abs_delta - delta_target), _DELTA_PLACES), q.strike)
        )

    def select_wing(self, short: Quote, *, width_points: float) -> Quote:
        """The protective leg ``width_points`` from ``short``, same expiry.

        Below the short strike for a put, above it for a call — the side that
        caps the loss. A wing on the other side is a different structure with a
        different maximum loss, and an engine that picked one would be sizing
        against a number that does not describe the position.

        An exact match, never the nearest listed strike: the sampled ladder is
        spaced 8–15 points, and quietly widening a spread changes the very
        number ``fixed_risk`` sizing divides by.
        """
        if width_points <= 0:
            raise ValueError(f"width_points must be > 0, got {width_points}")
        protective = -1.0 if short.right == "put" else 1.0
        wanted = short.strike + protective * width_points
        for quote in self.quotes:
            if (
                quote.right == short.right
                and quote.expiration == short.expiration
                and abs(quote.strike - wanted) < _STRIKE_EPSILON
            ):
                return quote
        raise NoSuchContract(
            f"{self.session}: no {short.right} at strike {wanted:g} expiring "
            f"{short.expiration} to wing the short {short.strike:g}",
            category="no_leg_or_wing",
        )

    def select_wing_by_delta(
        self,
        short: Quote,
        *,
        delta_target: float,
        delta_tolerance: float = DEFAULT_DELTA_TOLERANCE,
    ) -> Quote:
        """The protective leg named by *its own* delta rather than by a distance.

        This is the form the cache actually supports, and the difference is not
        marginal. Measured on the SPY research window against a 16-delta short
        put: a fixed 10-point wing resolves on **23%** of sessions, a 6-delta
        wing on **98%**. The listed widths below a 16-delta strike are 8, 9, 10,
        18, 25, 35 and 45 points depending on the session, because the vendor
        samples about 24 rungs from a ladder that lists hundreds -- so "ten
        points wide" is not a rule this data can express, and a search told to
        use one would be selecting on ladder sampling.

        Delta is also the quantity a trader would name. It is stable across
        volatility regimes, where a point width silently becomes a much wider or
        narrower bet as the underlying and its vol move.

        Ties go to the strike nearest the short leg -- the narrower spread, and
        so the smaller position -- rather than to whichever row came first.
        """
        if delta_target < 0:
            raise ValueError(f"delta_target is a magnitude, got {delta_target}")
        protective = [
            quote
            for quote in self.quotes
            if quote.right == short.right
            and quote.expiration == short.expiration
            and _is_protective(quote.strike, short)
            and abs(quote.abs_delta - delta_target) <= delta_tolerance + _TOLERANCE_EPSILON
        ]
        if not protective:
            raise NoSuchContract(
                f"{self.session}: no {short.right} within {delta_tolerance:.2f} of delta "
                f"{delta_target:.2f} on the protective side of the {short.strike:g} "
                f"{short.right} expiring {short.expiration}",
                category="no_leg_or_wing",
            )
        return min(
            protective,
            key=lambda q: (
                round(abs(q.abs_delta - delta_target), _DELTA_PLACES),
                abs(q.strike - short.strike),
            ),
        )

    def _expiry_for(self, right: Right, dte_target: int, dte_tolerance: int) -> date:
        candidates = [
            expiry
            for expiry in self.expiries(right)
            if abs(self.dte(expiry) - dte_target) <= dte_tolerance
        ]
        if not candidates:
            offered = ", ".join(f"{self.dte(e)}" for e in self.expiries(right)) or "none"
            raise NoSuchContract(
                f"{self.session}: no {right} expiry within {dte_tolerance} days of "
                f"{dte_target} DTE (offered: {offered})",
                category="no_expiry_in_band",
            )
        return min(candidates, key=lambda e: (abs(self.dte(e) - dte_target), e))


@dataclass(frozen=True, slots=True)
class ChainIndex:
    """Every session, in order, each one queryable.

    Built from the plain dicts :meth:`~aqr.data.option_embargo.OptionChain.as_dicts`
    returns rather than from the container itself, so this module imports
    nothing from ``data/`` and the seal's sensor stays the only thing that has
    an opinion about which rows were read.
    """

    _by_session: dict[date, SessionChain]

    @property
    def sessions(self) -> tuple[date, ...]:
        return tuple(self._by_session)

    def __len__(self) -> int:
        return len(self._by_session)

    def __contains__(self, session: date) -> bool:
        return session in self._by_session

    def __getitem__(self, session: date) -> SessionChain:
        """The chain for ``session``, or ``KeyError``.

        Never an empty chain. "Nothing qualified today" and "that session is not
        in the cache" are different facts, and a caller that cannot tell them
        apart will silently treat a hole in the data as a run of days on which
        the rule declined to trade.
        """
        try:
            return self._by_session[session]
        except KeyError:
            raise KeyError(
                f"{session} is not in the chain cache "
                f"({len(self._by_session)} sessions, "
                f"{self.first()} .. {self.last()})"
            ) from None

    def first(self) -> date | None:
        return next(iter(self._by_session), None)

    def last(self) -> date | None:
        return next(reversed(self._by_session), None)

    def slice_dates(self, start: date, stop: date) -> ChainIndex:
        """Sessions in ``[start, stop)`` -- the input a walk-forward fold slices.

        specs/10 D8 defines a fold in calendar time, not in bar position, and
        the fold boundary has to be enforced on the *input* the engine can act
        on rather than on its output: the engine's session loop only ever asks
        "which sessions does ``chain.sessions`` offer", so restricting that
        sequence is what stops an entry from happening outside the fold at
        all, as opposed to computing one full-window run and discarding the
        trades whose entry fell outside the window afterwards -- which is the
        failure mode specs/10 D8 names directly: a position entered in a train
        window can settle inside the following test window, and slicing the
        *output* after the fact does not stop that from having happened
        during the run.

        Half-open on the session date, matching every other range in this
        codebase (``Split``, ``Bars.slice``): ``stop`` is the first date this
        chain no longer covers, so consecutive folds compose with
        ``a.slice_dates(x, y)`` and ``b.slice_dates(y, z)`` sharing no
        session.
        """
        if stop < start:
            raise ValueError(f"slice_dates: stop {stop} precedes start {start}")
        kept = {
            session: chain
            for session, chain in self._by_session.items()
            if start <= session < stop
        }
        return ChainIndex(_by_session=kept)

    def exclude_dates(self, start: date, stop: date) -> ChainIndex:
        """Every session except those in ``[start, stop)`` -- the complement
        of :meth:`slice_dates`.

        Built for the leave-one-year-out robustness pass (specs/10 D8):
        removing a whole year of *entry opportunities* has to happen on the
        input the same way a fold's window does, and for the same reason. A
        position entered just before the excluded range that happens to
        settle inside it is unaffected either way it is sliced, because
        settlement is priced from the underlying's close and never from the
        chain (D1) -- what this actually controls is only whether the excluded
        range could ever have supplied a new entry.
        """
        if stop < start:
            raise ValueError(f"exclude_dates: stop {stop} precedes start {start}")
        kept = {
            session: chain
            for session, chain in self._by_session.items()
            if not (start <= session < stop)
        }
        return ChainIndex(_by_session=kept)

    @classmethod
    def from_rows(
        cls, rows: Iterable[Mapping[str, str]], *, before: date | None = None
    ) -> ChainIndex:
        """Parse the vendor's columns into sessions of quotes.

        ``before`` drops sessions on or after a boundary. That is the crude
        cousin of D3's rule — the engine still has to refuse an entry whose
        *expiry* crosses the embargo, which this cannot see — and it exists so a
        caller can hold a sealed-root chain and index only the research half
        without writing the filter itself.
        """
        collected: dict[date, list[Quote]] = {}
        for row in rows:
            session = date.fromisoformat(row["date"])
            if before is not None and session >= before:
                continue
            collected.setdefault(session, []).append(_quote(row))
        ordered = {
            session: SessionChain(
                session=session,
                # Sorted so the tuple does not inherit the vendor's file order.
                # Selection breaks its own ties explicitly, but a container whose
                # contents depend on input order is a determinism bug waiting for
                # a caller that iterates.
                quotes=tuple(sorted(quotes, key=lambda q: (q.expiration, q.strike, q.right))),
            )
            for session, quotes in sorted(collected.items())
        }
        return cls(_by_session=ordered)


def _is_protective(strike: float, short: Quote) -> bool:
    """Below the short strike for a put, above it for a call.

    The side that caps the loss. A leg on the other side is a different
    structure with a different maximum loss wearing the same name.
    """
    return strike < short.strike if short.right == "put" else strike > short.strike


def _quote(row: Mapping[str, str]) -> Quote:
    right: Right = "put" if row["call_put"].strip().lower().startswith("p") else "call"
    return Quote(
        expiration=date.fromisoformat(row["expiration"]),
        strike=float(row["strike"]),
        right=right,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        iv=float(row["vol"]),
        delta=float(row["delta"]),
        gamma=float(row["gamma"]),
        theta=float(row["theta"]),
        vega=float(row["vega"]),
        rho=float(row["rho"]),
    )
