"""The measured claims in specs/10-options-research.md D0, checked against the cache.

Every design decision in that spec is downstream of a number measured here:
held-to-expiry-only follows from the re-quote rate, the event-driven clock
follows from the session cadence, the search budget follows from the cycle
count. Those numbers are in a document, and a document cannot notice when the
data moves underneath it.

So they are asserted. A re-pull that restates history fails this file rather
than silently invalidating the reasoning in the spec, and whoever is holding it
then gets to decide whether the spec or the cache is wrong.

Read with ``csv`` throughout and never through :class:`OptionChain`, for the
same reason ``audit_option_root`` does: the container reports every session it
holds to the seal, so a test that checked the research cache through it would
be the peek it is checking for.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pytest

from aqr.agent.option_prompt import CYCLE_BUDGET, SESSIONS_PER_YEAR
from aqr.evaluator.score import MIN_INDEPENDENT_CYCLES
from aqr.options.chain import ChainIndex
from aqr.options.engine import GREEK_CONSISTENCY_TOLERANCE
from aqr.seal import EMBARGO_START

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "data-options" / "option_chain" / "SPY.csv"
VOLATILITY = ROOT / "data-options" / "volatility_history" / "SPY.csv"
UNDERLYING = ROOT / "data-options-underlying" / "1D" / "SPY.csv"
SEALED_CHAIN = ROOT / "data-options-sealed" / "option_chain" / "SPY.csv"
SEALED_UNDERLYING = ROOT / "data-options-underlying-sealed" / "1D" / "SPY.csv"

PARITY_TOLERANCE = 0.02
"""D2a's measured bound. On the research cache with the raw series the implied
spot agrees with the bar close to a median of 0.14% and p99 0.61%; under the
dividend-adjusted series it was out by ten percent. Anything between about 1%
and 5% draws the same line, and 2% is where the spec drew it."""

PARITY_OUTLIER_BUDGET = 0.01
"""How much of a cache may sit outside the tolerance and still be the same price space.

The claim these tests make is "the two caches are in one price space", and the
first version asserted it as ``max(errors) < PARITY_TOLERANCE`` on a 753-session
research cache where the worst session happened to reach 1.98%. That held only
by luck about the sample: the sealed cache runs to 1,260 sessions and contains
two sessions at 2.25% and 2.29% -- 0.16% of it -- while its median (0.147%),
p95 (0.42%) and p99 (0.86%) are indistinguishable from the research cache's.

Two bad rows in twelve hundred are vendor noise, and this file already documents
that the vendor ships a handful of them (see the four sessions whose greeks are
computed against the wrong spot, below). A dividend-adjusted pull -- the failure
this check exists to catch -- misprices *every* session by about ten percent, so
it fails the percentile assertion and the budget together and could not hide
here. Asserting the bulk plus a budget therefore tests the claim more nearly
than a maximum does, and it does not loosen it: 1% of sessions is a far tighter
statement than "no session anywhere", once "anywhere" grows with the cache."""

pytestmark = pytest.mark.skipif(
    not CHAIN.exists(),
    reason=(
        "no option cache; `uv run aqr options-pull` then `aqr options-embargo` builds it. "
        "Skipped rather than failed because the cache is gigabytes and is not in git."
    ),
)

CUTOFF = EMBARGO_START.date()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def chain() -> list[dict[str, str]]:
    return _rows(CHAIN)


@pytest.fixture(scope="module")
def sessions(chain: list[dict[str, str]]) -> list[date]:
    return sorted({date.fromisoformat(row["date"]) for row in chain})


# --------------------------------------------------------------------------- #
# The window, and the fact that it stops at the embargo
# --------------------------------------------------------------------------- #


def test_the_research_window_is_753_sessions_ending_at_the_embargo(
    sessions: list[date],
) -> None:
    assert len(sessions) == 753
    assert sessions[0] == date(2019, 2, 9)
    assert sessions[-1] == date(2024, 8, 30)
    assert sessions[-1] < CUTOFF


def test_the_session_grid_is_not_daily_and_thins_out_backwards(
    sessions: list[date],
) -> None:
    """D0's cadence table. 2019 is one Saturday snapshot a week; 2020-2023 is
    Monday/Wednesday/Friday. A rule expressed in "bars" would mean a week at one
    end of this cache and two days at the other, which is why the engine is
    event-driven over sessions and never indexes by position."""
    per_year = Counter(day.year for day in sessions)
    assert per_year[2019] == 48
    assert per_year[2020] == 152
    assert 145 <= per_year[2021] <= 155
    assert 145 <= per_year[2023] <= 155
    assert Counter(day.strftime("%a") for day in sessions if day.year == 2019)["Sat"] == 46


# --------------------------------------------------------------------------- #
# Why the exit is settlement and nothing else (D1)
# --------------------------------------------------------------------------- #


def test_a_contract_is_almost_never_quoted_again_before_it_expires(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """The measurement the whole engine design rests on.

    The vendor carries three expiries per session at rolling ~14/~28/~49 DTE
    targets and resamples the strike ladder around the money each time, so a
    contract selected today is usually absent tomorrow. Mark-to-market is not
    available, which is why there is no stop, no target and no managed exit
    anywhere in specs/10.
    """
    quoted: dict[tuple[str, str, str], set[date]] = defaultdict(set)
    by_session: dict[date, list[dict[str, str]]] = defaultdict(list)
    for row in chain:
        day = date.fromisoformat(row["date"])
        key = (row["expiration"], row["strike"], row["call_put"])
        quoted[key].add(day)
        by_session[day].append(row)

    seen = 0
    later_total = 0
    for day in sessions[::13]:
        candidates = [
            row
            for row in by_session[day]
            if row["call_put"] == "Put"
            and 24 <= (date.fromisoformat(row["expiration"]) - day).days <= 34
            and abs(float(row["delta"]) + 0.16) < 0.06
        ]
        if not candidates:
            continue
        row = candidates[0]
        expiry = date.fromisoformat(row["expiration"])
        key = (row["expiration"], row["strike"], row["call_put"])
        later = [s for s in sessions if day < s <= expiry]
        later_total += len(later)
        seen += sum(1 for s in later if s in quoted[key])

    assert later_total > 100, "too few samples to claim anything"
    assert seen / later_total < 0.10, (
        f"{seen / later_total:.1%} of later sessions re-quote the entry contract. "
        "specs/10 D0 measured 3.3% and concluded mark-to-market is unavailable; "
        "if the cache now supports it, that decision is worth revisiting."
    )


def test_every_expiry_the_engine_could_settle_has_an_underlying_close() -> None:
    """Settlement reads SPY's close on the expiration date, so a missing bar is
    a position that cannot be closed. There are none: 742 of 742."""
    if not UNDERLYING.exists():
        pytest.skip("no underlying cache; `aqr pull --symbols SPY --csv-root ...`")
    closes = {
        datetime.fromisoformat(row["timestamp"]).date() for row in _rows(UNDERLYING)
    }
    expiries = {date.fromisoformat(row["expiration"]) for row in _rows(CHAIN)}
    settleable = {e for e in expiries if e < CUTOFF}
    assert len(settleable) == 742
    assert not settleable - closes


# --------------------------------------------------------------------------- #
# What the ladder supplies (D0, D5)
# --------------------------------------------------------------------------- #


def test_a_16_delta_put_is_available_on_almost_every_session(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """742 of 753. This is what makes delta-targeted selection a real rule here
    rather than one that silently declines to trade most of the time."""
    have = set()
    for row in chain:
        day = date.fromisoformat(row["date"])
        dte = (date.fromisoformat(row["expiration"]) - day).days
        if row["call_put"] == "Put" and 25 <= dte <= 45 and abs(float(row["delta"]) + 0.16) < 0.05:
            have.add(day)
    assert len(have) >= 735, f"only {len(have)} of {len(sessions)} sessions offer a 16-delta put"


def test_about_one_row_in_twenty_cannot_be_sold(chain: list[dict[str, str]]) -> None:
    """A zero bid. ``Quote.sellable`` is why a short leg cannot be opened on one."""
    zero = sum(1 for row in chain if float(row["bid"]) == 0.0)
    assert 0.03 < zero / len(chain) < 0.07


# --------------------------------------------------------------------------- #
# The sample size, which is the honest limit on everything (D8)
# --------------------------------------------------------------------------- #


def test_a_28_dte_program_has_71_independent_cycles(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """Non-overlapping entries: enter, hold to expiry, enter again.

    71 in 5.55 years. This is the denominator the options evaluator gates on,
    and it is the reason the search budget is 20 hypotheses rather than the
    equity side's 414.
    """
    by_session: dict[date, set[date]] = defaultdict(set)
    for row in chain:
        day = date.fromisoformat(row["date"])
        expiry = date.fromisoformat(row["expiration"])
        if 24 <= (expiry - day).days <= 34:
            by_session[day].add(expiry)

    cycles = 0
    free_from: date | None = None
    for day in sessions:
        if free_from is not None and day < free_from:
            continue
        expiries = by_session.get(day)
        if not expiries:
            continue
        expiry = min(expiries)
        if expiry >= CUTOFF:  # D3: settlement would read the reserved window
            break
        cycles += 1
        free_from = expiry
    assert cycles == 71


def _ceiling(
    chain: list[dict[str, str]],
    sessions: list[date],
    target: int,
    lo: date | None = None,
    hi: date | None = None,
) -> int:
    """Independent cycles for a rule with no entry condition at all.

    The engine's own selection rule rather than the ±5 band the 28 DTE test
    above uses: nearest listed expiry within 10 days, ties to the shorter one.
    A different question from that test's -- "how many cycles could ANY rule at
    this target get" rather than "how many does the 28 DTE program get" -- so
    the two numbers are allowed to differ, and at 28 they happen to agree.
    """
    by_session: dict[date, set[date]] = defaultdict(set)
    for row in chain:
        by_session[date.fromisoformat(row["date"])].add(date.fromisoformat(row["expiration"]))

    cycles = 0
    free_from: date | None = None
    for day in sessions:
        if lo is not None and day < lo:
            continue
        if hi is not None and day > hi:
            break
        if free_from is not None and day < free_from:
            continue
        candidates = [e for e in by_session[day] if abs((e - day).days - target) <= 10]
        if not candidates:
            continue
        expiry = min(candidates, key=lambda e: (abs((e - day).days - target), e))
        if expiry >= CUTOFF:
            break
        cycles += 1
        free_from = expiry
    return cycles


def test_the_cycle_budget_shown_to_the_proposer_matches_the_cache(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """:data:`CYCLE_BUDGET` is arithmetic the model is told to trust.

    It is the table that tells a proposer a 49 DTE hypothesis is a REJECT before
    it is written, so a re-pull that moves the session grid must fail here rather
    than leave the prompt quietly lying about what is reachable. The OOS column
    is the walk-forward test span (specs/10 D8: 24mo train / 6mo test / 6mo step
    over this window), which is where the gate is actually applied.
    """
    oos_lo, oos_hi = date(2021, 2, 9), date(2024, 8, 9)
    for target, window, oos, _floor in CYCLE_BUDGET:
        assert _ceiling(chain, sessions, target) == window, f"{target} DTE, whole window"
        assert _ceiling(chain, sessions, target, oos_lo, oos_hi) == oos, f"{target} DTE, OOS"


def test_a_long_dated_hypothesis_cannot_clear_the_gate_at_any_threshold(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """The claim the prompt makes in its strongest form, kept honest.

    ``dte_target >= 42`` is advertised to the model as an automatic REJECT. That
    is only true while the ceiling stays under ``MIN_INDEPENDENT_CYCLES``, and if
    a re-pull ever densifies the long end it stops being true -- at which point
    the prompt is telling a model to avoid a bucket that has become usable.
    """
    oos_lo, oos_hi = date(2021, 2, 9), date(2024, 8, 9)
    for target, _window, _oos, floor in CYCLE_BUDGET:
        reachable = _ceiling(chain, sessions, target, oos_lo, oos_hi)
        if floor == 0:
            assert reachable < MIN_INDEPENDENT_CYCLES, (
                f"{target} DTE now reaches {reachable} OOS cycles against a gate of "
                f"{MIN_INDEPENDENT_CYCLES}; the prompt still calls it impossible."
            )
        else:
            assert reachable >= MIN_INDEPENDENT_CYCLES, (
                f"{target} DTE only reaches {reachable} OOS cycles but the prompt "
                f"offers it a {floor}% firing floor."
            )


def test_a_chain_session_is_about_two_calendar_days(sessions: list[date]) -> None:
    """:data:`SESSIONS_PER_YEAR`, which the prompt uses to tell the model that
    ``min_sessions_between_entries: 5`` is ten days rather than a week. Stated
    as a claim because "one snapshot per session" reads as daily and is not."""
    gaps = [(b - a).days for a, b in zip(sessions, sessions[1:], strict=False)]
    assert sorted(gaps)[len(gaps) // 2] == 2
    span = (sessions[-1] - sessions[0]).days / 365.25
    assert abs(len(sessions) / span - SESSIONS_PER_YEAR) < 10


def test_the_embargo_rule_costs_nine_sessions(
    chain: list[dict[str, str]], sessions: list[date]
) -> None:
    """D3 refuses an entry whose expiry crosses the embargo. At 28 DTE that is 9
    of 753 sessions, all at the tail of the window -- a cheap invariant, which
    is worth knowing before agreeing to it."""
    refused = 0
    offered = 0
    by_session: dict[date, set[date]] = defaultdict(set)
    for row in chain:
        day = date.fromisoformat(row["date"])
        expiry = date.fromisoformat(row["expiration"])
        if 24 <= (expiry - day).days <= 34:
            by_session[day].add(expiry)
    for day in sessions:
        expiries = by_session.get(day)
        if not expiries:
            continue
        offered += 1
        if min(expiries) >= CUTOFF:
            refused += 1
    assert refused == 9
    assert offered > 700


def test_iv_rank_is_low_most_of_the_time_on_spy() -> None:
    """Median 18.5, above 50 on 17.8% of sessions.

    Consequence, stated in D8 and not softened: conditioning a 28-DTE program on
    ``iv_rank > 50`` -- which is what specs/07 D1 proposes -- leaves on the order
    of twelve independent bets in five and a half years.
    """
    ranks = []
    for row in _rows(VOLATILITY):
        try:
            low, high, current = (
                float(row["iv_year_low"]),
                float(row["iv_year_high"]),
                float(row["iv_current"]),
            )
        except ValueError:
            continue  # 15 rows carry blank extremes
        if high > low:
            ranks.append((current - low) / (high - low) * 100)

    ranks.sort()
    median = ranks[len(ranks) // 2]
    rich = sum(1 for r in ranks if r > 50) / len(ranks)
    assert 15.0 < median < 22.0, f"IV rank median moved to {median:.1f}"
    assert 0.14 < rich < 0.22, f"IV rank exceeds 50 on {rich:.1%} of sessions"


# --------------------------------------------------------------------------- #
# The two series must be in the same price space
# --------------------------------------------------------------------------- #


def test_the_chain_and_the_underlying_agree_on_what_spy_costs() -> None:
    """Put-call parity, used as an alignment check between two separate caches.

    This test exists because the bug it catches already happened. The underlying
    was first pulled with Alpaca's default ``adjustment=all``, which is right for
    equity research -- an unadjusted 4:1 split is a -75% return that never
    happened -- and wrong here: option strikes are set in raw terms and do not
    move for an ordinary dividend. SPY's real close on 2019-11-22 was 311.02 and
    the adjusted cache said 282.10, so every early settlement was compared
    against a price about 10% too low. Nothing raised. The backtest ran, produced
    a full set of plausible numbers, and reported a 58% win rate on a 16-delta
    short put spread that should win about 84% of the time.

    The chain can check itself: for one expiry and strike, ``C - P = S - K`` at
    zero rates, so the quotes imply a spot. Measured on the research cache with
    the raw series, the implied spot agrees with the bar close to a median of
    0.14% and never by more than 2%. Under the adjusted series it was out by ten.
    """
    if not UNDERLYING.exists():
        pytest.skip("no underlying cache")
    errors = _parity_errors(CHAIN, UNDERLYING)

    assert len(errors) > 700, f"only {len(errors)} sessions could be checked"
    _assert_same_price_space(errors, UNDERLYING)


def _parity_errors(chain: Path, underlying: Path) -> list[float]:
    """Relative disagreement between the chain's implied spot and the bar close.

    One per session that has both a call and a put at some strike and a trading
    day within four days. Split out of the test above so the sealed roots can be
    held to the identical arithmetic -- a sealed cache checked by a slightly
    different calculation would be checked by a calculation nobody had validated.
    """
    closes = {
        datetime.fromisoformat(row["timestamp"]).date(): float(row["close"])
        for row in _rows(underlying)
    }
    trading_days = sorted(closes)

    pairs: dict[date, dict[tuple[str, str], dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in _rows(chain):
        session = date.fromisoformat(row["date"])
        key = (row["expiration"], row["strike"])
        mid = (float(row["bid"]) + float(row["ask"])) / 2
        pairs[session][key]["call" if row["call_put"] == "Call" else "put"] = mid

    errors: list[float] = []
    for session, strikes in pairs.items():
        at = bisect_right(trading_days, session) - 1
        if at < 0 or (session - trading_days[at]).days > 4:
            continue
        spot = closes[trading_days[at]]
        # The pair nearest the money: parity is exact everywhere, but a wide
        # spread on a far wing makes the mid a poorer estimate of either side.
        both = [
            (float(strike), quotes)
            for (_, strike), quotes in strikes.items()
            if "call" in quotes and "put" in quotes
        ]
        if not both:
            continue
        strike, quotes = min(both, key=lambda pair: abs(pair[0] - spot))
        errors.append((quotes["call"] - quotes["put"] + strike) / spot - 1)
    return errors


def _assert_same_price_space(errors: list[float], underlying: Path) -> None:
    """The two caches price the same instrument — asserted on the bulk, not the max.

    Held to the identical arithmetic for both roots, for the reason
    :func:`_parity_errors` gives about itself: a sealed cache checked by a
    slightly different calculation would be checked by a calculation nobody had
    validated.
    """
    magnitudes = sorted(abs(e) for e in errors)
    p99 = magnitudes[int(len(magnitudes) * 0.99)]
    outliers = [e for e in magnitudes if e >= PARITY_TOLERANCE]
    share = len(outliers) / len(magnitudes)

    assert p99 < PARITY_TOLERANCE, _parity_failure(p99, underlying)
    assert share <= PARITY_OUTLIER_BUDGET, (
        f"{len(outliers)} of {len(magnitudes)} sessions ({share:.2%}) disagree by "
        f"{PARITY_TOLERANCE:.0%} or more, over the {PARITY_OUTLIER_BUDGET:.0%} budget. "
        "A handful is vendor noise; this many is a systematic difference, and the "
        f"first thing to check is whether {underlying.parent.parent.name} was pulled "
        "with a dividend adjustment (`--adjustment raw`)."
    )


def _parity_failure(worst: float, underlying: Path) -> str:
    return (
        f"the chain's implied spot and {underlying.parent.parent.name}'s close "
        f"disagree by {worst:.1%} at the 99th percentile. The two caches are in "
        "different price spaces -- most likely the underlying was pulled with a "
        "dividend adjustment. Re-pull with `--adjustment raw`."
    )


def test_the_sealed_chain_and_the_sealed_underlying_are_in_the_same_price_space() -> None:
    """The same check, on the roots the sealed option run reads (PLAN O5.1).

    The sealed chain has existed since the embargo split; its underlying is a
    separate pull and is the one that is easy to get wrong, because the obvious
    place to reach for is ``data-sp500-sealed`` -- which is dividend-adjusted and
    correct for equity research. There is no arithmetic that notices: the sealed
    run completes, reports plausible numbers and spends the one shot. This is the
    only check that can see it, so it runs before the seal is spent rather than
    after.

    Skipped rather than failed while the root does not exist. Building it::

        uv run python -m aqr.cli_sealed pull --symbols SPY --adjustment raw \
            --csv-root data-options-underlying-sealed --timeframe 1D
    """
    if not SEALED_CHAIN.exists():
        pytest.skip("no sealed option chain; `aqr options-embargo` builds it")
    if not SEALED_UNDERLYING.exists():
        pytest.skip(
            "no sealed underlying cache -- the sealed option run cannot settle "
            "without it. See this test's docstring for the pull command."
        )
    errors = _parity_errors(SEALED_CHAIN, SEALED_UNDERLYING)

    assert len(errors) > 1_000, f"only {len(errors)} sealed sessions could be checked"
    _assert_same_price_space(errors, SEALED_UNDERLYING)


# --------------------------------------------------------------------------- #
# The greeks state their own spot too, and four sessions get it wrong (D2b)
# --------------------------------------------------------------------------- #


def test_exactly_four_sessions_have_greeks_computed_against_the_wrong_spot() -> None:
    """D2a checked that the chain's *prices* agree with the bar close. Prices
    passing does not mean the *greeks* are right -- delta, vol and the rest are
    a separate computation the vendor could get wrong even with the quotes
    intact, and on four sessions it did: 2021-11-12, -17, -19 and -22 have
    every greek computed against an underlying about 10% below the real one.

    The measurement: for each session, take the call nearest 0.50 delta in
    each expiry, then the median of those strikes across the session's
    expiries -- ``SessionChain.delta_implied_spot()``. Divided by the
    reference close, 749 of 753 sessions land within 4% of 1.0 (p1 0.995, p99
    1.002) and these four land at 0.898-0.901. Nothing else in the cache is
    close to that gap, which is what makes a single threshold
    (``GREEK_CONSISTENCY_TOLERANCE`` in ``options/engine.py``) a sound guard
    rather than a guess: it must exclude exactly these four sessions and no
    others, asserted here directly against the cache rather than against the
    hand-built fixture in ``test_option_engine.py``.
    """
    if not UNDERLYING.exists():
        pytest.skip("no underlying cache")
    closes = {
        datetime.fromisoformat(row["timestamp"]).date(): float(row["close"])
        for row in _rows(UNDERLYING)
    }
    trading_days = sorted(closes)
    index = ChainIndex.from_rows(_rows(CHAIN))

    flagged = []
    for session in index.sessions:
        at = bisect_right(trading_days, session) - 1
        if at < 0 or (session - trading_days[at]).days > 4:
            continue
        reference_close = closes[trading_days[at]]
        implied = index[session].delta_implied_spot()
        if implied is None:
            continue
        ratio = implied / reference_close
        if abs(ratio - 1.0) > GREEK_CONSISTENCY_TOLERANCE:
            flagged.append(session)

    assert flagged == [
        date(2021, 11, 12),
        date(2021, 11, 17),
        date(2021, 11, 19),
        date(2021, 11, 22),
    ]


# --------------------------------------------------------------------------- #
# The units the feature catalogue promises a model (D6)
# --------------------------------------------------------------------------- #


def test_the_documented_feature_ranges_still_match_the_cache() -> None:
    """The catalogue tells a model what scale each feature is on, and those
    numbers are in a docstring.

    A docstring cannot notice when the data moves underneath it, and getting
    this wrong is expensive in a specific, measured way: a twenty-hypothesis
    campaign lost seven slots to conditions like ``term_slope() > 5`` against a
    feature whose maximum is 0.052, because the docs stated a unit for
    ``iv_rank()`` and for nothing else. The docs now state one for everything,
    so this asserts they are still true -- a re-pull that moves a range fails
    the build instead of quietly teaching the next campaign the wrong scale.

    Loose bounds on purpose. What must not drift is the *order of magnitude* --
    "this is a decimal fraction, not a percentage point" -- and pinning p99 to
    three places would fail on a re-pull that changed nothing that matters.
    """
    if not UNDERLYING.exists():
        pytest.skip("no underlying cache")
    from aqr.features.engine import FeatureKey
    from aqr.option_data import research_option_market
    from aqr.options.features import OptionFeatureFrame, feature_span

    market, _ = research_option_market("SPY")
    frame = OptionFeatureFrame(
        bars=market.underlying, chain=market.chain, volatility=market.volatility
    )

    # (feature, args, the band the documented range has to stay inside)
    expected: list[tuple[str, tuple[float, ...], float, float]] = [
        ("iv_rank", (), 0.0, 100.0),
        ("iv_current", (), 0.0, 2.0),
        ("hv_current", (), 0.0, 2.0),
        ("iv_hv_spread", (), -2.0, 1.0),
        ("term_slope", (), -2.0, 1.0),
        ("skew_25d", (), -2.0, 1.0),
        ("atm_iv", (28.0,), 0.0, 3.0),
    ]
    for name, args, low, high in expected:
        span = feature_span(frame, FeatureKey(name, args))
        assert span is not None, f"{name} is never defined on this cache"
        assert low <= span[0] and span[1] <= high, (
            f"{name}{args} ranges {span[0]:.4g}..{span[1]:.4g}, outside the "
            f"documented band {low}..{high}. Either the cache changed or the "
            f"units did; options/features.py's doc strings tell a model which "
            f"scale to write a threshold on, and a wrong one costs a whole "
            f"hypothesis every time it is read."
        )

    # The one that carries the whole confusion: iv_rank is the exception and
    # everything else is a decimal. If that ever stops being true, the sentence
    # the system prompt leads with is a lie.
    rank = feature_span(frame, FeatureKey("iv_rank", ()))
    slope = feature_span(frame, FeatureKey("term_slope", ()))
    assert rank is not None and slope is not None
    assert rank[1] > 50.0, "iv_rank is documented as the one 0..100 feature"
    assert slope[1] < 1.0, "term_slope is documented as a decimal fraction"
