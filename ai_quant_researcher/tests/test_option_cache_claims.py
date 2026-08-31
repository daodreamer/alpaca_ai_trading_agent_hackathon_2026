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

from aqr.seal import EMBARGO_START

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "data-options" / "option_chain" / "SPY.csv"
VOLATILITY = ROOT / "data-options" / "volatility_history" / "SPY.csv"
UNDERLYING = ROOT / "data-options-underlying" / "1D" / "SPY.csv"

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
    closes = {
        datetime.fromisoformat(row["timestamp"]).date(): float(row["close"])
        for row in _rows(UNDERLYING)
    }
    trading_days = sorted(closes)

    pairs: dict[date, dict[tuple[str, str], dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in _rows(CHAIN):
        session = date.fromisoformat(row["date"])
        key = (row["expiration"], row["strike"])
        mid = (float(row["bid"]) + float(row["ask"])) / 2
        pairs[session][key]["call" if row["call_put"] == "Call" else "put"] = mid

    errors = []
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

    assert len(errors) > 700, f"only {len(errors)} sessions could be checked"
    worst = max(abs(e) for e in errors)
    assert worst < 0.02, (
        f"the chain's implied spot and the bar close disagree by up to {worst:.1%}. "
        "The two caches are in different price spaces -- most likely the underlying "
        "was pulled with a dividend adjustment. Re-pull with `--adjustment raw`."
    )
