"""The options engine — specs/10-options-research.md D1–D3, test plan 4–15.

The engine is the correctness surface, so most of what is asserted here is a
refusal or an invariant rather than a number:

* a decision may only see bars that closed before the fill (D2),
* an entry whose settlement would read the reserved window is refused (D3),
* the crossed spread is charged in full and never a mid (D2),
* the money never moves when the pricing model moves (D1a).

The fixture is hand-built rather than pulled from the cache. Every price here is
a number a reader can multiply out, which is the only way a settlement test
proves anything.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from aqr.backtest.engine import BacktestResult
from aqr.backtest.metrics import compute_metrics
from aqr.data.bars import Bars
from aqr.options.chain import ChainIndex
from aqr.options.engine import (
    OptionBacktestConfig,
    SkipCensus,
    affordability_bound_fraction,
    run_option_backtest,
)
from aqr.options.spec import Anchor, Cadence, DteTarget, OptionSizing, OptionSpec, StructureSpec

# --------------------------------------------------------------------------- #
# A small, exactly-known world
# --------------------------------------------------------------------------- #

FIRST = date(2023, 1, 2)
SPOT = 400.0


def trading_days(n: int, start: date = FIRST) -> list[date]:
    """Weekdays only, which is what makes 'the bar before' unambiguous."""
    days: list[date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def make_underlying(closes: dict[date, float], days: list[date]) -> Bars:
    price = np.array([closes.get(d, SPOT) for d in days], dtype=np.float64)
    stamps = np.array(
        [int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) for d in days],
        dtype=np.int64,
    )
    return Bars(
        symbol="SPY",
        timeframe="1D",
        event_time=stamps,
        open=price.copy(),
        high=price.copy(),
        low=price.copy(),
        close=price,
        volume=np.full(len(days), 1e8),
    )


def chain_row(
    session: date,
    expiry: date,
    strike: float,
    right: str,
    *,
    bid: float,
    ask: float,
    delta: float,
) -> dict[str, str]:
    return {
        "date": session.isoformat(),
        "act_symbol": "SPY",
        "expiration": expiry.isoformat(),
        "strike": f"{strike:.2f}",
        "call_put": "Put" if right == "put" else "Call",
        "bid": f"{bid:.2f}",
        "ask": f"{ask:.2f}",
        "vol": "0.20",
        "delta": f"{delta:.4f}",
        "gamma": "0.0010",
        "theta": "-0.0500",
        "vega": "0.0900",
        "rho": "0.0200",
    }


def build_world(
    *,
    sessions: list[date],
    days: list[date],
    dte: int = 28,
    closes: dict[date, float] | None = None,
) -> tuple[ChainIndex, Bars]:
    """One expiry per session at ``dte`` days out, a three-strike put ladder.

    Prices are chosen so the arithmetic is checkable by hand: the 16-delta put
    is 1.50/1.60 and the 9-delta put ten points below it is 0.45/0.50, so a
    credit spread collects exactly 1.00 on a 10-point width.
    """
    rows: list[dict[str, str]] = []
    for session in sessions:
        expiry = session + timedelta(days=dte)
        while expiry.weekday() >= 5:  # settlement needs a trading day
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            chain_row(session, expiry, 370.0, "put", bid=0.20, ask=0.25, delta=-0.04),
            chain_row(session, expiry, 410.0, "call", bid=1.40, ask=1.50, delta=0.16),
            chain_row(session, expiry, 420.0, "call", bid=0.40, ask=0.45, delta=0.09),
        ]
    return ChainIndex.from_rows(rows), make_underlying(closes or {}, days)


def credit_spread_spec(**overrides: object) -> OptionSpec:
    fields: dict[str, object] = {
        "name": "test_put_credit_spread",
        "underlying": "SPY",
        "entry": "close > 0",  # always true; the engine's gating is tested separately
        "structure": StructureSpec(
            type="put_credit_spread",
            dte=DteTarget(target=28, tolerance=10),
            anchor=Anchor(delta=0.16, tolerance=0.06),
            width_points=10.0,
        ),
        "sizing": OptionSizing(risk_per_trade=0.01, max_concurrent=1),
        "cadence": Cadence(min_sessions_between_entries=1),
    }
    fields.update(overrides)
    return OptionSpec(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The fill convention (D2)
# --------------------------------------------------------------------------- #


def test_entry_collects_the_bid_and_pays_the_ask() -> None:
    """1.50 in, 0.50 out, on a 10-wide spread. Credit 1.00, risk 9.00.

    A mid-price fill would collect 1.55 and pay 0.475 -- a 7.5% better entry on
    every trade, compounding into a strategy that does not exist.
    """
    days = trading_days(80)
    # days[5], not days[0]: the first bar has nothing before it, so no decision
    # bar exists and the engine correctly declines to trade there.
    result = run_option_backtest(credit_spread_spec(), *build_world(sessions=days[5:6], days=days))
    (trade,) = result.option_trades
    assert trade.entry_cash == pytest.approx(1.00)
    assert trade.max_loss == pytest.approx(9.00)


def test_the_decision_bar_closes_before_the_fill() -> None:
    """D2: decide on information available at t-1, fill from t's chain.

    Asserted on the dates rather than on a result, because it is the property
    every other number depends on.
    """
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    (trade,) = run_option_backtest(credit_spread_spec(), chain, bars).option_trades
    assert trade.decision_bar < trade.entry_reference
    assert trade.entry_reference == trade.entry_session
    assert trade.decision_bar == days[4]


def test_a_session_that_is_not_a_trading_day_still_decides_from_an_earlier_bar() -> None:
    """2019's snapshots are dated Saturday and reflect Friday's close (D0).

    Deciding from Friday and filling on a Saturday snapshot would be trading on
    the close at the close: no future information, but no time to act either.
    So the decision bar is the one before the session's *reference* bar, which
    for a Saturday means Thursday.
    """
    days = trading_days(80)
    saturday = days[5] + timedelta(days=(5 - days[5].weekday()) % 7 or 7)
    chain, bars = build_world(sessions=[saturday], days=days)
    (trade,) = run_option_backtest(credit_spread_spec(), chain, bars).option_trades
    assert trade.entry_session == saturday
    assert trade.entry_reference < saturday
    assert trade.decision_bar < trade.entry_reference


def test_a_session_with_no_recent_bar_is_skipped() -> None:
    """More than four days from the last close, the reference price is a guess."""
    days = trading_days(60)
    stale = days[-1] + timedelta(days=30)
    chain, bars = build_world(sessions=[stale], days=days)
    assert run_option_backtest(credit_spread_spec(), chain, bars).option_trades == []


# --------------------------------------------------------------------------- #
# Settlement (D1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("close_at_expiry", "pnl_per_share"),
    [
        (400.0, 1.00),  # above both strikes: keep the credit
        (390.0, 1.00),  # exactly at the short strike
        (385.0, -4.00),  # halfway through
        (380.0, -9.00),  # at the long strike: maximum loss
        (350.0, -9.00),  # beyond it: still capped
    ],
)
def test_settlement_is_the_underlying_close_on_the_expiration_date(
    close_at_expiry: float, pnl_per_share: float
) -> None:
    days = trading_days(80)
    session = days[5]
    expiry = session + timedelta(days=28)
    while expiry.weekday() >= 5:
        expiry += timedelta(days=1)
    chain, bars = build_world(sessions=[session], days=days, closes={expiry: close_at_expiry})
    (trade,) = run_option_backtest(credit_spread_spec(), chain, bars).option_trades
    assert trade.expiration == expiry
    assert trade.entry_cash + trade.settlement_cash == pytest.approx(pnl_per_share)


def test_realised_pnl_is_the_per_share_result_times_contracts_and_the_multiplier() -> None:
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    (trade,) = result.option_trades
    expected = (trade.entry_cash + trade.settlement_cash) * 100 * trade.contracts
    assert trade.gross_pnl == pytest.approx(expected)
    assert trade.net_pnl == pytest.approx(expected - trade.fees)


# --------------------------------------------------------------------------- #
# The embargo (D3)
# --------------------------------------------------------------------------- #


def test_an_entry_whose_expiry_crosses_the_boundary_is_refused() -> None:
    """Settlement would read the reserved window. Refused at selection time,
    with the expiry named -- not truncated afterwards, when the decision has
    already been made."""
    days = trading_days(80)
    session = days[5]
    chain, bars = build_world(sessions=[session], days=days)
    config = OptionBacktestConfig(settle_before=session + timedelta(days=1))
    result = run_option_backtest(credit_spread_spec(), chain, bars, config)
    assert result.option_trades == []
    assert any("expir" in reason for reason in result.skipped_reasons)


def test_entries_before_the_boundary_are_untouched_by_it() -> None:
    days = trading_days(120)
    chain, bars = build_world(sessions=days[5:6], days=days)
    far = OptionBacktestConfig(settle_before=days[-1])
    assert len(run_option_backtest(credit_spread_spec(), chain, bars, far).option_trades) == 1


# --------------------------------------------------------------------------- #
# The greek-consistency guard (D2b)
# --------------------------------------------------------------------------- #


def test_a_session_with_inconsistent_greeks_is_skipped_not_traded() -> None:
    """Four real sessions (2021-11-12, -17, -19, -22) have greeks the vendor
    computed against a spot about 10% below the real one; prices are correct
    (D2a), so nothing about selection or settlement notices on its own. This
    reproduces the shape by hand: an ATM call whose delta says the session's
    spot is 360 when the reference close says 400. The engine must refuse the
    whole session rather than build a structure priced and sized against a
    spot that greeks alone got wrong, and it must trade the next, honest
    session normally.
    """
    days = trading_days(80)
    good, bad = days[5], days[7]
    rows: list[dict[str, str]] = []
    for session, atm_strike in ((good, 400.0), (bad, 360.0)):
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            chain_row(session, expiry, atm_strike, "call", bid=10.00, ask=10.10, delta=0.50),
        ]
    chain = ChainIndex.from_rows(rows)
    bars = make_underlying({}, days)  # flat at SPOT=400 on every day
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=2),
    )
    result = run_option_backtest(spec, chain, bars)

    entered = {t.entry_session for t in result.option_trades}
    assert good in entered
    assert bad not in entered
    assert any(
        str(bad) in reason and "D2b" in reason for reason in result.skipped_reasons
    )


def test_a_session_with_no_near_the_money_call_is_not_flagged() -> None:
    """``delta_implied_spot()`` returns ``None`` when the ladder has nothing
    near 0.50 delta -- build_world's default fixture, which only carries the
    16- and 9-delta legs a credit spread needs. The guard must not turn a
    thin call ladder into a false refusal on every existing engine test."""
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    assert len(result.option_trades) == 1
    assert not any("D2b" in reason for reason in result.skipped_reasons)


# --------------------------------------------------------------------------- #
# Sizing, cadence, concurrency (D4, D5)
# --------------------------------------------------------------------------- #


def test_size_follows_from_the_risk_budget_and_the_structures_maximum_loss() -> None:
    """1% of 100,000 is 1,000; risk per contract is 9.00 * 100 = 900. One contract.

    Never two: the fraction is a ceiling on the loss, so the count rounds down.
    """
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    (trade,) = run_option_backtest(credit_spread_spec(), chain, bars).option_trades
    assert trade.contracts == 1


def test_a_risk_budget_too_small_for_one_contract_skips_the_entry() -> None:
    """Rounding up to one would take four times the risk the rule authorised."""
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    spec = credit_spread_spec(sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=1))
    assert run_option_backtest(spec, chain, bars).option_trades == []


def test_cadence_holds_entries_apart() -> None:
    days = trading_days(200)
    sessions = days[5:25]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=10),
    )
    entered = [t.entry_session for t in run_option_backtest(spec, chain, bars).option_trades]
    positions = [sessions.index(s) for s in entered]
    assert all(b - a >= 5 for a, b in zip(positions, positions[1:], strict=False))


def test_concurrency_is_capped() -> None:
    days = trading_days(200)
    sessions = days[5:40]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=2),
    )
    result = run_option_backtest(spec, chain, bars)
    for trade in result.option_trades:
        overlapping = sum(
            1
            for other in result.option_trades
            if other.entry_session <= trade.entry_session < other.expiration
        )
        assert overlapping <= 2


# --------------------------------------------------------------------------- #
# The skip census (D8a)
# --------------------------------------------------------------------------- #


def test_the_affordability_skip_is_categorised_and_the_reason_matches() -> None:
    """The same fixture as ``test_a_risk_budget_too_small_for_one_contract_skips_the_entry``,
    read through the census instead of the trade list: the risk budget (0.2%
    of 100,000 is 200, against 900 of risk per contract) cannot cover one
    contract, so this must be the *only* thing that skipped, and it must be
    counted as ``affordability`` -- not left as a string only a human could
    categorise by reading it.
    """
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    spec = credit_spread_spec(sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=1))
    result = run_option_backtest(spec, chain, bars)
    assert result.option_trades == []
    assert result.skip_census.affordability == 1
    assert result.skip_census.total == 1
    (entry,) = result.skips
    assert entry.category == "affordability"
    assert entry.reason == result.skipped_reasons[0]


def test_the_embargo_skip_is_categorised_as_embargo() -> None:
    days = trading_days(80)
    session = days[5]
    chain, bars = build_world(sessions=[session], days=days)
    config = OptionBacktestConfig(settle_before=session + timedelta(days=1))
    result = run_option_backtest(credit_spread_spec(), chain, bars, config)
    assert result.option_trades == []
    assert result.skip_census.embargo_refusal == 1
    assert result.skip_census == result.skip_census.__class__(embargo_refusal=1)


def test_the_greek_consistency_skip_is_categorised_separately_from_a_stale_reference() -> None:
    """Reuses the D2b fixture: one good session, one whose greeks were
    computed against a spot 10% off. The good session trades; the bad one is
    counted under ``greek_consistency`` and nothing else."""
    days = trading_days(80)
    good, bad = days[5], days[7]
    rows: list[dict[str, str]] = []
    for session, atm_strike in ((good, 400.0), (bad, 360.0)):
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            chain_row(session, expiry, atm_strike, "call", bid=10.00, ask=10.10, delta=0.50),
        ]
    chain = ChainIndex.from_rows(rows)
    bars = make_underlying({}, days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=2),
    )
    result = run_option_backtest(spec, chain, bars)
    assert len(result.option_trades) == 1
    assert result.skip_census.greek_consistency == 1
    assert result.skip_census.stale_underlying == 0


def test_a_stale_underlying_reference_is_categorised_as_stale_underlying() -> None:
    days = trading_days(60)
    stale = days[-1] + timedelta(days=30)
    chain, bars = build_world(sessions=[stale], days=days)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    assert result.option_trades == []
    assert result.skip_census.stale_underlying == 1
    assert result.skip_census.total == 1


def test_a_delta_the_ladder_never_offers_is_categorised_as_no_leg_or_wing() -> None:
    """``build_world``'s fixture only ever quotes 16- and 9-delta legs.
    Naming a 35-delta anchor at a 2% tolerance asks for a leg the ladder
    never carries, which is a different failure from a missing expiry and
    must be counted differently."""
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    spec = credit_spread_spec(
        structure=StructureSpec(
            type="put_credit_spread",
            dte=DteTarget(target=28, tolerance=10),
            anchor=Anchor(delta=0.35, tolerance=0.02),
            width_points=10.0,
        )
    )
    result = run_option_backtest(spec, chain, bars)
    assert result.option_trades == []
    assert result.skip_census.no_leg_or_wing == 1
    assert result.skip_census.no_expiry_in_band == 0


def test_an_expiry_outside_the_dte_band_is_categorised_as_no_expiry_in_band() -> None:
    """``build_world`` quotes one expiry, ~28 DTE out. Asking for ~90 DTE at
    a 5-day tolerance means no listed expiry answers at all -- refused before
    delta selection is ever attempted, and counted as the DTE-band failure
    rather than the leg failure."""
    days = trading_days(80)
    chain, bars = build_world(sessions=days[5:6], days=days)
    spec = credit_spread_spec(
        structure=StructureSpec(
            type="put_credit_spread",
            dte=DteTarget(target=90, tolerance=5),
            anchor=Anchor(delta=0.16, tolerance=0.06),
            width_points=10.0,
        )
    )
    result = run_option_backtest(spec, chain, bars)
    assert result.option_trades == []
    assert result.skip_census.no_expiry_in_band == 1
    assert result.skip_census.no_leg_or_wing == 0


def test_skipped_reasons_and_skips_never_disagree_on_order_or_count() -> None:
    """A run that mixes several skip categories: the flat, formatted list
    D8a keeps for backward compatibility must stay exactly what
    ``skips`` would produce, session by session, so a caller reading either
    view sees the same story."""
    days = trading_days(200)
    sessions = days[5:20]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=10),
    )
    result = run_option_backtest(spec, chain, bars)
    assert result.skipped_reasons == [entry.reason for entry in result.skips]
    assert result.skip_census.total == len(result.skips)
    assert result.skip_census == SkipCensus.from_entries(result.skips)


def test_affordability_bound_fraction_matches_the_spec_worked_example() -> None:
    """specs/10 D8a's own measured table: 578 affordability skips against 31
    trades is 94.9% -- the denominator is skips-plus-trades, not
    skips-over-everything, because every other skip category never reached
    the sizing step this fraction is asking about."""
    census = SkipCensus(affordability=578, no_leg_or_wing=20, embargo_refusal=9)
    assert affordability_bound_fraction(census, 31) == pytest.approx(578 / 609)


def test_affordability_bound_fraction_ignores_skips_that_never_reached_sizing() -> None:
    """A rule starved entirely by a narrow DTE band -- never by the account
    -- must report 0%, not some fraction diluted by skips that were never in
    play for an affordability question."""
    census = SkipCensus(no_expiry_in_band=400, no_leg_or_wing=50)
    assert affordability_bound_fraction(census, 10) == 0.0


def test_affordability_bound_fraction_is_zero_when_nothing_reached_sizing() -> None:
    """No trades and no affordability skips -- every session was refused
    earlier -- must not raise a division by zero and must not read as
    "fully affordability-bound," which would be the wrong direction to fail
    in for a gate that decides whether a REJECT is trustworthy."""
    assert affordability_bound_fraction(SkipCensus(), 0) == 0.0


# --------------------------------------------------------------------------- #
# The curve, and the wall around the model (D1a)
# --------------------------------------------------------------------------- #


def test_the_model_never_moves_the_money() -> None:
    """Test plan 12. Change the pricing input, and every realised number must be
    byte-identical. If this fails, a model has reached the P&L."""
    days = trading_days(200)
    chain, bars = build_world(sessions=days[5:30], days=days)
    spec = credit_spread_spec(sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3))
    cheap = run_option_backtest(spec, chain, bars, OptionBacktestConfig(rate=0.0))
    dear = run_option_backtest(spec, chain, bars, OptionBacktestConfig(rate=0.10))
    assert [t.gross_pnl for t in cheap.option_trades] == [
        t.gross_pnl for t in dear.option_trades
    ]
    assert [t.entry_cash for t in cheap.option_trades] == [
        t.entry_cash for t in dear.option_trades
    ]


def test_the_marked_curve_moves_between_entry_and_expiry() -> None:
    """Test plan 13. Under cash accounting the curve is flat here, the position
    shows no beta, and residual alpha credits the whole return to skill."""
    days = trading_days(120)
    session = days[5]
    falling = {d: SPOT - 2.0 * i for i, d in enumerate(days[6:20])}
    chain, bars = build_world(sessions=[session], days=days, closes=falling)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    # The epoch stamp lives on the generic Trade view; OptionTrade speaks dates,
    # because a chain session is a date and pretending otherwise invents a time.
    entry_at = int(np.searchsorted(result.timeline, result.trades[0].entry_time))
    window = result.equity[entry_at : entry_at + 10]
    assert window.min() < window[0], "a short put spread must lose as the underlying falls"


def test_the_curve_ends_where_the_realised_pnl_says_it_should() -> None:
    """The model and the money agree at settlement, so the curve does not jump
    on the expiration date for a reason nobody can point at."""
    days = trading_days(120)
    chain, bars = build_world(sessions=days[5:6], days=days)
    result = run_option_backtest(credit_spread_spec(), chain, bars)
    (trade,) = result.option_trades
    expected = result.initial_equity + trade.net_pnl
    assert float(result.equity[-1]) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Determinism, and the seam (test plan 14–15)
# --------------------------------------------------------------------------- #


def test_two_runs_produce_identical_results() -> None:
    days = trading_days(200)
    chain, bars = build_world(sessions=days[5:30], days=days)
    spec = credit_spread_spec(sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3))
    first = run_option_backtest(spec, chain, bars)
    second = run_option_backtest(spec, chain, bars)
    assert np.array_equal(first.equity, second.equity)
    assert [t.as_dict() for t in first.option_trades] == [t.as_dict() for t in second.option_trades]


def test_the_result_is_a_backtest_result_the_existing_metrics_can_read() -> None:
    """The one architectural commitment in specs/10. Asserted by running the
    real downstream function, not by checking the type."""
    days = trading_days(250)
    chain, bars = build_world(sessions=days[5:40], days=days)
    spec = credit_spread_spec(sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3))
    result = run_option_backtest(spec, chain, bars)
    assert isinstance(result, BacktestResult)
    metrics = compute_metrics(result)
    assert metrics.num_trades == len(result.option_trades)
    assert np.isfinite(metrics.total_return)
    assert np.isfinite(metrics.max_drawdown)
