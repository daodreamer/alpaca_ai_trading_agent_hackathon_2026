"""A persisted implied-volatility history — the input `iv_rank` needs.

Alpaca serves current greeks but no historical implied volatility, and the
historical *option bars* that `blackscholes.py` could invert are gated behind an
OPRA agreement this account has not signed:

    GET /v1beta1/options/bars → 403 {"message":"OPRA agreement is not signed"}

That is an entitlement, not a bug, and it is not something to route around by
inventing numbers. So the history is **accumulated**: one at-the-money implied
volatility observation per underlying per trading day, appended to a JSONL file
that grows as the agent runs.

The consequence is stated plainly because it changes what the system can claim:
**`iv_rank` is unavailable until `MIN_HISTORY` sessions have been observed.**
Inside a four-day competition window it will not become available from live
observation alone. `MarketRead.iv_rank` is therefore `None` throughout, the
prompt renders it `unmeasured`, and the model is told to prefer declining when
something it needs is unmeasured. What carries the "is premium rich" question in
the meantime is `iv_vs_hv`, which is exactly computable today from the chain's
own greeks and the underlying's own bars, and `hv_rank`, which ranks realised
volatility against its own history and needs no options entitlement at all.

`seed_from_option_bars` exists for the moment the OPRA agreement is signed: it
back-fills the store by inverting Black–Scholes over one contract's daily
closes. One call, and `iv_rank` becomes real.

One observation per day, not per cycle. Sixteen cycles a day would make a
"20-observation history" a day and a half of one afternoon's weather, and the
rank would be measuring the lunch lull.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

from alphagate.core.identifiers import Ticker
from alphagate.marketdata.port import MarketData
from alphagate.options import OptionContract, Right
from alphagate.options.blackscholes import implied_volatility, time_to_expiry_years

__all__ = ["OPRA_HINT", "IvHistoryStore"]

OPRA_HINT: Final = (
    "historical option bars need a signed OPRA agreement on the Alpaca account; "
    "until then the IV history is accumulated one session at a time"
)


@dataclass(frozen=True, slots=True)
class IvHistoryStore:
    """One JSONL file per underlying: `{"day": ..., "iv": ...}` per line."""

    directory: Path

    def path_for(self, symbol: Ticker) -> Path:
        return self.directory / f"iv_{symbol}.jsonl"

    def observations(self, symbol: Ticker) -> list[float]:
        """The history, oldest first. Missing file means no history, not zero."""
        return [value for _, value in self._entries(symbol)]

    def latest(self, symbol: Ticker) -> tuple[date, float] | None:
        """The most recent observation, with its date. `None` when there is none.

        `observations()` drops the date because most callers only want the
        series; the staleness check `agent/perceive.py` runs before trusting a
        rank against this series needs the date back — a vendor row seeded two
        months ago is not "no history", but it is not current either, and
        `iv_rank` must be able to tell the two apart (specs/07 D3).
        """
        entries = self._entries(symbol)
        return entries[-1] if entries else None

    def record(self, symbol: Ticker, day: date, implied: float) -> bool:
        """Append one observation. Returns False if the day is already recorded.

        Idempotent per day on purpose: the agent runs many cycles a session, and
        every one of them would otherwise add a point. Twenty points would then
        be a day and a half rather than a month, and the rank would be measuring
        the lunch lull.
        """
        if implied <= 0 or not _finite(implied):
            return False
        if any(recorded == day for recorded, _ in self._entries(symbol)):
            return False
        path = self.path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"day": day.isoformat(), "iv": implied}) + "\n")
            handle.flush()
        return True

    def seed_from_option_bars(
        self,
        data: MarketData,
        symbol: Ticker,
        contract: OptionContract,
        closes_by_day: Mapping[date, Decimal],
        *,
        start: date,
        end: date,
        rate: float = 0.04,
    ) -> int:
        """Back-fill by inverting Black–Scholes over one contract's daily bars.

        Returns how many observations were added. Zero is the expected answer
        while the OPRA agreement is unsigned — the call raises no exception
        because an unentitled account is a configuration fact, not a failure of
        this cycle.

        The approximation is the same one `perceive.iv_history` documents: a
        single fixed contract drifts along the skew and shortens in maturity, so
        this is a reconstruction rather than an observation, and the store marks
        it as seeded.
        """
        from alphagate.agent.perceive import OPTIONS_HISTORY_FLOOR

        try:
            bars = data.option_daily_bars(
                contract, start=max(start, OPTIONS_HISTORY_FLOOR), end=end
            )
        except Exception:
            return 0

        added = 0
        for bar in bars:
            underlying = closes_by_day.get(bar.session_date)
            if underlying is None or bar.close <= 0:
                continue
            years = time_to_expiry_years(contract.days_to_expiry(bar.session_date))
            vol = implied_volatility(
                price=float(bar.close),
                spot=float(underlying),
                strike=float(contract.strike),
                years=years,
                is_call=contract.right is Right.CALL,
                rate=rate,
            )
            if vol is not None and self.record(symbol, bar.session_date, vol):
                added += 1
        return added

    def seed_from_vendor_history(
        self,
        symbol: Ticker,
        rows: Iterable[Mapping[str, str]],
        *,
        column: str = "iv_current",
        since: date | None = None,
    ) -> int:
        """Back-fill from a vendor volatility table. Returns observations added.

        The other way out of the OPRA problem, and the one that is actually
        available: the free DoltHub table behind `ai_quant_researcher`'s research
        cache carries one at-the-money implied volatility per session for SPY
        back to 2019, which is exactly what this store wants and needs no
        entitlement at all.

        **Why this does not breach CLAUDE.md §2b.** It takes parsed rows, not a
        path and not a researcher object: the caller reads a CSV and hands over
        mappings, so the seam between the two projects stays a file in this
        direction too. Nothing here imports `aqr`, and `tests/test_boundaries.py`
        guard 9 still holds.

        **Why the window matters, and why `since` is not optional in practice.**
        `options/volatility.py` ranks against the whole history it is given, so
        seeding a trailing *year* makes `iv_rank` mean what the researched rule
        meant by it — current IV against its own one-year range, the vendor's
        own `(iv_current - iv_year_low) / (iv_year_high - iv_year_low)`. Seeding
        seven years would rank against a range containing March 2020 and answer
        a different question under the same name, which is the failure specs/07
        D3 refuses for unmeasurable features and must equally refuse here.

        Idempotent per day, through `record`. Re-running after a data refresh
        adds only the sessions that are new.
        """
        added = 0
        for row in rows:
            raw_day = str(row.get("date", "")).strip()
            raw_iv = str(row.get(column, "")).strip()
            if not raw_day or not raw_iv:
                continue
            try:
                day = date.fromisoformat(raw_day)
                implied = float(raw_iv)
            except ValueError:
                # A vendor blank is a missing observation, not a zero. Skipping
                # it leaves a gap the rank tolerates; recording 0.0 would put an
                # impossible low into the range and flatter every later rank.
                continue
            if since is not None and day < since:
                continue
            if self.record(symbol, day, implied):
                added += 1
        return added

    def _entries(self, symbol: Ticker) -> Sequence[tuple[date, float]]:
        path = self.path_for(symbol)
        if not path.is_file():
            return ()
        entries: list[tuple[date, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                entries.append((date.fromisoformat(parsed["day"]), float(parsed["iv"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A truncated tail, same discipline as the journal (specs/06 D1).
                continue
        entries.sort(key=lambda pair: pair[0])
        return tuple(entries)


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
