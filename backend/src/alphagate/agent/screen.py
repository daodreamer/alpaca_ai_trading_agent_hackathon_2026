"""Step 2 — the screen. specs/05 D1.

A deterministic pre-filter: `MarketRead` in, `Setup` or `None` out. Pure, no
clock, no model.

**The concrete rules are specs/07's, deliberately.** specs/05 says so in as many
words — "written separately so that tuning thresholds does not mean editing the
orchestration" — so what lives here is the *interface* and one default that is
honest rather than clever. When specs/07 lands, it supplies a `Screen` and this
file does not change.

The default's whole job is to fail closed. It refuses to produce a `Setup` when
the read is incomplete, when an earnings event is inside the window or nobody
checked, and when the trend was not measured. Every one of those is a case where
the *next* step — handing a menu to a model — would be asking a question the
inputs cannot answer, and D6's rule is that doing nothing is always available
and correct.

What it does *not* do is judge whether premium is rich. That is the strategy's
call and it needs `iv_rank`, which this account cannot compute (see
`iv_store.py`). The default screen passes the read through with a bias taken
from the trend and lets the proposer weigh `iv_vs_hv`; a screen that guessed at
richness would be inventing the very number the rest of the layer refuses to
invent.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Protocol, runtime_checkable

from alphagate.agent.model import MarketRead, Setup

__all__ = ["BIAS_BEARISH", "BIAS_BULLISH", "BIAS_NEUTRAL", "DefaultScreen", "Screen"]

BIAS_BULLISH: Final = "bullish"
BIAS_BEARISH: Final = "bearish"
BIAS_NEUTRAL: Final = "neutral"

_BULLISH_PHASES: Final = frozenset({"STRONG_BULLISH", "BULLISH", "WATCH_BULL"})
_BEARISH_PHASES: Final = frozenset({"STRONG_BEARISH", "BEARISH", "WATCH_BEAR"})


@runtime_checkable
class Screen(Protocol):
    """Turn a read into a setup, or decline to. Pure."""

    def screen(self, read: MarketRead) -> Setup | None: ...


@dataclass(frozen=True, slots=True)
class DefaultScreen:
    """Fail-closed until specs/07 replaces it.

    `min_trend_strength` is the one threshold here, and it is deliberately mild:
    the trend engine has already applied its own confirmation count and exit
    margin, so a second opinion on the same evidence would mostly be a way to
    disagree with a state machine that was built to be disagreed with less.
    """

    min_trend_strength: float = 40.0
    name: str = "default-screen-v1"

    def screen(self, read: MarketRead) -> Setup | None:
        reason = self._refusal(read)
        if reason is not None:
            return None

        bias = self.bias_of(read)
        strength = float(getattr(read.trend, "strength", 0.0))
        phase = _phase_name(read.trend)
        return Setup(
            underlying=read.underlying,
            name=f"{bias}_{phase.lower()}",
            bias=bias,
            reason=(
                f"trend {phase} at strength {strength:.0f}, "
                f"IV/HV {read.iv_vs_hv}, ATR {read.atr_pct}%, no earnings in the window"
            ),
        )

    def _refusal(self, read: MarketRead) -> str | None:
        """Why this read produces no setup. `None` means it produces one.

        Returned rather than raised, and named rather than boolean, because the
        journal's `NO_SETUP` entries are the majority and "why didn't it trade at
        14:30?" is the question they exist to answer.
        """
        if read.trend is None:
            return "trend unmeasured"
        if read.earnings_within_dte is not False:
            return (
                "earnings inside the window"
                if read.earnings_within_dte
                else "no earnings calendar for this underlying"
            )
        if read.atr_pct is None or read.iv_vs_hv is None:
            return "volatility unmeasured"
        if float(getattr(read.trend, "strength", 0.0)) < self.min_trend_strength:
            return f"trend strength below {self.min_trend_strength:.0f}"
        return None

    def explain(self, read: MarketRead) -> str:
        """The refusal reason, for the journal's `note`. Public on purpose."""
        return self._refusal(read) or "setup found"

    @staticmethod
    def bias_of(read: MarketRead) -> str:
        """Which structure family the trend argues for.

        Bullish trend → sell put premium below the market; bearish → sell call
        premium above it. A neutral or unmeasured trend argues for neither
        direction, which is a real answer and not a shrug: the caller builds a
        two-sided structure or nothing.
        """
        phase = _phase_name(read.trend)
        if phase in _BULLISH_PHASES:
            return BIAS_BULLISH
        if phase in _BEARISH_PHASES:
            return BIAS_BEARISH
        return BIAS_NEUTRAL


def _phase_name(trend: object) -> str:
    state = getattr(trend, "state", None)
    return str(getattr(state, "value", state or "UNKNOWN"))


def premium_is_rich(read: MarketRead, *, threshold: Decimal = Decimal("1.05")) -> bool | None:
    """Whether options are pricing more movement than the underlying delivers.

    Offered as a helper for specs/07 rather than used by `DefaultScreen`, and
    tri-state for the usual reason: an unmeasured ratio is not a cheap one.
    """
    if read.iv_vs_hv is None:
        return None
    return read.iv_vs_hv >= threshold
