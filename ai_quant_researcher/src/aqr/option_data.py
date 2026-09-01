"""Assembling an :class:`~aqr.options.run.OptionMarket` from the cache on disk.

``options/`` is a pure layer and never opens a file (tests/test_boundaries.py),
and ``data/`` must not import ``options/`` — the chain index imports ``Bars``,
so the arrow already points that way and reversing it would close a cycle. So
the one function that needs both lives here, at the top level, beside
``pipeline.py`` and ``target_book.py`` for the same reason those do: it is
composition, not a layer.

Two entry points, and the split is the seal's, not a convenience:

:func:`research_option_market` reads the truncated roots through
``load_research`` and ``ResearchProvider``. It cannot see an embargoed session
because the rows are not on disk *and* because the provider clamps — the two
locks ``data/embargo.py`` argues for.

:func:`sealed_option_market` takes an already-wrapped provider from its caller.
It does not mint a :class:`~aqr.data.embargo.SealToken` and must not: the token
is constructible in ``embargo.py`` and ``cli_sealed.py`` and nowhere else
(``test_no_module_outside_the_embargo_layer_constructs_a_seal_token``), which is
what keeps "search" and "read the embargoed years" in two binaries. This
function is reachable from the research process; what it cannot do there is
succeed, because ``load_sealed`` refuses outside the sealed phase.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from aqr.data.bars import Bars
from aqr.data.embargo import ResearchProvider
from aqr.data.option_embargo import (
    CHAIN_TABLE,
    VOLATILITY_TABLE,
    load_research,
    load_sealed,
)
from aqr.data.providers import CsvProvider, Provider
from aqr.options.chain import ChainIndex
from aqr.options.features import VolatilityHistory
from aqr.options.run import OptionMarket

__all__ = [
    "DEFAULT_OPTIONS_ROOT",
    "DEFAULT_OPTIONS_SEALED_ROOT",
    "DEFAULT_UNDERLYING_ROOT",
    "DEFAULT_UNDERLYING_SEALED_ROOT",
    "option_dataset_version",
    "research_option_market",
    "sealed_option_market",
]

DEFAULT_OPTIONS_ROOT = "data-options"
DEFAULT_OPTIONS_SEALED_ROOT = "data-options-sealed"
DEFAULT_UNDERLYING_ROOT = "data-options-underlying"
DEFAULT_UNDERLYING_SEALED_ROOT = "data-options-underlying-sealed"
"""Four roots, two of them mirrors of the other two.

The underlying has its own pair, separate from ``data-sp500``, because specs/10
D0 requires it to be pulled **raw**: option strikes are set in raw terms and do
not move for an ordinary dividend, so a dividend-adjusted close compared against
a strike reports a moneyness the trade never had. SPY's real close on 2019-11-22
was 311.02 and the adjusted series says 282.10 — a ten percent error, silent,
and the backtest completes and reports plausible numbers either way. The choice
travels in the dataset version (``alpaca:sip:raw:...``) and is checked by
put-call parity against the chain (D2a).
"""

_EARLIEST = datetime(1990, 1, 1, tzinfo=UTC)
"""Load the whole underlying file rather than a window.

The engine settles a structure against the close on its expiration date and
warms bar features (``sma(200)``) before the first chain session, so both ends
of the underlying series are load-bearing and neither is derivable from the
chain's own span. The provider clamps the far end at the embargo in the research
phase, so "everything" means "everything a search may see".
"""


def option_dataset_version(
    underlying_version: str, chain_root: str | Path, sessions: int
) -> str:
    """What produced this market, recorded with every experiment.

    Names both halves, because an option result depends on both and a version
    that named only the bars would be identical across a chain re-pull. The
    session count is the cheap tripwire: 753 is the research window and 1,260 is
    the sealed one, so a run that silently read the wrong root says so in every
    row it writes.
    """
    return f"{underlying_version}+chain:{Path(chain_root).name}:{sessions}sessions"


def research_option_market(
    underlying: str = "SPY",
    *,
    chain_root: str | Path = DEFAULT_OPTIONS_ROOT,
    underlying_root: str | Path = DEFAULT_UNDERLYING_ROOT,
    timeframe: str = "1D",
    before: date | None = None,
) -> tuple[OptionMarket, str]:
    """The research market: truncated chain, clamped bars, and its dataset version.

    ``before`` drops chain sessions on or after a boundary. That is the crude,
    disk-level half of D3's embargo and it is *not* the one that matters — the
    engine still refuses an entry whose **expiry** crosses the boundary, which
    no session filter can see. It exists so a caller holding a sealed root can
    index only the research half without writing the filter itself.
    """
    chain, volatility = _read_chain(underlying, Path(chain_root), before, sealed=False)
    provider = ResearchProvider(CsvProvider(underlying_root), label="options-underlying")
    bars = _load_bars(provider, underlying, timeframe)
    version = option_dataset_version(
        _version(provider, timeframe), chain_root, len(chain.sessions)
    )
    return OptionMarket(underlying=bars, chain=chain, volatility=volatility), version


def sealed_option_market(
    underlying: str = "SPY",
    *,
    provider: Provider,
    chain_root: str | Path = DEFAULT_OPTIONS_SEALED_ROOT,
    timeframe: str = "1D",
    since: date | None = None,
) -> tuple[OptionMarket, str]:
    """The sealed market. ``provider`` must already be sealed by its caller.

    The chain is read whole rather than clipped to ``since``: a sealed
    measurement scores the sessions from the embargo onward but the rule's
    features need history before it to be warm, exactly as the equity sealed run
    loads 1,200 days of warm-up. ``since`` is therefore not a load parameter and
    is accepted only to be recorded — a caller that wanted the clip would be
    asking for a cold-started rule.
    """
    chain, volatility = _read_chain(underlying, Path(chain_root), None, sealed=True)
    bars = _load_bars(provider, underlying, timeframe)
    version = option_dataset_version(
        _version(provider, timeframe), chain_root, len(chain.sessions)
    )
    if since is not None:
        version = f"{version}@since-{since.isoformat()}"
    return OptionMarket(underlying=bars, chain=chain, volatility=volatility), version


# --------------------------------------------------------------------------- #


def _read_chain(
    symbol: str, root: Path, before: date | None, *, sealed: bool
) -> tuple[ChainIndex, VolatilityHistory | None]:
    """Both vendor tables, through the container that reports them to the seal.

    ``volatility_history`` is optional and its absence is not an error: specs/10
    D6's features go ``NaN`` without it and a rule that does not name one runs
    unchanged. Refusing here would make the chain unusable for the majority of
    rules that only read bar features.
    """
    read = load_sealed if sealed else load_research
    held = read(symbol, table=CHAIN_TABLE, root=root)
    chain = ChainIndex.from_rows(held.as_dicts(), before=before)

    volatility: VolatilityHistory | None = None
    try:
        vol_rows = read(symbol, table=VOLATILITY_TABLE, root=root)
    except FileNotFoundError:
        return chain, None
    volatility = VolatilityHistory.from_rows(vol_rows.as_dicts(), before=before)
    return chain, volatility


def _load_bars(provider: Provider, symbol: str, timeframe: str) -> Bars:
    end = datetime.now(UTC)
    return provider.load(symbol, _EARLIEST, end, timeframe)


def _version(provider: Provider, timeframe: str) -> str:
    """What the underlying half of the market is, named as precisely as it can be.

    A provider that can describe itself does (``AlpacaProvider`` names its feed
    *and* its price adjustment, which is the one that matters here — specs/10 D0
    and D2a). ``CsvProvider`` cannot: the adjustment was decided when the cache
    was pulled and is not in the file. What is available is which root was read,
    and that is the checkable thing, because ``data-options-underlying`` is the
    raw cache and ``data-sp500`` is the adjusted one. Naming the root at least
    makes "this run read the wrong cache" visible in every recorded row; the
    parity check in ``tests/test_option_cache_claims.py`` is what makes it
    *impossible* to read the wrong one and not notice.

    ``ResearchProvider.dataset_version`` delegates to its inner provider and so
    raises for a CSV cache. Caught rather than avoided: the fallback is the
    honest answer, and a caller should not have to know which providers can
    describe themselves.
    """
    describe = getattr(provider, "dataset_version", None)
    if callable(describe):
        try:
            return str(describe(timeframe))
        except AttributeError:
            pass
    return f"csv:{_root_of(provider)}:{timeframe}"


def _root_of(provider: object) -> str:
    """The cache directory behind a provider, through any wrapper."""
    inner = getattr(provider, "_inner", provider)
    root = getattr(inner, "root", None)
    return Path(root).name if root is not None else type(inner).__name__
