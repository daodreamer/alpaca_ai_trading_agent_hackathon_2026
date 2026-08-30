"""Point-in-time market data: containers, providers, and an on-disk cache.

Five providers. ``SyntheticProvider`` and ``CsvProvider`` are offline and are
what the tests and the default research loop use. ``YFinanceProvider``,
``AlpacaProvider`` and ``IbkrProvider`` reach a network and are pulled through
``aqr pull`` into the CSV cache, so that every run after the first is offline
and reproducible.
"""

from aqr.data.alpaca import AlpacaProvider
from aqr.data.bars import Bars, bar_duration, ensure_utc
from aqr.data.ibkr import IbkrProvider
from aqr.data.providers import CsvProvider, Provider, SyntheticProvider, YFinanceProvider

__all__ = [
    "AlpacaProvider",
    "Bars",
    "CsvProvider",
    "IbkrProvider",
    "Provider",
    "SyntheticProvider",
    "YFinanceProvider",
    "bar_duration",
    "ensure_utc",
]
