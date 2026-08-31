"""Options research — specs/10-options-research.md.

A pure layer, on the same terms as ``core``, ``features``, ``dsl``,
``backtest``, ``validation`` and ``evaluator``: no network, no filesystem, no
clock, no database. Loading option rows is ``data/``'s job, where
``OptionChain.__post_init__`` already reports every session it holds to the
seal; what lives here is what happens to those rows afterwards, and it must be
replayable a year from now with the same answer.
"""

from __future__ import annotations

__all__: list[str] = []
