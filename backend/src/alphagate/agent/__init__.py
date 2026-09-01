"""The agent — specs/05.

The orchestration layer, and the only package permitted to import an LLM SDK
(specs/01 Rule 1).

The layer's whole safety argument is in two type signatures. `Proposer.propose`
receives a `MarketRead` and a list of `Candidate`s and returns a `Choice`; a
`Choice` holds an index or `None`. There is no field anywhere in that path for a
symbol, a strike, a quantity or a price — so a hallucinated contract cannot
reach the broker, and a model cannot size a trade, because neither is
expressible.

`deepseek` is deliberately not re-exported: importing this package should not
require the live model's dependencies to be importable, the same way
`alphagate.execution` does not re-export `stdio`.
"""

from alphagate.agent.book import BookRead, HeldPosition, read_book
from alphagate.agent.candidates import (
    MENU_LIMIT,
    build_candidates,
    spreads_by_delta,
    summarise_menu,
    vertical_credit_spreads,
)
from alphagate.agent.cycle import CycleRecord, cycle_id_for, run_cycle, run_exit_cycle
from alphagate.agent.earnings import (
    ETF_UNDERLYINGS,
    EarningsCalendar,
    NoEarningsCalendar,
    StaticEarningsCalendar,
    earnings_within,
)
from alphagate.agent.exits import (
    DEFAULT_EXIT_POLICY,
    ExitDecision,
    ExitPolicy,
    ExitRule,
    evaluate_exit,
)
from alphagate.agent.iv_store import IvHistoryStore
from alphagate.agent.levels import LevelRead, read_confluence, read_levels
from alphagate.agent.model import Candidate, Choice, MarketRead, ModelCall, Setup, Stage
from alphagate.agent.option_book import (
    MEASURABLE_FEATURES,
    OPTION_BOOK_SCHEMA_VERSION,
    EntryRule,
    OptionBook,
    OptionRule,
    SealedOptionRun,
    UnusableOptionBook,
    load_option_book,
    measurable_read,
)
from alphagate.agent.perceive import Perception, perceive
from alphagate.agent.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from alphagate.agent.proposer import (
    DecliningProposer,
    DeterministicProposer,
    Proposal,
    Proposer,
    RecordedProposer,
)
from alphagate.agent.runner import CycleInputs, SessionResult, run_session
from alphagate.agent.schedule import CycleKind, Slot, next_slot, session_slots
from alphagate.agent.screen import BookScreen, DefaultScreen, Screen
from alphagate.agent.sizing import size_for
from alphagate.agent.trend import TrendRead, read_trend
from alphagate.agent.watchlist import (
    COMPETITION_EARNINGS,
    WATCHLIST,
    Underlying,
    tradeable_today,
)

__all__ = [
    "COMPETITION_EARNINGS",
    "DEFAULT_EXIT_POLICY",
    "ETF_UNDERLYINGS",
    "MEASURABLE_FEATURES",
    "MENU_LIMIT",
    "OPTION_BOOK_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "WATCHLIST",
    "BookRead",
    "BookScreen",
    "Candidate",
    "Choice",
    "CycleInputs",
    "CycleKind",
    "CycleRecord",
    "DecliningProposer",
    "DefaultScreen",
    "DeterministicProposer",
    "EarningsCalendar",
    "EntryRule",
    "ExitDecision",
    "ExitPolicy",
    "ExitRule",
    "HeldPosition",
    "IvHistoryStore",
    "LevelRead",
    "MarketRead",
    "ModelCall",
    "NoEarningsCalendar",
    "OptionBook",
    "OptionRule",
    "Perception",
    "Proposal",
    "Proposer",
    "RecordedProposer",
    "Screen",
    "SealedOptionRun",
    "SessionResult",
    "Setup",
    "Slot",
    "Stage",
    "StaticEarningsCalendar",
    "TrendRead",
    "Underlying",
    "UnusableOptionBook",
    "build_candidates",
    "build_user_message",
    "cycle_id_for",
    "earnings_within",
    "evaluate_exit",
    "load_option_book",
    "measurable_read",
    "next_slot",
    "perceive",
    "read_book",
    "read_confluence",
    "read_levels",
    "read_trend",
    "run_cycle",
    "run_exit_cycle",
    "run_session",
    "session_slots",
    "size_for",
    "spreads_by_delta",
    "summarise_menu",
    "tradeable_today",
    "vertical_credit_spreads",
]
