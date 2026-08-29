"""The proposer, once there are two forms a hypothesis can take.

Everything upstream of here now supports a ranked, always-invested book, and
none of it matters while the model can only describe a trigger. The registry
already shows what happens when the available forms do not fit the ideas: 214 of
283 proposals were short or market-neutral and every one died, because the only
form on offer was one that gives up the drift and is then measured against it.

So the schema gains a ``mode``. Two things about how, both of which would be
easy to get wrong in a way that produces plausible rubbish rather than an error:

**The fields are mode-specific and enforced.** A portfolio proposal carrying an
``entry``, or a signal proposal carrying a ``rank_by``, has not chosen a form.
Silently ignoring the extra field would run a strategy nobody proposed, so it is
a rejection with a message the repair turn can act on.

**The universe still is not the model's to choose.** ``hold`` and
``rebalance_every`` are the model's -- they are the hypothesis -- but the symbol
set stays with the run configuration, for the same reason it always did: a model
that could choose its own universe could choose the one its idea happens to work
on.
"""

from __future__ import annotations

import pytest

from aqr.agent.prompts import PROPOSAL_SCHEMA, SYSTEM_PROMPT
from aqr.agent.proposer import Proposal, build_spec, check_proposal, spec_to_proposal_fields

# Wide enough that a hold of 10 is a legitimate request rather than something
# the clamp has to rescue -- the clamp gets its own test below.
SYMBOLS = [f"S{i:02d}" for i in range(12)]


def _portfolio_fields(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "xs_relative_strength_v1",
        "hypothesis": "names leading their peers keep leading over a month",
        "mode": "portfolio",
        "rank_by": "rs_rank(60) - rs_rank(5)",
        "hold": 10,
        "rebalance_every": 21,
    }
    base.update(over)
    return base


def _signal_fields(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "pullback_v1",
        "hypothesis": "a shallow pullback in an uptrend is profit-taking",
        "entry": "close <= ema(20) and rsi(14) > 40",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 2.0,
        "max_holding_bars": 20,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# The schema


def test_the_schema_offers_both_modes() -> None:
    mode = PROPOSAL_SCHEMA["properties"]["mode"]
    assert set(mode["enum"]) == {"signal", "portfolio"}


def test_the_schema_describes_the_portfolio_fields() -> None:
    for field in ("rank_by", "hold", "rebalance_every"):
        assert field in PROPOSAL_SCHEMA["properties"], field


def test_entry_is_no_longer_unconditionally_required() -> None:
    """A portfolio proposal has no entry. Leaving it required would make the
    schema reject the form it is being extended to allow."""
    assert "entry" not in PROPOSAL_SCHEMA.get("required", [])


def test_the_system_prompt_explains_when_to_use_each_form() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "portfolio" in lowered
    assert "rank_by" in lowered


# --------------------------------------------------------------------------
# Checking what came back


def test_a_portfolio_proposal_passes_its_own_checks() -> None:
    assert check_proposal(_portfolio_fields()) == []


def test_a_signal_proposal_still_passes() -> None:
    assert check_proposal(_signal_fields()) == []


def test_a_portfolio_proposal_without_a_ranking_is_rejected() -> None:
    problems = check_proposal(_portfolio_fields(rank_by=""))
    assert any("rank_by" in p for p in problems), problems


def test_a_portfolio_proposal_carrying_an_entry_is_rejected() -> None:
    """It has not chosen a form. Ignoring the extra field would run a strategy
    nobody proposed."""
    problems = check_proposal(_portfolio_fields(entry="close > ema(20)"))
    assert any("entry" in p for p in problems), problems


def test_a_signal_proposal_carrying_a_ranking_is_rejected() -> None:
    problems = check_proposal(_signal_fields(rank_by="roc(20)"))
    assert any("rank_by" in p or "mode" in p for p in problems), problems


def test_a_ranking_that_is_a_condition_is_rejected_with_a_usable_message() -> None:
    """The repair turn gets one shot, so the message has to say what to do."""
    problems = check_proposal(_portfolio_fields(rank_by="close > ema(20)"))
    assert any("number" in p or "condition" in p for p in problems), problems


def test_a_ranking_naming_an_unknown_feature_is_rejected() -> None:
    problems = check_proposal(_portfolio_fields(rank_by="ema_of_tomorrow(20)"))
    assert problems


# --------------------------------------------------------------------------
# Building the spec


def test_a_portfolio_proposal_becomes_a_portfolio_spec() -> None:
    spec = build_spec(Proposal(fields=_portfolio_fields(), source="test"), SYMBOLS)
    assert spec.mode == "portfolio"
    assert spec.rank_by == "rs_rank(60) - rs_rank(5)"
    assert spec.hold == 10
    assert spec.rebalance_every == 21
    assert spec.entry == ""


def test_the_universe_is_still_not_the_models_to_choose() -> None:
    """``hold`` is the hypothesis; the symbol set is the run configuration."""
    spec = build_spec(
        Proposal(fields=_portfolio_fields(universe=["ZZZ"]), source="test"), SYMBOLS
    )
    assert set(spec.universe.symbols) == set(SYMBOLS)


def test_a_hold_larger_than_the_universe_is_clamped_not_rejected() -> None:
    """A model asking for the top 20 of a 4-name universe has made a reasonable
    request against a configuration it cannot see. Clamping keeps the idea;
    rejecting spends an iteration on an accident of the run."""
    spec = build_spec(Proposal(fields=_portfolio_fields(hold=999), source="test"), SYMBOLS)
    assert spec.hold <= len(SYMBOLS)


def test_the_sleeve_is_configuration_not_proposal() -> None:
    """The 80/20 split is an architectural decision about this book. A model
    that could set it could quietly turn itself into a pure benchmark holding."""
    assert "sleeve" not in PROPOSAL_SCHEMA["properties"]
    spec = build_spec(Proposal(fields=_portfolio_fields(), source="test"), SYMBOLS)
    assert spec.sleeve.budget == pytest.approx(0.20)
    assert spec.sleeve.idle == "benchmark"


def test_a_portfolio_spec_round_trips_back_into_proposal_fields() -> None:
    """Research memory is built from past proposals. A portfolio strategy that
    could not be described back to the model would be invisible to the search
    that produced it."""
    spec = build_spec(Proposal(fields=_portfolio_fields(), source="test"), SYMBOLS)
    fields = spec_to_proposal_fields(spec)
    assert fields["mode"] == "portfolio"
    assert fields["rank_by"] == spec.rank_by
    assert check_proposal(fields) == []
