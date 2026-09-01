"""The option proposer and its prompt — specs/10-options-research.md D4, D5, D6.

The proposer is a containment boundary, not a convenience, so what is asserted
here is mostly what it *refuses*:

* a structure outside D4's whitelist does not compile,
* a width that is not below its anchor does not compile,
* an expiry outside the three the vendor carries does not compile,
* an exit rule has nowhere to go, because there is no field for one (D1),
* and every rejection carries a message naming the offending value, because that
  message is fed straight back to the model as a repair turn.

The offline half is asserted to be usable with no API key and no network, which
is the property that makes the option research loop runnable in CI at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aqr.agent.option_prompt import (
    ALLOWED_DTE_TARGETS,
    OPTION_PROPOSAL_SCHEMA,
    OPTION_SYSTEM_PROMPT,
    STRUCTURE_CATALOGUE,
    build_option_user_prompt,
    option_feature_catalogue,
    structure_catalogue,
)
from aqr.agent.option_proposer import (
    DeepSeekOptionProposer,
    TemplateOptionProposer,
    build_option_spec,
    check_option_proposal,
    option_spec_to_proposal_fields,
    unreachable_thresholds,
)
from aqr.agent.proposer import Proposal
from aqr.features.registry import REGISTRY
from aqr.options.features import OPTION_FEATURES
from aqr.options.spec import dumps_option_spec, loads_option_spec

VALID: dict[str, Any] = {
    "name": "iv_rank_put_credit_spread_v1",
    "hypothesis": "Index puts carry a variance risk premium paid by hedgers.",
    "entry": "iv_rank() > 40 and close > sma(200)",
    "structure_type": "put_credit_spread",
    "dte_target": 28,
    "anchor_delta": 0.16,
    "width_delta": 0.06,
    "call_anchor_delta": 0.0,
    "call_width_delta": 0.0,
    "min_sessions_between_entries": 5,
    "expected_cycles_per_year": 12,
}


def fields(**overrides: Any) -> dict[str, Any]:
    return {**VALID, **overrides}


# --------------------------------------------------------------------------- #
# The prompt states what the data can answer, and not more
# --------------------------------------------------------------------------- #


def test_the_catalogue_is_generated_from_both_real_vocabularies() -> None:
    """D6: an entry expression parses against the option table *and* the bar
    registry, so a prompt that enumerated one of them would have the model
    proposing rules that do not compile, or never reaching for half the
    vocabulary it is allowed."""
    catalogue = option_feature_catalogue()
    for name in OPTION_FEATURES:
        assert f"{name}(" in catalogue
    for name in REGISTRY:
        assert name in catalogue


def test_the_structure_catalogue_is_exactly_d4s_whitelist() -> None:
    rendered = structure_catalogue()
    for kind, _ in STRUCTURE_CATALOGUE:
        assert kind in rendered
    # The kinds D4 names and refuses. Absent from the enum, absent from the
    # prompt, and unrepresentable in the spec -- three places, on purpose.
    for absent in ("covered_call", "cash_secured_put", "naked_put", "custom"):
        assert absent not in rendered
        assert absent not in OPTION_PROPOSAL_SCHEMA["properties"]["structure_type"]["enum"]


def test_the_schema_has_no_exit_field_at_all() -> None:
    """D1/D5: a field whose semantics are fabricated is worse than a missing one,
    because the model will fill it in and every number downstream will quietly
    describe a strategy nobody can trade."""
    properties = set(OPTION_PROPOSAL_SCHEMA["properties"])
    forbidden = {
        "stop_loss",
        "stop_loss_atr_multiple",
        "take_profit",
        "take_profit_r_multiple",
        "max_holding_bars",
        "signal_exit",
        "exit",
        "roll",
        "manage_at",
    }
    assert not properties & forbidden
    assert OPTION_PROPOSAL_SCHEMA["additionalProperties"] is False


def test_the_system_prompt_states_the_four_facts_a_model_gets_wrong() -> None:
    text = OPTION_SYSTEM_PROMPT.lower()
    assert "there is no exit" in text
    assert "delta, not by points" in text
    assert "three expiries" in text
    assert "one underlying" in text
    assert "independent cycles" in text


def test_the_user_prompt_shows_the_remaining_budget() -> None:
    """D8's cap is only a deterrent to the model if the model can see it: told
    nothing, a proposer keeps offering small variations of its last idea."""
    prompt = build_option_user_prompt(underlying="SPY", memory=[], budget=(17, 20))
    assert "hypothesis 18 of 20" in prompt


def test_memory_shows_cycles_beside_trades() -> None:
    """A prompt that showed the trade count alone would invite the model to
    propose something that trades more, when the binding constraint is how many
    of those trades were independent (D8)."""
    prompt = build_option_user_prompt(
        underlying="SPY",
        memory=[
            {
                "name": "put_credit_spread_v1",
                "verdict": "REVIEW",
                "hypothesis": "the variance risk premium",
                "oos_trades": 89,
                "oos_cycles": 37,
            }
        ],
    )
    assert "89 trades" in prompt
    assert "37 independent cycles" in prompt


# --------------------------------------------------------------------------- #
# What a proposal must survive
# --------------------------------------------------------------------------- #


def test_a_well_formed_proposal_has_no_problems() -> None:
    assert check_option_proposal(VALID) == []


def test_an_unknown_structure_is_rejected_by_name() -> None:
    problems = check_option_proposal(fields(structure_type="naked_put"))
    assert any("naked_put" in p and "structure_type" in p for p in problems)


@pytest.mark.parametrize("dte", [0, 1, 7, 30, 365])
def test_only_the_three_listed_expiry_targets_are_accepted(dte: int) -> None:
    """D0: the vendor samples three rolling targets. Anything else names a
    contract this cache cannot price, and a rule that cannot be priced is a
    wasted iteration out of twenty."""
    assert dte not in ALLOWED_DTE_TARGETS
    problems = check_option_proposal(fields(dte_target=dte))
    assert any("dte_target" in p for p in problems)


def test_a_wing_at_or_above_its_anchor_is_rejected_with_both_numbers() -> None:
    problems = check_option_proposal(fields(anchor_delta=0.16, width_delta=0.20))
    assert any("0.2" in p and "0.16" in p for p in problems)


def test_a_spread_without_a_width_is_rejected() -> None:
    problems = check_option_proposal(fields(width_delta=0.0))
    assert any("width_delta" in p and "maximum loss" in p for p in problems)


def test_a_one_legged_structure_with_a_width_is_rejected() -> None:
    problems = check_option_proposal(fields(structure_type="long_put", width_delta=0.06))
    assert any("one leg" in p for p in problems)


def test_call_side_fields_are_iron_condor_only() -> None:
    problems = check_option_proposal(fields(call_anchor_delta=0.16))
    assert any("iron-condor only" in p for p in problems)


def test_an_iron_condor_may_set_both_sides() -> None:
    assert (
        check_option_proposal(
            fields(
                structure_type="iron_condor",
                call_anchor_delta=0.20,
                call_width_delta=0.08,
            )
        )
        == []
    )


def test_an_unknown_feature_is_rejected_against_the_real_grammar() -> None:
    problems = check_option_proposal(fields(entry="dealer_gamma() > 0"))
    assert any("entry does not parse" in p for p in problems)


def test_an_entry_that_is_not_a_condition_is_rejected_with_a_suggestion() -> None:
    problems = check_option_proposal(fields(entry="iv_rank()"))
    assert any("Compare it to something" in p for p in problems)


def test_an_entry_that_smuggles_in_an_exit_is_named_as_such() -> None:
    """The model's most likely mistake, and ``unknown feature 'stop_loss'`` would
    read as a typo rather than as "that concept does not exist here" (D1)."""
    problems = check_option_proposal(fields(entry="stop_loss(2) > 0"))
    assert any("no exit" in p for p in problems)


def test_every_problem_is_reported_in_one_turn() -> None:
    """Three round trips out of a twenty-hypothesis budget is 15% of the search
    spent on form-filling."""
    problems = check_option_proposal(
        fields(structure_type="naked_put", dte_target=7, min_sessions_between_entries=0)
    )
    assert len(problems) >= 3


# --------------------------------------------------------------------------- #
# Proposal -> spec
# --------------------------------------------------------------------------- #


def test_the_run_configuration_owns_the_risk_budget_not_the_proposal() -> None:
    """D8a measured the same rule producing 21 cycles at 1% of equity and 57 at
    2%. A model that could set this could buy its own significance."""
    spec = build_option_spec(
        Proposal(fields=fields(risk_per_trade=0.09), source="test"),
        "SPY",
        risk_per_trade=0.02,
        max_concurrent=3,
    )
    assert spec.sizing.risk_per_trade == 0.02
    assert spec.sizing.max_concurrent == 3


def test_the_underlying_comes_from_the_run_not_the_proposal() -> None:
    spec = build_option_spec(Proposal(fields=fields(underlying="TSLA"), source="test"), "SPY")
    assert spec.underlying == "SPY"


def test_a_zero_width_becomes_no_width_rather_than_an_invalid_one() -> None:
    """A JSON schema cannot express "omit this field", so the proposer writes 0
    for an absent width; ``StructureSpec`` would refuse 0 with a message about
    positive numbers."""
    spec = build_option_spec(
        Proposal(fields=fields(structure_type="long_put", width_delta=0.0), source="test"), "SPY"
    )
    assert spec.structure.width_delta is None


def test_a_structure_outside_the_whitelist_does_not_construct() -> None:
    with pytest.raises(ValueError):
        build_option_spec(Proposal(fields=fields(structure_type="naked_put"), source="test"))


def test_proposal_fields_round_trip_through_a_spec() -> None:
    """Research memory and the mutation loop are built from these. A rule that
    could not be described back to the model would be invisible to the search
    that produced it."""
    spec = build_option_spec(Proposal(fields=VALID, source="test"), "SPY")
    rendered = option_spec_to_proposal_fields(spec)
    assert check_option_proposal(rendered) == []
    again = build_option_spec(Proposal(fields=rendered, source="test"), "SPY")
    assert again.fingerprint() == spec.fingerprint()


def test_a_spec_round_trips_through_yaml() -> None:
    """The registry stores a spec as text, so this is what makes a recorded
    option rule readable back into the type that produced it."""
    spec = build_option_spec(Proposal(fields=VALID, source="test"), "SPY")
    restored = loads_option_spec(dumps_option_spec(spec))
    assert restored.fingerprint() == spec.fingerprint()
    assert restored.name == spec.name
    assert restored.hypothesis == spec.hypothesis
    assert restored.structure == spec.structure


def test_an_option_yaml_is_not_loadable_as_an_equity_spec() -> None:
    """The two formats are deliberately not interchangeable. A shared top-level
    key would let ``aqr backtest`` read an option rule as a spec with no entry
    condition rather than refusing it."""
    from aqr.dsl.loader import loads as loads_equity

    spec = build_option_spec(Proposal(fields=VALID, source="test"), "SPY")
    with pytest.raises(ValueError):
        loads_equity(dumps_option_spec(spec))


# --------------------------------------------------------------------------- #
# The offline proposer
# --------------------------------------------------------------------------- #


def test_the_offline_proposer_produces_valid_varied_rules_with_no_api_key() -> None:
    """O4.1's acceptance, and the property that makes the loop runnable in CI."""
    proposer = TemplateOptionProposer()
    seen: set[str] = set()
    kinds: set[str] = set()
    for _ in range(20):
        proposal = proposer.propose(underlying="SPY", memory=[])
        assert check_option_proposal(proposal.fields) == [], proposal.fields
        spec = build_option_spec(proposal, "SPY")
        assert spec.fingerprint() not in seen, f"{spec.name} duplicates an earlier rule"
        seen.add(spec.fingerprint())
        kinds.add(spec.structure.type)
    assert len(kinds) >= 4


def test_the_offline_proposer_is_deterministic() -> None:
    first = [TemplateOptionProposer().propose(underlying="SPY", memory=[]) for _ in range(1)]
    left = TemplateOptionProposer()
    right = TemplateOptionProposer()
    for _ in range(16):
        assert left.propose(underlying="SPY", memory=[]).fields == right.propose(
            underlying="SPY", memory=[]
        ).fields
    assert first  # the library is walked in order, so the first is always the same


def test_the_offline_library_covers_the_hand_written_campaign() -> None:
    """PLAN-OPTIONS.md records results for six hand-written structures. An
    offline run has to be able to reproduce that campaign, or the model has no
    baseline to be measured against."""
    proposer = TemplateOptionProposer()
    names = {proposer.propose(underlying="SPY", memory=[]).fields["name"] for _ in range(12)}
    assert {
        "put_credit_spread_unconditional",
        "put_credit_spread_in_uptrend",
        "put_credit_spread_high_iv_rank",
        "iron_condor_neutral",
        "long_put_crash_hedge",
        "call_credit_spread_unconditional",
    } <= names


def test_a_mutation_changes_exactly_one_knob() -> None:
    """A mutation that alters three parameters and improves the result has taught
    nobody which of the three mattered."""
    proposer = TemplateOptionProposer()
    parent = proposer.propose(underlying="SPY", memory=[]).fields
    child = proposer.propose(underlying="SPY", memory=[], parent=parent).fields
    changed = {
        key
        for key in ("dte_target", "anchor_delta", "min_sessions_between_entries")
        if parent[key] != child[key]
    }
    assert len(changed) == 1
    # The wing moves with the anchor rather than independently: a width that
    # crossed its anchor would not be a variant of the rule, it would be an
    # invalid structure.
    if "anchor_delta" in changed and child["width_delta"]:
        assert child["width_delta"] < child["anchor_delta"]
    assert check_option_proposal(child) == []


# --------------------------------------------------------------------------- #
# The repair loop, without a network
# --------------------------------------------------------------------------- #


class FakeChat:
    """The OpenAI client surface the proposer actually uses, and nothing more."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.chat = self

    @property
    def completions(self) -> FakeChat:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(list(kwargs["messages"]))
        text = self.replies.pop(0)
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("M", (), {"content": text})()})()]},
        )()


def test_a_rejected_proposal_is_sent_back_with_the_specific_errors() -> None:
    bad = json.dumps(fields(structure_type="naked_put"))
    good = json.dumps(VALID)
    client = FakeChat([bad, good])
    proposer = DeepSeekOptionProposer(client=client)

    proposal = proposer.propose(underlying="SPY", memory=[])

    assert proposal.fields["structure_type"] == "put_credit_spread"
    assert len(client.calls) == 2
    repair = client.calls[1][-1]["content"]
    assert "naked_put" in repair
    assert "rejected" in repair


def test_the_repair_loop_is_bounded_and_the_failure_names_the_reason() -> None:
    """One repair turn, not five: a model that cannot fill in the form after
    being told exactly what it got wrong is not about to produce a good
    hypothesis on the third try, and the next iteration is a cheaper place to
    spend those tokens."""
    bad = json.dumps(fields(structure_type="naked_put"))
    client = FakeChat([bad, bad])
    proposer = DeepSeekOptionProposer(client=client)

    with pytest.raises(ValueError, match="naked_put"):
        proposer.propose(underlying="SPY", memory=[])
    assert len(client.calls) == 2


def test_a_repair_that_is_still_unusable_raises_rather_than_returning_it() -> None:
    """The loop treats a raised repair as "no repair" and evaluates the original,
    which keeps the real rejection reason attached to the real attempt."""
    client = FakeChat([json.dumps(fields(dte_target=7))])
    proposer = DeepSeekOptionProposer(client=client)
    with pytest.raises(ValueError, match="still unusable"):
        proposer.repair(
            proposal=Proposal(fields=VALID, source="test"), problems=["opened nothing"]
        )


def test_the_dead_rule_instruction_carries_the_census() -> None:
    """A rule can be perfectly satisfiable and still trade nothing, because the
    account could not afford the structure (D8a). Telling the model to "loosen
    the condition" for an affordability skip would have it weaken a hypothesis
    to fix an account."""
    client = FakeChat([json.dumps(fields(name="repaired_v2"))])
    proposer = DeepSeekOptionProposer(client=client)
    proposer.repair(
        proposal=Proposal(fields=VALID, source="test"),
        problems=["the rule opened no positions: affordability=578"],
    )
    instruction = client.calls[0][-1]["content"]
    assert "affordability=578" in instruction
    assert "put_credit_spread" in instruction


def test_the_dead_rule_instruction_quotes_the_ranges_of_the_features_it_named() -> None:
    """The first version of this instruction quoted ``iv_rank()``'s range no
    matter what the rule said -- and in the campaign that motivated the fix,
    ``iv_rank()`` was the one feature that was never the problem. The model read
    a range for something it had not misused and repeated its unit error on the
    features it had."""
    client = FakeChat([json.dumps(fields(name="repaired_v2"))])
    proposer = DeepSeekOptionProposer(client=client)
    proposer.repair(
        proposal=Proposal(
            fields=fields(entry="term_slope() > 0.01 and close > sma(200)"),
            source="test",
        ),
        problems=["the rule opened no positions"],
        span=_span,
    )
    instruction = client.calls[0][-1]["content"]
    assert "term_slope()" in instruction
    assert "sma(200)" in instruction
    assert "iv_rank" not in instruction


# --------------------------------------------------------------------------- #
# Thresholds nothing can satisfy
# --------------------------------------------------------------------------- #
#
# The measured failure this guards, in full: a twenty-hypothesis campaign lost
# eight slots to rules that opened nothing, and seven of those eight were a
# units error. The volatility features here are decimal fractions; the model
# wrote percentage points, because ``iv_rank()`` is the only feature on a 0..100
# scale and it was the only one that documented itself. Each of those is a good
# hypothesis in the wrong unit, and each cost a slot of a fixed budget.

_SPANS: dict[str, tuple[float, float]] = {
    "term_slope()": (-1.019, 0.0519),
    "iv_hv_spread()": (-0.4783, 0.1683),
    "iv_current()": (0.0923, 0.7663),
    "skew_25d()": (-0.8855, 0.3019),
    "iv_rank()": (0.0, 100.0),
    "realized_vol(20)": (0.0473, 0.8619),
    "sma(200)": (100.0, 560.0),
    "close": (100.0, 570.0),
}


def _span(key: Any) -> tuple[float, float] | None:
    return _SPANS.get(str(key))


@pytest.mark.parametrize(
    "entry",
    [
        "term_slope() > 5",
        "term_slope() > 5 and iv_rank() > 30 and close > sma(200)",
        "iv_hv_spread() > 5",
        "iv_hv_spread() > 2 and close > sma(200)",
        "rvol(5) > 1.5 and roc(5) > 0 and iv_current() > 25",
        "close > highest(20) and realized_vol(20) > 15 and iv_rank() > 5",
        "skew_25d() > 2.0",
        "5 < term_slope()",  # the mirrored form is the same claim
    ],
)
def test_a_threshold_no_session_can_satisfy_is_caught(entry: str) -> None:
    problems = unreachable_thresholds(entry, _span)
    assert problems, entry
    assert "never" in problems[0]
    assert "DECIMAL" in problems[0]


@pytest.mark.parametrize(
    "entry",
    [
        "term_slope() > 0",
        "iv_rank() > 50",
        "iv_hv_spread() > 0.05",
        "skew_25d() > 0.12",  # rare, and rare is not the same as impossible
        "close > sma(200)",
        "term_slope() > 0 and iv_rank() < 50",
    ],
)
def test_a_reachable_threshold_is_left_alone(entry: str) -> None:
    """Only "none" is flagged, never "few". A rule firing on 2% of sessions is a
    small sample and the cycle gate already says so; flagging it here would put
    the proposer in the business of preferring rules that trade more."""
    assert unreachable_thresholds(entry, _span) == []


def test_a_feature_with_no_measured_span_is_not_guessed_at() -> None:
    assert unreachable_thresholds("adx(14) > 9000", _span) == []


def test_an_unparseable_entry_is_left_to_the_parser() -> None:
    """The parser's "unknown feature, did you mean..." is the useful message;
    a second complaint in the language of ranges would bury it."""
    assert unreachable_thresholds("dealer_gamma() > 5", _span) == []


def test_the_range_check_runs_inside_the_proposers_own_retry_loop() -> None:
    """This is the whole point: the model gets the numbers back and fixes it
    before a spec exists, so an unreachable threshold costs a retry rather than
    a slot of the search budget."""
    bad = json.dumps(fields(entry="term_slope() > 5"))
    good = json.dumps(fields(entry="term_slope() > 0.01"))
    client = FakeChat([bad, good])
    proposer = DeepSeekOptionProposer(client=client)

    proposal = proposer.propose(underlying="SPY", memory=[], span=_span)

    assert proposal.fields["entry"] == "term_slope() > 0.01"
    repair = client.calls[1][-1]["content"]
    assert "term_slope() is never above 5" in repair
    assert "-1.019 .. 0.0519" in repair


def test_without_a_span_the_range_check_simply_does_not_run() -> None:
    """A caller with no market -- a schema test, a hand-written spec -- can still
    check everything else."""
    assert check_option_proposal(fields(entry="term_slope() > 5")) == []
    assert check_option_proposal(fields(entry="term_slope() > 5"), span=_span) != []
