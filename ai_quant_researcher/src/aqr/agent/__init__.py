"""The ONLY layer permitted to call an LLM.

Enforced by ``tests/test_boundaries.py``: no module outside this package may
import an LLM SDK. A model call inside the evaluator would mean the evaluator is
no longer an independent judge of the model's own proposals.

Two searches live here and they share nothing but the plumbing. ``prompts.py`` /
``proposer.py`` / ``research.py`` are the equity side; ``option_prompt.py`` /
``option_proposer.py`` / ``option_research.py`` are the option side. The two
vocabularies are unmixed on purpose (CLAUDE.md §2, specs/10 D5): an equity
prompt talks about stops, targets and holding periods, and every one of those
words is a lie about a structure held to expiry.
"""

from aqr.agent.option_prompt import (
    OPTION_PROPOSAL_SCHEMA,
    OPTION_SYSTEM_PROMPT,
    build_option_user_prompt,
    option_feature_catalogue,
)
from aqr.agent.option_proposer import (
    AnthropicOptionProposer,
    DeepSeekOptionProposer,
    OpenAICompatOptionProposer,
    OptionProposer,
    TemplateOptionProposer,
    build_option_spec,
    check_option_proposal,
    option_spec_to_proposal_fields,
)
from aqr.agent.option_research import (
    OPTION_SEARCH_BUDGET,
    OptionResearchConfig,
    OptionResearchLoop,
    OptionResearchStep,
)
from aqr.agent.prompts import PROPOSAL_SCHEMA, SYSTEM_PROMPT, build_user_prompt, feature_catalogue
from aqr.agent.proposer import (
    AnthropicProposer,
    DeepSeekProposer,
    HeuristicProposer,
    OpenAICompatProposer,
    Proposal,
    Proposer,
    build_spec,
    check_proposal,
    spec_to_proposal_fields,
)
from aqr.agent.research import ResearchConfig, ResearchLoop, ResearchStep

__all__ = [
    "OPTION_PROPOSAL_SCHEMA",
    "OPTION_SEARCH_BUDGET",
    "OPTION_SYSTEM_PROMPT",
    "AnthropicOptionProposer",
    "AnthropicProposer",
    "DeepSeekOptionProposer",
    "OpenAICompatOptionProposer",
    "OptionProposer",
    "OptionResearchConfig",
    "OptionResearchLoop",
    "OptionResearchStep",
    "TemplateOptionProposer",
    "build_option_spec",
    "build_option_user_prompt",
    "check_option_proposal",
    "option_feature_catalogue",
    "option_spec_to_proposal_fields",
    "DeepSeekProposer",
    "OpenAICompatProposer",
    "HeuristicProposer",
    "PROPOSAL_SCHEMA",
    "Proposal",
    "Proposer",
    "ResearchConfig",
    "ResearchLoop",
    "ResearchStep",
    "SYSTEM_PROMPT",
    "build_spec",
    "check_proposal",
    "build_user_prompt",
    "feature_catalogue",
    "spec_to_proposal_fields",
]
