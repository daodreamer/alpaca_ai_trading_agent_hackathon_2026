"""The ONLY layer permitted to call an LLM.

Enforced by ``tests/test_boundaries.py``: no module outside this package may
import an LLM SDK. A model call inside the evaluator would mean the evaluator is
no longer an independent judge of the model's own proposals.
"""

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
    "AnthropicProposer",
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
