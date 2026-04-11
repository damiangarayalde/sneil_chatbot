from __future__ import annotations

from functools import lru_cache

from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.llm_provider import is_mock_mode, get_llm, MockChain

from .models import ClassifierOutput


@lru_cache(maxsize=1)
def get_classifier_resources():
    """
    Build classifier chain lazily and cache it.
    Returns: (chain, classifier_cfg)

    When LLM_MOCK=true the chain is a MockChain that reads from
    tests/fixtures/mock_llm/ClassifierOutput.json — no API key required.
    """
    _classifier_prompt, classifier_cfg = make_chat_prompt_for_route("CLASSIFIER")
    if is_mock_mode():
        chain = MockChain(ClassifierOutput)
    else:
        llm = get_llm(model="gpt-4o-mini", temperature=0)
        chain = _classifier_prompt | llm.with_structured_output(ClassifierOutput)
    return chain, classifier_cfg


def classifier_cfg() -> dict:
    return get_classifier_resources()[1]


def classifier_chain():
    return get_classifier_resources()[0]


def max_routing_attempts_before_handoff() -> int:
    cfg = classifier_cfg()
    return int(cfg.get("max_attempts_before_handoff") or 0)


def route_lock_threshold() -> float:
    cfg = classifier_cfg()
    return float(cfg.get("route_lock_threshold") or 0.7)


@lru_cache(maxsize=64)
def max_solve_attempts_for_route(route_id: str) -> int:
    # Reads route cfg to store threshold in state when locking
    _prompt, cfg = make_chat_prompt_for_route(route_id)
    return int(cfg.get("max_attempts_before_handoff") or 0)
