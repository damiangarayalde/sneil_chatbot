from __future__ import annotations

from functools import lru_cache

from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.llm_provider import is_mock_mode, get_llm, MockChain
from app.core.llm_client import invoke_chain_safe

from .models import HandlerOutput


@lru_cache(maxsize=64)
def get_route_chain(route_id: str):
    """Build route handler chain lazily and cache it.

    When LLM_MOCK=true the chain is a MockChain that reads from
    tests/fixtures/mock_llm/HandlerOutput.json — no API key required.
    """
    if is_mock_mode():
        return MockChain(HandlerOutput)
    prompt_template, _route_cfg = make_chat_prompt_for_route(route_id)
    llm = get_llm(model="gpt-4o-mini", temperature=0)
    return prompt_template | llm.with_structured_output(HandlerOutput)


def _handler_fallback() -> HandlerOutput:
    """Return a safe fallback when handler LLM fails."""
    return HandlerOutput(
        is_topic_switch=False,
        answer="I apologize, I'm experiencing a temporary issue processing your request. Please try again in a moment.",
        increment_solve_attempts=False,
    )


def get_route_chain_safe_invoke(route_id: str, inputs: dict) -> HandlerOutput:
    """Invoke route handler chain with retry, timeout, and fallback.

    On any error (timeout, rate limit, API error, etc.), returns a fallback
    HandlerOutput with a generic error message, allowing the graph to handle
    the failure gracefully without crashing.

    Args:
        route_id: The route identifier (e.g., "TPMS", "AA")
        inputs: Dict with prompt variables (user_text, history, context, etc.)

    Returns:
        HandlerOutput (from chain or fallback)
    """
    chain = get_route_chain(route_id)
    return invoke_chain_safe(
        chain,
        inputs,
        fallback_fn=_handler_fallback,
    )
