from __future__ import annotations

from functools import lru_cache

from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.llm_provider import is_mock_mode, get_llm, MockChain

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
