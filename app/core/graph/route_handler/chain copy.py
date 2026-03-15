from __future__ import annotations

from functools import lru_cache

from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import init_llm
from app.core.tools.catalog_tool_llm import create_catalog_lookup_tool
from .models import HandlerOutput


@lru_cache(maxsize=64)
def get_route_chain(route_id: str):
    """Build route handler chain lazily and cache it."""

    llm = init_llm(model="gpt-4o-mini", temperature=0)

    catalog_tool = create_catalog_lookup_tool()

    llm_with_tools = llm.bind_tools([catalog_tool])

    prompt_template, _route_cfg = make_chat_prompt_for_route(route_id)

    return prompt_template | llm_with_tools.with_structured_output(HandlerOutput)
