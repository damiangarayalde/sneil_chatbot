from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.graph.msg_heuristics_no_llm import (
    direct_route_from_keywords,
    route_disambiguation_question,
    wrap_with_greeting,
)
from app.core.graph.state import (
    ChatState,
    get_history_and_last_msg,
    lock_route,
)
from app.core.utils import is_valid_route

from .chain import (
    classifier_chain,
    max_solve_attempts_for_route,
    route_lock_threshold,
)
from .models import ALLOWED_ROUTES


def node__classify_user_intent(state: ChatState) -> ChatState:
    """Hub / classifier node.

    Rules (in order):
    1) If already locked -> pass-through
    2) If cheap keyword routing works -> lock route (no LLM)
    3) Else call LLM:
        - low confidence -> greet + clarifier question (routing_attempts += 1)
        - high confidence -> lock route
    """
    locked = state.get("locked_route")
    if is_valid_route(locked):
        return {}

    prior_messages, last_message = get_history_and_last_msg(
        state.get("messages") or [])
    last_message = last_message or ""
    routing_attempts = int(state.get("routing_attempts") or 0)

    # cheap direct routing (keywords, route mentions, etc.)
    direct = direct_route_from_keywords(last_message, ALLOWED_ROUTES)
    if direct:
        return lock_route(
            direct,
            confidence=1.0,
            max_solve_attempts=max_solve_attempts_for_route(direct),
        )

    # LLM classifier
    meta_text = f"routing_attempts={routing_attempts}\n"
    result = classifier_chain().invoke(
        {
            "user_text": last_message,
            "history": prior_messages,
            "context": "",
            "meta": meta_text,
        }
    )

    has_clarifier = bool((result.clarifying_question or "").strip())
    if (not has_clarifier) and float(result.confidence) >= route_lock_threshold():
        route_id = result.estimated_route
        return lock_route(
            route_id,
            confidence=float(result.confidence),
            max_solve_attempts=max_solve_attempts_for_route(route_id),
        )

    # low confidence => ask a routing question (counts as routing_attempt)
    question = (result.clarifying_question or "").strip() or route_disambiguation_question(
        result.estimated_route
    )
    return {
        "confidence": float(result.confidence),
        "estimated_route": result.estimated_route,
        "routing_attempts": routing_attempts + 1,
        "messages": [AIMessage(content=wrap_with_greeting(question))],
    }
