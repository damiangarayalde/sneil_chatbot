from __future__ import annotations
from app.core.graph.state import ChatState

from langchain_core.messages import AIMessage
from app.core.graph.msg_heuristics_no_llm import (
    default_clarifier,
    escalation_message,
    route_disambiguation_question,
    wrap_with_greeting,
)
from app.core.graph.state import (
    ChatState,
    reset_routing_state,
    reset_solve_state,
)
from app.core.utils import is_valid_route
from app.core.graph.classifier.models import ALLOWED_ROUTES


def node__clarify(state: ChatState) -> ChatState:
    """
    Clarify when user msg is too short / low info.
    Works for both:
      - before routing is locked (generic clarifier)
      - after routing is locked (route-specific disambiguation question)
    """
    locked = state.get("locked_route")
    if is_valid_route(locked):
        q = route_disambiguation_question(locked)
        text = wrap_with_greeting(q)
    else:
        text = default_clarifier()

    return {
        "messages": [AIMessage(content=text)],
        "retrieved": None,
    }


def node__handoff(state: ChatState) -> ChatState:
    """
    Handoff when:
      - user asks for human
      - routing attempts exceeded
      - solve attempts exceeded for locked route
    """
    msg = (
        "Disculpá — para no hacerte perder tiempo, mejor lo pasamos con una persona.\n\n"
        f"{escalation_message()}"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "escalated_to_human": True,
        **reset_routing_state(),
        **reset_solve_state(),
    }


def end_turn_node_name() -> str:
    """Single place to define the end-of-turn node name."""
    return "end_of_turn"


def node__end_of_turn(state: ChatState) -> ChatState:
    """Finalize a single graph invocation ("turn").

    Keeping for future use and flexibility
    """

    updates: dict = {}

    return updates
