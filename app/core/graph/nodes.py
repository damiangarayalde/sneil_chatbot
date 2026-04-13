from __future__ import annotations
from app.core.graph.state import ChatState, get_last_msg

from langchain_core.messages import AIMessage, BaseMessage
from app.core.graph.msg_heuristics_no_llm import (
    default_clarifier,
    route_disambiguation_question,
    wrap_with_greeting,
    asked_for_human,
)
from app.core.graph.state import (
    ChatState,
    reset_routing_state,
    reset_solve_state,
)
from app.core.utils import is_valid_route, load_cfg
from app.core.graph.route_classifier.models import ALLOWED_ROUTES
from app.core.logging_config import get_logger

_logger = get_logger("sneil.handoff")


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
      - user asks for human (explicit request)
      - routing attempts exceeded
      - solve attempts exceeded for locked route
    """
    last_msg = get_last_msg(state.get("messages") or [])
    reason = "unknown"

    # Determine why we're handing off
    if asked_for_human(last_msg):
        reason = "user_requested_human"
    elif state.get("solve_attempts"):
        reason = "solve_attempts_exceeded"
    elif state.get("routing_attempts"):
        reason = "routing_attempts_exceeded"

    _logger.info(
        "handoff initiated",
        extra={
            "reason": reason,
            "locked_route": state.get("locked_route"),
            "routing_attempts": state.get("routing_attempts"),
            "solve_attempts": state.get("solve_attempts"),
        },
    )

    # Different messages based on handoff reason
    if reason == "user_requested_human":
        msg = "De acuerdo, para hablar con uno de nuestros asesores escribinos por WhatsApp a nuestro número de soporte (humano)."
    else:
        # For solve_attempts_exceeded or routing_attempts_exceeded
        msg = "Disculpa, en este caso mejor te derivo con un asesor tecnico, por favor escribinos por WhatsApp a nuestro número de soporte (humano)."

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
    - Truncate message history to MAX_HISTORY_MESSAGES to prevent context explosion.
    - Keep the last N messages; if fewer than N exist, keep all.
    """
    updates: dict = {}

    # Load config and get max_history_messages
    cfg = load_cfg()
    max_history = cfg.get("MAX_HISTORY_MESSAGES", 20)

    # Truncate messages if needed
    messages: list[BaseMessage] = state.get("messages") or []
    if len(messages) > max_history:
        # Keep only the last N messages
        truncated_messages = messages[-max_history:]
        updates["messages"] = truncated_messages
        _logger.info(
            "message history truncated",
            extra={
                "total_messages": len(messages),
                "kept_messages": len(truncated_messages),
                "max_history": max_history,
            },
        )

    return updates
