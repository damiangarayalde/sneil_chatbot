from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field

# How much history to include in the prompt (keep small for cost)
CLASSIFIER_HISTORY_MAX_MESSAGES = 10
CLASSIFIER_HISTORY_MAX_CHARS = 2500


class ChatState(TypedDict, total=False):
    """
    Shared TypedDict describing the shape of the chat/graph state.

    `messages` uses LangGraph's `add_messages` reducer so nodes can return
    `{ "messages": [msg] }` and have it appended (not overwritten).
    """

    # Conversation
    messages: Annotated[List[BaseMessage], add_messages]

    # Routing
    confidence: Annotated[Optional[float], Field(ge=0, le=1)]
    locked_route: Optional[str]          # currently selected route (if any)
    # last classifier guess (even if not locked yet)
    estimated_route: Optional[str]

    # Escalation flag (handoff to human)
    escalated_to_human: bool

    # Attempts
    routing_attempts: int                # classifier/disambiguation attempts
    solve_attempts: int                  # answer attempts inside locked route
    max_solve_attempts: Optional[int]    # per-route cap stored at lock time

    # RAG output (list of docs as dicts)
    retrieved: Optional[List[Dict[str, Any]]]


# --------------------------------------------------------------------------------------
# Message helpers

def get_last_msg(messages: list[BaseMessage]) -> str:
    """Return the content of the last message (empty string if missing)."""
    if not messages:
        return ""
    last = messages[-1]
    return getattr(last, "content", "") or ""


def get_history_and_last_msg(messages: list[BaseMessage]) -> tuple[list[BaseMessage], str]:
    """Split the full messages list into (history, last_user_text).

    - `history`: all messages except the last
    - `last_msg`: content of the last message (empty string if missing)

    This keeps a single, shared convention across classifier + routes.
    """
    if not messages:
        return [], ""
    last = messages[-1]
    last_msg = getattr(last, "content", "") or ""
    history = list(messages[:-1])
    return history, last_msg


# --------------------------------------------------------------------------------------
# State update helpers (quick-win: keep mutations consistent across nodes)

def reset_routing_state() -> Dict[str, Any]:
    """Clear routing-related fields so the classifier can re-route cleanly."""
    return {
        "locked_route": None,
        "confidence": 0.0,
        "estimated_route": None,
        "routing_attempts": 0,
    }


def reset_solve_state() -> Dict[str, Any]:
    """Clear solve-related fields for the current route."""
    return {
        "solve_attempts": 0,
        "max_solve_attempts": None,
        "retrieved": None,
    }


def lock_route(route_id: str, *, confidence: float, max_solve_attempts: Optional[int]) -> Dict[str, Any]:
    """Lock a route and initialize solve counters."""
    return {
        "locked_route": route_id,
        "estimated_route": route_id,
        "confidence": float(confidence),
        "routing_attempts": 0,
        "solve_attempts": 0,
        "max_solve_attempts": max_solve_attempts,
        "retrieved": None,
    }


def clear_lock() -> Dict[str, Any]:
    """Clear ONLY the lock (used when a handler detects a topic switch)."""
    return {
        "locked_route": None,
        "confidence": 0.0,
    }
