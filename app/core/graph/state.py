from __future__ import annotations

from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
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
    locked_route: Optional[str]

    # Last classifier guess (even if not locked yet)
    estimated_route: Optional[str]

    # Escalation flag (handoff to human)
    escalated_to_human: bool

    routing_attempts: int   # classifier/disambiguation attempts
    solve_attempts: int     # answer attempts inside locked route

    # # (TEMP) legacy counter — keep during migration so old code won’t break.
    # attempts: int

    # Optional per-route solve threshold to allow high-level gating later
    max_solve_attempts: Optional[int]

    # RAG output (its a list of docs)
    retrieved: Optional[List[Dict[str, Any]]]


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

    This keeps a single, shared convention across hub + handlers.
    """
    if not messages:
        return [], ""
    last = messages[-1]
    last_msg = getattr(last, "content", "") or ""
    history = list(messages[:-1])
    return history, last_msg
