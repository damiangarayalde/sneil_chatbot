from __future__ import annotations

from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import Field


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
    routing_attempts: int

    # Per-route attempts / escalation
    attempts: Dict[str, int]

    # RAG output (its a list of docs)
    retrieved: Optional[List[Dict[str, Any]]]


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
