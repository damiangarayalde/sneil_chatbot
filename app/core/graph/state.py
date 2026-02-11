from __future__ import annotations

from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from pydantic import Field


class ChatState(TypedDict, total=False):
    """Shared TypedDict describing the shape of the chat/graph state.

    `messages` uses LangGraph's `add_messages` reducer so nodes can return
      `{ "messages": [msg] }` and have it appended (not overwritten).
    """

    # Conversation
    messages: Annotated[List[BaseMessage], add_messages]

    # Routing
    confidence: Annotated[Optional[float], Field(ge=0, le=1)]
    locked_route: Optional[str]
    routing_attempts: int
    triage_question: Optional[str]
    triage_summary: Optional[str]

    # Per-route attempts / escalation
    attempts: Dict[str, int]

    # RAG output (its a list of docs)
    retrieved: Optional[List[Dict[str, Any]]]
