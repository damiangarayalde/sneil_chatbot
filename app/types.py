from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import Field


class ChatState(TypedDict):
    """Shared TypedDict describing the shape of the chat/graph state.

    Centralized so modules can import it without creating circular
    imports between node implementations and the main graph builder.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    # route: Optional[str]
    attempts: Dict[str, int]
    retrieved: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    handling_channel: Optional[str]
    # confidence is optional and constrained to [0, 1]
    confidence: Annotated[Optional[float], Field(ge=0, le=1)]
