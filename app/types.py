from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    """Shared TypedDict describing the shape of the chat/graph state.

    This centralizes the state type so modules can import it without
    creating circular imports between node implementations and the main
    graph builder.
    """
    messages: List[BaseMessage]
    handling_channel:  str | None
    product_family:  str | None
    confidence: float | None
    attempts: Dict[str, int]
    retrieved: Optional[List[Dict[str, Any]]]
    answer:  str | None
