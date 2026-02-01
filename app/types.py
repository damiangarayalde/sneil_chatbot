from langgraph.graph.message import add_messages
from typing import Annotated
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from pydantic import Field


class ChatState(TypedDict):
    """Shared TypedDict describing the shape of the chat/graph state.

    This centralizes the state type so modules can import it without
    creating circular imports between node implementations and the main
    graph builder.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    handling_channel:  str | None
    # product_family:  str | None
    confidence: float | None

    # attempts: Dict[str, int]
    # retrieved: Optional[List[Dict[str, Any]]]
    # answer:  str | None
