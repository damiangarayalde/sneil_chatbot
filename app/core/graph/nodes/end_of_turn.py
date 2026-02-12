from __future__ import annotations

from app.core.graph.state import ChatState


def node__end_of_turn(state: ChatState) -> ChatState:
    """Finalize a single graph invocation ("turn").

    Keeping for future use and flexibility
    """

    updates: dict = {}

    return updates
