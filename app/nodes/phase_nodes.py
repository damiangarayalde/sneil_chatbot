from __future__ import annotations

from app.types import ChatState


def node__closed(state: ChatState) -> ChatState:
    """End-of-run node.

    Nuance:
    - We use this node to end the current graph invocation.
    - We only mark the *case* as closed when we were in the handling phase.
      (When triage asks a clarifying question, we end the run but remain in triage.)
    """
    if state.get("phase") == "handling":
        return {"phase": "closed"}
    return {}
