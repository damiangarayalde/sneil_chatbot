from __future__ import annotations

from app.core.types import ChatState


def node__finalize_turn(state: ChatState) -> ChatState:
    """Finalize a single graph invocation ("turn").

    Why this exists:
    - In the hub-and-spoke architecture, we frequently want to END the current
      invocation after either:
        * the classifier asks a clarifying question, or
        * a route handler produces an answer.
    - This node is a convenient place to do small, predictable state hygiene
      without scattering it across edges.

    Notes
    - We do NOT try to decide resolution here. Route subgraphs may set `phase="closed"`
      when they explicitly detect the case is resolved.
    - We only set `phase` when it is missing / still in a transient value.
    """

    # If a handler already decided the case is closed, do not override it.
    if state.get("phase") == "closed":
        return {}

    updates: dict = {}

    # If we asked a triage question this turn, we are in triage.
    if state.get("triage_question"):
        updates["phase"] = "triage"
        return updates

    locked = state.get("locked_route")

    # If no route is locked, we are in triage (ready to classify next turn).
    if locked is None:
        updates["phase"] = "triage"

        # Optional hygiene: after a topic switch, confidence/attempts from a previous
        # route can be misleading.
        if state.get("routing_attempts"):
            updates["routing_attempts"] = 0
        if state.get("confidence") is not None:
            updates["confidence"] = 0.0

        return updates

    # Otherwise we are mid-handling (locked route exists across turns).
    updates["phase"] = "handling"
    return updates


# def node__closed(state: ChatState) -> ChatState:
#     """End-of-run node (legacy name).

#     Nuance:
#     - We use this node to end the current graph invocation.
#     - We only mark the *case* as closed when we were in the handling phase.
#       (When triage asks a clarifying question, we end the run but remain in triage.)

#     This function is kept for backward compatibility; prefer `node__finalize_turn`.
#     """
#     # Keep the old behavior, but also reuse the new hygiene logic.
#     if state.get("phase") == "handling":
#         return {"phase": "closed"}
#     return node__finalize_turn(state)
