from __future__ import annotations

from app.types import ChatState

# NOTE:
# These are *phase controller* nodes for the top-level graph.
# They are intentionally LLM-free and only do control-flow decisions.
#
# Prompt ideas (comment only):
# - triage: later can emit a welcome message on first turn.
# - handling: later can enforce per-module limits (max turns, escalation).
# - closed: later can write final summaries / analytics.


def node__triage(state: ChatState) -> ChatState:
    """Top-level TRIAGE phase controller.

    Incremental behavior:
    - If a route is already locked, skip triage LLM and jump to `handling`.
    - Otherwise, run the triage LLM node (`classify_user_intent`).
    """
    if state.get("locked_route"):
        return {"phase": "triage", "next": "handling"}
    return {"phase": "triage", "next": "classify_user_intent"}


def node__handling(state: ChatState) -> ChatState:
    """Top-level HANDLING phase controller.

    Incremental behavior:
    - Dispatch to `handle__X` if state.next already points there.
    - Else derive handler from `locked_route`.
    - Else fall back to triage (recover).

    """
    nxt = state.get("next")
    if isinstance(nxt, str) and nxt.startswith("handle__"):
        return {"phase": "handling", "next": nxt}

    locked = state.get("locked_route")
    if locked:
        return {"phase": "handling", "next": f"handle__{locked}"}

    else:  # If we don't have a locked route, we cannot safely dispatch.
        return {"phase": "handling", "next": "triage"}


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
