from __future__ import annotations

from app.types import ChatState
# from app.utils import is_valid_route

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
    - If a valid route is already locked, skip triage LLM and jump to `handling`.
    - Otherwise, run the triage LLM node (`classify_user_intent`).
    """
    # locked = state.get("locked_route")
    # if is_valid_route(locked):
    if state.get("locked_route"):
        return {"phase": "triage", "next": "handling"}
    return {"phase": "triage", "next": "classify_user_intent"}


def node__handling(state: ChatState) -> ChatState:
    """Top-level HANDLING phase controller.

    Incremental behavior:
    - Dispatch to `handle__X` if state.next already points there.
    - Else derive handler from `locked_route`.
    - Else fall back to triage (recover).

    # - If `next` already points to a handler node (handle__X), keep it.
    # - Otherwise, derive handler from locked_route.
    # - If the locked route is missing/invalid, bounce back to triage.
    """
    nxt = state.get("next")
    if isinstance(nxt, str) and nxt.startswith("handle__"):
        return {"phase": "handling", "next": nxt}

    locked = state.get("locked_route")
    if locked:  # is_valid_route(locked):
        return {"phase": "handling", "next": f"handle__{locked}"}

    else:  # If we dont have a locked route, then re-run triage
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
