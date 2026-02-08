from __future__ import annotations

from typing import Dict
from app.core.graph.state import ChatState
from app.core.utils import is_valid_route


def route_node(route: str) -> str:
    """Canonical node name for a route handler."""
    return f"handle__{route}"


def is_locked(state: ChatState) -> bool:
    """True when state has a valid locked_route."""
    return is_valid_route(state.get("locked_route"))


def handler_edge_map(routes: list[str]) -> Dict[str, str]:
    """Mapping used by add_conditional_edges for all handler nodes."""
    return {route_node(r): route_node(r) for r in routes}


def end_turn_node() -> str:
    """Single place to define the end-of-turn node name."""
    return "finalize_turn"


def route_from_start(state: ChatState) -> str:
    """START router: go straight to handler if locked, else to hub."""
    if is_locked(state):
        return route_node(state.get("locked_route"))
    return "hub"


def route_after_hub(state: ChatState) -> str:
    """After hub:
    - If we asked a triage question: end the current invocation.
    - If we locked a route: run its handler.
    - Otherwise: end the current invocation.
    """
    if state.get("triage_question"):
        return end_turn_node()
    if is_locked(state):
        return route_node(state.get("locked_route"))
    return end_turn_node()


def route_after_handler(state: ChatState) -> str:
    """After a handler:
    - If handler cleared lock (topic switch): go back to hub.
    - Otherwise: end the invocation.
    """
    if state.get("locked_route") is None:
        return "hub"
    return end_turn_node()
