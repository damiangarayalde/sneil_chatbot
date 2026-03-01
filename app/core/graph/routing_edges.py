from __future__ import annotations

from typing import Dict
from app.core.graph.state import ChatState
from app.core.utils import is_valid_route
from app.core.graph.node_names import route_node, end_turn_node


def is_locked(state: ChatState) -> bool:
    """True when state has a valid locked_route."""
    return is_valid_route(state.get("locked_route"))


# def route_from_start(state: ChatState) -> str:
#     """START router: go straight to handler if locked, else to hub."""
#     if is_locked(state):
#         return route_node(state.get("locked_route"))
#     return "hub"


def route_after_hub(state: ChatState) -> str:
    """After hub:
    - If we locked a route: run its handler.
    - Otherwise: end the current invocation (we may be at triage).
    """
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


def handler_edge_map(routes: list[str]) -> Dict[str, str]:
    """Mapping used by add_conditional_edges for all handler nodes."""
    return {route_node(r): route_node(r) for r in routes}
