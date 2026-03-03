from __future__ import annotations
from app.core.graph.route_classifier.chain import max_routing_attempts_before_handoff
from app.core.graph.state import ChatState, get_last_msg
from app.core.graph.msg_heuristics_no_llm import asked_for_human, is_low_info
from app.core.graph.route_handler.factory_and_nodes import route_node_name
from app.core.graph.nodes import end_turn_node_name

from typing import Dict
from app.core.utils import is_valid_route


def is_locked(state: ChatState) -> bool:
    """True when state has a valid locked_route."""
    return is_valid_route(state.get("locked_route"))


# def route_from_start(state: ChatState) -> str:
#     """START router: go straight to handler if locked, else to classifier."""
#     if is_locked(state):
#         return route_node(state.get("locked_route"))
#     return "classifier"


def route_after_classifier(state: ChatState) -> str:
    """After classifier:
    - If we locked a route: run its handler.
    - Otherwise: end the current invocation (we may be at triage).
    """
    if is_locked(state):
        return route_node_name(state.get("locked_route"))
    return end_turn_node_name()


def route_after_handler(state: ChatState) -> str:
    """After a handler:
    - If handler cleared lock (topic switch): go back to classifier.
    - Otherwise: end the invocation.
    """
    if state.get("locked_route") is None:
        return "classifier"
    return end_turn_node_name()


def handler_edge_map(routes: list[str]) -> Dict[str, str]:
    """Mapping used by add_conditional_edges for all handler nodes."""
    return {route_node_name(r): route_node_name(r) for r in routes}


def route_from_start_precheck(state: ChatState) -> str:
    """
    START router:
      1) handoff (human request / attempts exceeded)
      2) clarify (low info)
      3) if locked => route handler
      4) else => classifier
    """
    last_msg = get_last_msg(state.get("messages") or [])

    routing_attempts = int(state.get("routing_attempts") or 0)
    solve_attempts = int(state.get("solve_attempts") or 0)
    max_solve_attempts = int(state.get("max_solve_attempts") or 0)

    locked = state.get("locked_route")

    # 1) explicit human request always wins
    if asked_for_human(last_msg):
        return "handoff"

    # 2) routing attempts cap
    max_routing = max_routing_attempts_before_handoff()
    if max_routing and routing_attempts >= max_routing:
        return "handoff"

    # 3) solve attempts cap (only if locked)
    if is_valid_route(locked) and max_solve_attempts and solve_attempts >= max_solve_attempts:
        return "handoff"

    # 4) low-info clarify (generic or route-specific)
    if is_low_info(last_msg):
        return "clarify"

    # 5) locked => handler, else classifier
    if is_valid_route(locked):
        return route_node_name(locked)

    return "classifier"
