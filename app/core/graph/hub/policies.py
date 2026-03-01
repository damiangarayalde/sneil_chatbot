from __future__ import annotations

from app.core.graph.msg_heuristics_no_llm import asked_for_human, is_low_info
from app.core.graph.node_names import route_node
from app.core.graph.state import ChatState, get_last_msg
from app.core.utils import is_valid_route

from .chain import max_routing_attempts_before_handoff


def route_from_start_precheck(state: ChatState) -> str:
    """
    START router:
      1) handoff (human request / attempts exceeded)
      2) clarify (low info)
      3) if locked => route handler
      4) else => hub
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

    # 5) locked => handler, else hub
    if is_valid_route(locked):
        return route_node(locked)

    return "hub"
