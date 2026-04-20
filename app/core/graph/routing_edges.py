from __future__ import annotations
from app.core.graph.route_classifier.chain import max_routing_attempts_before_handoff
from app.core.graph.state import ChatState, get_last_msg
from app.core.graph.msg_heuristics_no_llm import asked_for_human, is_low_info
from app.core.graph.route_handler.factory_and_nodes import route_node_name
from app.core.graph.nodes import end_turn_node_name

from typing import Dict
from app.core.utils import is_valid_route
from langchain_core.messages import HumanMessage


def is_locked(state: ChatState) -> bool:
    """True when state has a valid locked_route."""
    return is_valid_route(state.get("locked_route"))


def _check_recent_messages_for_human_request(messages: list) -> bool:
    """Check recent HumanMessages for human request.

    Looks at the last 3 user messages to catch handoff requests even if
    interspersed with assistant responses. This prevents getting stuck in
    loops when a user asks for human while locked in a product handler.
    """
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    # Check last 3 human messages (most recent first)
    for msg in reversed(human_messages[-3:]):
        if asked_for_human(msg.content or ""):
            return True
    return False


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
    """After a handler: always end the turn.

    Topic switches (handler cleared locked_route) are handled on the next
    user message via route_from_start_precheck → classifier.  Routing back
    to classifier within the same ainvoke created an infinite loop when the
    classifier re-locked to the same route the handler just cleared.
    """
    return end_turn_node_name()


def handler_edge_map(routes: list[str]) -> Dict[str, str]:
    """Mapping used by add_conditional_edges for all handler nodes."""
    return {route_node_name(r): route_node_name(r) for r in routes}


def route_from_start_precheck(state: ChatState) -> str:
    """
    START router:
      0) human request (priority!) - always escalate
      1) handoff (routing attempts / solve attempts exceeded)
      2) clarify (low info)
      3) if locked => route handler
      4) else => classifier
    """
    messages = state.get("messages") or []
    last_msg = get_last_msg(messages)

    routing_attempts = int(state.get("routing_attempts") or 0)
    solve_attempts = int(state.get("solve_attempts") or 0)
    max_solve_attempts = int(state.get("max_solve_attempts") or 0)

    locked = state.get("locked_route")

    # 0) explicit human request ALWAYS takes priority (even if other checks would pass)
    # Check both last message AND recent messages to avoid getting stuck in loops
    if asked_for_human(last_msg) or _check_recent_messages_for_human_request(messages):
        return "handoff"

    # 1) routing attempts cap
    max_routing = max_routing_attempts_before_handoff()
    if max_routing and routing_attempts >= max_routing:
        return "handoff"

    # 2) solve attempts cap (only if locked)
    if is_valid_route(locked) and max_solve_attempts and solve_attempts >= max_solve_attempts:
        return "handoff"

    # 3) low-info clarify — only before a route is locked.
    # Once locked, short replies like "si" / "ok" are valid in-conversation responses;
    # the handler has full history to interpret them correctly.
    if not is_valid_route(locked) and is_low_info(last_msg):
        return "clarify"

    # 4) locked => handler, else classifier
    if is_valid_route(locked):
        return route_node_name(locked)

    return "classifier"
