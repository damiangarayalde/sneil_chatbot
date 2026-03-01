from langgraph.graph import StateGraph, START, END
from app.core.graph.state import ChatState
from app.core.graph.routes.factory import make_route_subgraph
from app.core.graph.hub.policies import route_from_start_precheck
from app.core.graph.hub.nodes import (
    node__classify_user_intent,
    node__clarify,
    node__handoff,
)
from app.core.graph.end_of_turn import node__end_of_turn
from app.core.graph.flow_logging import wrap_node
from app.core.utils import get_routes
from app.core.persistence import get_sqlite_checkpointer

from app.core.graph.routing_edges import (
    handler_edge_map,
    route_after_hub,
    route_after_handler,
)
from app.core.graph.node_names import route_node, end_turn_node

# Routes are config-drivens (config/routes.(yaml|yml))
ROUTES = get_routes()

# Pre-build route subgraphs
subgraphs = {route: make_route_subgraph(route) for route in ROUTES}

# Initialize checkpointer via the new utility
memory = get_sqlite_checkpointer()


def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    # --------------------------------------------------------------------------------------------
    # Nodes

    # Hub (intent classification + triage question)
    g.add_node("hub", wrap_node("hub", node__classify_user_intent))
    g.add_node("clarify", wrap_node("clarify", node__clarify))
    g.add_node("handoff", wrap_node("handoff", node__handoff))

    # Spokes - Route-specific handler subgraphs
    for route in ROUTES:
        g.add_node(route_node(route), wrap_node(
            route_node(route), subgraphs[route]))

    # End-of-turn finalizer (state hygiene + END)
    g.add_node(end_turn_node(), wrap_node(end_turn_node(), node__end_of_turn))

    # START -> precheck router
    start_map = {"hub": "hub", "clarify": "clarify", "handoff": "handoff"}
    start_map.update(handler_edge_map(ROUTES))  # handlers by name
    g.add_conditional_edges(START, route_from_start_precheck, start_map)

    # clarify/handoff end the turn
    g.add_edge("clarify", end_turn_node())
    g.add_edge("handoff", end_turn_node())

    # hub -> handler or end_of_turn
    after_hub_map = {end_turn_node(): end_turn_node()}
    after_hub_map.update(handler_edge_map(ROUTES))
    g.add_conditional_edges("hub", route_after_hub, after_hub_map)

    # handler -> hub if topic switch else end_of_turn
    for r in ROUTES:
        g.add_conditional_edges(
            route_node(r),
            route_after_handler,
            {"hub": "hub", end_turn_node(): end_turn_node()},
        )

    g.add_edge(end_turn_node(), END)

    return g.compile(checkpointer=memory)

    # ADD SESSION TIMEOUTs
    # when changing topic  the assistant dont ask anything
