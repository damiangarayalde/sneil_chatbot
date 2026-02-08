from langgraph.graph import StateGraph, START, END
from app.core.graph.state import ChatState
from app.core.graph.nodes.route_subgraph import make_route_subgraph
from app.core.graph.nodes.hub import node__classify_user_intent
from app.core.graph.nodes.lifecycle import node__finalize_turn
from app.core.graph_utils import wrap_node
from app.core.utils import get_routes
from app.core.persistence import get_sqlite_checkpointer

from app.core.graph.routes import (
    route_node,
    handler_edge_map,
    route_from_start,
    route_after_hub,
    route_after_handler,
    end_turn_node,
)

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

    # Spokes - Route-specific handler subgraphs
    for route in ROUTES:
        g.add_node(route_node(route), wrap_node(
            route_node(route), subgraphs[route]))

    # End-of-turn finalizer (state hygiene + END)
    g.add_node(end_turn_node(), wrap_node(
        end_turn_node(), node__finalize_turn))

    # Legacy (kept for reference; prefer finalize_turn)
    # from app.nodes.phase_nodes import node__closed
    # g.add_node("closed", wrap_node("closed", node__closed))

    # --------------------------------------------------------------------------------------------
    # Edges

    start_map = {"hub": "hub"}
    start_map.update(handler_edge_map(ROUTES))
    g.add_conditional_edges(START, route_from_start, start_map)

    after_hub_map = {end_turn_node(): end_turn_node()}
    after_hub_map.update(handler_edge_map(ROUTES))
    g.add_conditional_edges("hub", route_after_hub, after_hub_map)

    for r in ROUTES:
        g.add_conditional_edges(
            route_node(r),
            route_after_handler,
            {"hub": "hub", end_turn_node(): end_turn_node()},
        )

    g.add_edge(end_turn_node(), END)

    return g.compile(checkpointer=memory)

    # ADD SESSION TIMEOUTs
