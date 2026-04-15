from langgraph.graph import StateGraph, START, END
from app.core.graph.state import ChatState
from app.core.graph.route_handler.factory_and_nodes import make_route_subgraph, route_node_name
from app.core.graph.route_classifier.nodes import node__classify_user_intent
from app.core.graph.nodes import node__clarify, node__handoff, node__end_of_turn, end_turn_node_name
from app.core.graph.flow_logging import wrap_node
from app.core.utils import get_routes
from app.core.persistence import get_sqlite_checkpointer

from app.core.graph.routing_edges import (
    handler_edge_map,
    route_after_classifier,
    route_after_handler,
    route_from_start_precheck,
)

# Routes are config-driven (config/routes.(yaml|yml))
ROUTES = get_routes()

# Pre-build route subgraphs
subgraphs = {route: make_route_subgraph(route) for route in ROUTES}


def build_graph(checkpointer=None) -> StateGraph:
    """
    Compile the full chatbot graph.

    Args:
        checkpointer: LangGraph checkpointer to use. Defaults to a synchronous
            SqliteSaver (suitable for CLI). Pass an AsyncSqliteSaver when
            calling from an async context (e.g. the dev API).
    """
    if checkpointer is None:
        checkpointer = get_sqlite_checkpointer()
    g = StateGraph(ChatState)

    # --------------------------------------------------------------------------------------------
    # Nodes

    # Classifier (intent classification + triage question)
    g.add_node("classifier", wrap_node(
        "classifier", node__classify_user_intent))
    g.add_node("clarify", wrap_node("clarify", node__clarify))
    g.add_node("handoff", wrap_node("handoff", node__handoff))

    # Spokes - Route-specific handler subgraphs
    for route in ROUTES:
        g.add_node(route_node_name(route), wrap_node(
            route_node_name(route), subgraphs[route]))

    # End-of-turn finalizer (state hygiene + END)
    g.add_node(end_turn_node_name(), wrap_node(
        end_turn_node_name(), node__end_of_turn))

    # START -> precheck router
    start_map = {"classifier": "classifier",
                 "clarify": "clarify", "handoff": "handoff"}
    start_map.update(handler_edge_map(ROUTES))  # handlers by name
    g.add_conditional_edges(START, route_from_start_precheck, start_map)

    # clarify/handoff end the turn
    g.add_edge("clarify", end_turn_node_name())
    g.add_edge("handoff", end_turn_node_name())
    # classifier -> handler or end_of_turn
    after_classifier_map = {end_turn_node_name(): end_turn_node_name()}
    after_classifier_map.update(handler_edge_map(ROUTES))
    g.add_conditional_edges(
        "classifier", route_after_classifier, after_classifier_map)

    # handler -> hub if topic switch else end_of_turn
    for r in ROUTES:
        g.add_conditional_edges(
            route_node_name(r),
            route_after_handler,
            {"classifier": "classifier", end_turn_node_name(): end_turn_node_name()},
        )

    g.add_edge(end_turn_node_name(), END)

    return g.compile(checkpointer=checkpointer)

    # ADD SESSION TIMEOUTs
    # when changing topic  the assistant dont ask anything
