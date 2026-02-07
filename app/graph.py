import sqlite3  # New import
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.user_intent_router import node__route_by_user_intent
from app.nodes.phase_nodes import node__triage, node__handling, node__closed
from app.graph_utils import wrap_node
from app.utils import get_routes

# Routes are config-driven (config/routes.(yaml|yml))
ROUTES = get_routes()

# Pre-build route subgraphs
subgraphs = {route: make_route_subgraph(route) for route in ROUTES}

# Create a persistent connection to the database
# check_same_thread=False is important for WhatsApp/Web applications
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    # Phase nodes
    g.add_node("triage", wrap_node("triage", node__triage))
    g.add_node("handling", wrap_node("handling", node__handling))
    g.add_node("closed", wrap_node("closed", node__closed))

    # Triage LLM + routing
    g.add_node("classify_user_intent", wrap_node(
        "classify_user_intent", node__classify_user_intent))
    g.add_node("route_by_user_intent", wrap_node(
        "route_by_user_intent", node__route_by_user_intent))

    # Route handlers
    for route in ROUTES:
        g.add_node(f"handle__{route}", wrap_node(
            f"handle__{route}", subgraphs[route]))

    # START -> triage
    g.add_edge(START, "triage")
    # triage -> (classifier or handling)
    g.add_conditional_edges("triage", lambda s: s.get("next"), {
                            "classify_user_intent": "classify_user_intent", "handling": "handling"})
    # classify_user_intent can either:
    # - ask a clarifying question and end the run (`closed`)
    # - proceed to routing/handling when confident.
    g.add_conditional_edges("classify_user_intent", lambda s: s.get(
        "next"), {"closed": "closed", "route_by_user_intent": "route_by_user_intent"})
    # router -> handling
    g.add_edge("route_by_user_intent", "handling")

    # handling -> (dispatch to handler or back to triage)
    handling_edge_map = {"triage": "triage"}
    handling_edge_map.update({f"handle__{r}": f"handle__{r}" for r in ROUTES})
    g.add_conditional_edges(
        "handling", lambda s: s.get("next"), handling_edge_map)

    # handler -> closed -> END
    for r in ROUTES:
        g.add_edge(f"handle__{r}", "closed")
    g.add_edge("closed", END)

    # Compile with the persistent memory instance
    return g.compile(checkpointer=memory)

    # ADD SESSION TIMEOUTs
