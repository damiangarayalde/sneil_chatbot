from langgraph.graph import StateGraph, START, END
from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.user_intent_router import node__route_by_user_intent

ROUTES = ["TPMS", "AA", "CLIMATIZADOR"]  # ,
#           "GENKI", "CARJACK", "MAYORISTA", "CALDERA"]

# Pre-build route subgraphs (each is a compiled StateGraph)
subgraphs = {p: make_route_subgraph(p) for p in ROUTES}


def build_graph() -> StateGraph:

    g = StateGraph(ChatState)

    # Add processing nodes to the graph
    g.add_node("classify_user_intent", node__classify_user_intent)
    g.add_node("route_by_user_intent", node__route_by_user_intent)

    for handling_channel in ROUTES:
        g.add_node(f"handle__{handling_channel}", subgraphs[handling_channel])

    # Connect the nodes: start -> classify_user_intent -> route_by_user_intent -> (specific route handler subgraph or node) -> end
    g.add_edge(START, "classify_user_intent")
    g.add_edge("classify_user_intent", "route_by_user_intent")

    # Build conditional edge map: router-returned key -> node name
    edge_map = {
        f"handle__{handling_channel}": f"handle__{handling_channel}" for handling_channel in subgraphs.keys()}

    # Use a selector that reads the `next` value from state (set by the router node)
    g.add_conditional_edges(
        "route_by_user_intent", lambda state: state.get("next"), edge_map)

    # Ensure all route handler nodes flow to the END
    for handling_channel in subgraphs.keys():
        g.add_edge(f"handle__{handling_channel}", END)

    return g.compile()
