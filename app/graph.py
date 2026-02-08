from langgraph.graph import StateGraph, START, END
from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.user_intent_router import node__route_by_user_intent
from app.nodes.phase_nodes import node__triage, node__handling, node__closed
from app.graph_utils import wrap_node
from app.utils import get_routes, is_valid_route
from app.persistence import get_sqlite_checkpointer

# Routes are config-driven (config/routes.(yaml|yml))
ROUTES = get_routes()

# Pre-build route subgraphs
subgraphs = {route: make_route_subgraph(route) for route in ROUTES}

# Initialize checkpointer via the new utility
memory = get_sqlite_checkpointer()


def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    # Phase nodes
    g.add_node("triage",    wrap_node("triage",     node__triage))
    g.add_node("handling",  wrap_node("handling",   node__handling))
    g.add_node("closed",    wrap_node("closed",     node__closed))

    # Triage LLM + routing
    g.add_node("classify_user_intent", wrap_node(
        "classify_user_intent", node__classify_user_intent))
    g.add_node("route_by_user_intent", wrap_node(
        "route_by_user_intent", node__route_by_user_intent))

    # Route handlers
    for route in ROUTES:
        g.add_node(f"handle__{route}", wrap_node(
            f"handle__{route}", subgraphs[route]))

    # --------------------------------------------------------------------------------------------
    g.add_edge(START, "triage")

    # --- NEW: edge decision helpers (do not rely on state["next"]) ---
    def _after_triage(state: ChatState) -> str:
        locked = state.get("locked_route")
        if is_valid_route(locked):
            return "handling"
        return "classify_user_intent"

    def _after_classifier(state: ChatState) -> str:
        # If classifier asked a question, end this invocation
        if state.get("triage_question"):
            return "closed"
        locked = state.get("locked_route")
        if is_valid_route(locked):
            return "route_by_user_intent"
        # Safe fallback
        return "closed"

    def _after_handling(state: ChatState) -> str:
        # Prefer explicit handler if already set
        nxt = state.get("next")
        if isinstance(nxt, str) and nxt.startswith("handle__"):
            return nxt

        locked = state.get("locked_route")
        if is_valid_route(locked):
            return f"handle__{locked}"
        return "triage"

    # if locked_route present, skip classification and go straight to handling
    g.add_conditional_edges("triage", _after_triage,
                            {
                                "classify_user_intent": "classify_user_intent",
                                "handling": "handling"
                            })

    # classify_user_intent can either:
    # - ask a clarifying question and end the run (`closed`)
    # - proceed to routing/handling when confident.
    g.add_conditional_edges("classify_user_intent", _after_classifier,
                            {
                                "closed": "closed",
                                "route_by_user_intent": "route_by_user_intent"
                            })

    # router -> handling
    g.add_edge("route_by_user_intent", "handling")

    # handling -> (dispatch to handler or back to triage)
    handling_edge_map = {"triage": "triage"}
    handling_edge_map.update({f"handle__{r}": f"handle__{r}" for r in ROUTES})
    g.add_conditional_edges("handling", _after_handling, handling_edge_map)

    # handler -> closed -> END
    for r in ROUTES:
        g.add_edge(f"handle__{r}", "closed")
    g.add_edge("closed", END)

    # Compile with the persistent memory instance
    return g.compile(checkpointer=memory)

    # ADD SESSION TIMEOUTs
