from langgraph.graph import StateGraph, START, END
from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.phase_nodes import node__closed
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

    # Hub
    g.add_node("classify_user_intent", wrap_node(
        "classify_user_intent", node__classify_user_intent))

    # Spokes - Route-specific handler subgraphs
    for route in ROUTES:
        g.add_node(f"handle__{route}", wrap_node(
            f"handle__{route}", subgraphs[route]))

    g.add_node("closed",    wrap_node("closed",     node__closed))

    # --------------------------------------------------------------------------------------------

    # Small helpers to keep graph definitions declarative
    def _route_node(route: str) -> str:
        return f"handle__{route}"

    def _is_locked(state: ChatState) -> bool:
        return is_valid_route(state.get("locked_route"))

    def _start_router(state: ChatState) -> str:
        locked = state.get("locked_route")
        if _is_locked(state):
            return _route_node(locked)
        return "classify_user_intent"

    start_map = {"classify_user_intent": "classify_user_intent"}
    start_map.update({f"handle__{r}": f"handle__{r}" for r in ROUTES})
    g.add_conditional_edges(START, _start_router, start_map)

    def _after_classifier(state: ChatState) -> str:
        # If classifier asked a question, end this invocation
        if state.get("triage_question"):
            return "closed"
        locked = state.get("locked_route")
        if _is_locked(state):
            return _route_node(locked)
        return "closed"

    after_classifier_map = {"closed": "closed"}
    after_classifier_map.update(
        {f"handle__{r}": f"handle__{r}" for r in ROUTES})
    g.add_conditional_edges("classify_user_intent",
                            _after_classifier, after_classifier_map)

    # If the handler clears locked_route (topic switch), go back to hub.
    def _after_handler(state: ChatState) -> str:
        if state.get("locked_route") is None:
            return "classify_user_intent"
        return "closed"

    for r in ROUTES:
        g.add_conditional_edges(
            f"handle__{r}",
            _after_handler,
            {"classify_user_intent": "classify_user_intent", "closed": "closed"},
        )

    g.add_edge("closed", END)

    return g.compile(checkpointer=memory)

    # ADD SESSION TIMEOUTs
