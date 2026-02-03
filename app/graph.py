from langgraph.graph import StateGraph, START, END
from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.user_intent_router import node__route_by_user_intent

# later: "GENKI", "CARJACK", "MAYORISTA", "CALDERA"]
ROUTES = ["TPMS", "AA", "CLIMATIZADOR"]


# Pre-build route subgraphs
subgraphs = {p: make_route_subgraph(p) for p in ROUTES}


def node__triage(state: ChatState) -> ChatState:
    """
    High-level triage phase controller.

    For now (incremental):
    - Always enter triage when a new user message arrives.
    - If a route is already locked, skip classification and go to `handling`.
    - Otherwise, go to classifier.

    Prompt idea (comment only):
    - No LLM here. This is pure flow control.
    - Later you can add a welcome message only if conversation is new.
    """
    locked = state.get("locked_route")
    if locked in ROUTES:
        return {"phase": "triage", "next": "handling"}
    return {"phase": "triage", "next": "classify_user_intent"}


def node__handling(state: ChatState) -> ChatState:
    """
    High-level handling phase controller.

    For now (incremental):
    - Ensure `next` points to the chosen handler node (handle__X)
      and then dispatch into that node.
    - If something is missing, fall back to triage.

    Prompt idea (comment only):
    - No LLM here. Pure flow control / resume logic.
    - Later you can add per-route policies, e.g. max attempts.
    """
    # If the router already set next=handle__X, keep it.
    nxt = state.get("next")
    if isinstance(nxt, str) and nxt.startswith("handle__"):
        return {"phase": "handling", "next": nxt}

    # If we have a locked route, derive handler key from it.
    locked = state.get("locked_route")
    if locked in ROUTES:
        return {"phase": "handling", "next": f"handle__{locked}"}

    # If classifier provided handling_channel, derive handler key from it.
    pf = state.get("handling_channel")
    if pf in ROUTES:
        return {"phase": "handling", "next": f"handle__{pf}"}

    # Otherwise, go back to triage (classifier) to recover.
    return {"phase": "handling", "next": "triage"}


def node__closed(state: ChatState) -> ChatState:
    """
    Final phase node.

    For now:
    - Just mark phase closed. (No message emitted.)
    - Later you can emit a closing message, store a final summary, etc.
    """
    return {"phase": "closed"}


def build_graph() -> StateGraph:

    g = StateGraph(ChatState)

    # Phase nodes
    g.add_node("triage", node__triage)
    g.add_node("handling", node__handling)
    g.add_node("closed", node__closed)

    # Existing nodes
    g.add_node("classify_user_intent", node__classify_user_intent)
    g.add_node("route_by_user_intent", node__route_by_user_intent)

    # Route handlers
    for route in ROUTES:
        g.add_node(f"handle__{route}", subgraphs[route])

    # START -> triage
    g.add_edge(START, "triage")

    # triage -> (classifier or handling)
    g.add_conditional_edges(
        "triage",
        lambda s: s.get("next"),
        {
            "classify_user_intent": "classify_user_intent",
            "handling": "handling",
        },
    )

    # classifier -> router -> handling
    g.add_edge("classify_user_intent", "route_by_user_intent")
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

    return g.compile()
