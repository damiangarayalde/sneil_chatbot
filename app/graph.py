from langgraph.graph import StateGraph, START, END
from app.types import ChatState
from app.nodes.product_subgraph import make_product_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent


PRODUCTS = ["TPMS", "AA", "CLIMATIZADOR",
            "GENKI", "CARJACK", "MAYORISTA", "CALDERA"]

# Pre-build product subgraphs (each is a compiled StateGraph)
subgraphs = {p: make_product_subgraph(p) for p in PRODUCTS}


def route_by_user_intent(state: ChatState) -> ChatState:
    """Route to the appropriate product subgraph based on `state['product_family']`.

    If the product family matches a known product, invoke its subgraph; otherwise
    return a friendly greeting asking the user to choose a product.
    """
    r = state.get("product_family") or "UNKNOWN"
    if r in subgraphs:
        out = subgraphs[r].invoke(state)
        return out

    state["answer"] = "¡Hola! 😊 ¿En qué puedo ayudarte? (TPMS, AA, Climatizadores, Genki, Carjack, Mayorista o Calderas)"

    return state


def build_graph() -> StateGraph:

    g = StateGraph(ChatState)

    # Add processing nodes to the graph
    g.add_node("classify_user_intent", node__classify_user_intent)
    g.add_node("route_by_user_intent", route_by_user_intent)

    # Connect the nodes:
    g.add_edge(START, "classify_user_intent")
    g.add_edge("classify_user_intent", "route_by_user_intent")
    g.add_edge("route_by_user_intent", END)

    return g.compile()
