from app.nodes.product_subgraph import make_product_subgraph
from app.nodes.router import route_node
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    messages: List[BaseMessage]
    route: Optional[str]
    attempts: Dict[str, int]          # attempts per product
    retrieved: Optional[List[Dict[str, Any]]]
    answer: Optional[str]


PRODUCTS = ["TPMS", "AA", "CLIMATIZADOR",
            "GENKI", "CARJACK", "MAYORISTA", "CALDERA"]

subgraphs = {p: make_product_subgraph(p) for p in PRODUCTS}


def dispatch(state):
    r = state.get("route") or "UNKNOWN"
    if r in subgraphs:
        out = subgraphs[r].invoke(state)
        return out
    state["answer"] = "¡Hola! 😊 ¿En qué puedo ayudarte? (TPMS, AA, Climatizadores, Genki, Carjack, Mayorista o Calderas)"
    return state


def build_graph():
    g = StateGraph(dict)
    g.add_node("route", route_node)
    g.add_node("dispatch", dispatch)
    g.set_entry_point("route")
    g.add_edge("route", "dispatch")
    g.add_edge("dispatch", END)
    return g.compile()
