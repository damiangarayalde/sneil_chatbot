from langgraph.graph import StateGraph, START, END

from app.types import ChatState
from app.nodes.route_subgraph import make_route_subgraph
from app.nodes.user_intent_classifier import node__classify_user_intent
from app.nodes.user_intent_router import node__route_by_user_intent
from typing import Callable, Any


def _wrap_node(name: str, fn: Callable[[ChatState], ChatState]) -> Callable[[ChatState], ChatState]:
    """Wrap a node function (or compiled subgraph callable) to print
    concise entry/exit debug information.

    Prints: node name, a few key state fields on entry and the returned
    partial state on exit.
    """

    def _fmt_state_summary(s: ChatState) -> str:
        return (
            f"phase={s.get('phase')!r} next={s.get('next')!r} locked_route={s.get('locked_route')!r} "
            f"routing_attempts={s.get('routing_attempts')!r} handling_channel={s.get('handling_channel')!r}"
        )

    def wrapper(state: ChatState) -> ChatState:
        try:
            print(f"[GRAPH] Enter node: {name} | {_fmt_state_summary(state)}")
        except Exception:
            print(f"[GRAPH] Enter node: {name}")

        # Support plain callables and compiled StateGraph objects
        try:
            if callable(fn):
                result = fn(state)
            elif hasattr(fn, "invoke"):
                result = fn.invoke(state)
            else:
                # Fallback: attempt attribute-based invocation
                result = fn(state)
        except TypeError:
            # If the wrapped object isn't callable, but exposes 'invoke', use it.
            if hasattr(fn, "invoke"):
                result = fn.invoke(state)
            else:
                raise

        try:
            print(f"[GRAPH] Exit  node: {name} -> {result}")
        except Exception:
            print(f"[GRAPH] Exit  node: {name}")
        return result

    return wrapper


# later: add more routes like "GENKI", "CARJACK", "MAYORISTA", "CALDERA".
ROUTES = ["TPMS", "AA", "CLIMATIZADOR"]

# Pre-build route subgraphs
subgraphs = {route: make_route_subgraph(route) for route in ROUTES}


def node__triage(state: ChatState) -> ChatState:
    """High-level triage phase controller.

    Incremental behavior:
    - If a route is already locked, skip triage LLM and jump to `handling`.
    - Otherwise, run the triage LLM node (`classify_user_intent`).
    """
    locked = state.get("locked_route")
    if locked in ROUTES:
        return {"phase": "triage", "next": "handling"}
    return {"phase": "triage", "next": "classify_user_intent"}


def node__handling(state: ChatState) -> ChatState:
    """High-level handling phase controller.

    Incremental behavior:
    - Dispatch to `handle__X` if state.next already points there.
    - Else derive handler from locked_route / handling_channel.
    - Else fall back to triage.
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
    """End-of-run node.

    Important nuance for incremental development:
    - We use this node to end the current graph invocation.
    - We only mark the *case* as closed when we were in the handling phase.
      (When triage asks a clarifying question, we end the run but remain in triage.)

    Later:
    - Separate "turn_end" vs "case_closed" explicitly.
    """
    if state.get("phase") == "handling":
        return {"phase": "closed"}
    return {}


def build_graph() -> StateGraph:
    g = StateGraph(ChatState)

    # Phase nodes (wrapped with debug helper)
    g.add_node("triage", _wrap_node("triage", node__triage))
    g.add_node("handling", _wrap_node("handling", node__handling))
    g.add_node("closed", _wrap_node("closed", node__closed))

    # Triage LLM + routing
    g.add_node("classify_user_intent", _wrap_node(
        "classify_user_intent", node__classify_user_intent))
    g.add_node("route_by_user_intent", _wrap_node(
        "route_by_user_intent", node__route_by_user_intent))

    # Route handlers
    for route in ROUTES:
        # subgraphs[route] is a compiled StateGraph/callable — wrap it too
        g.add_node(f"handle__{route}", _wrap_node(
            f"handle__{route}", subgraphs[route]))

    # START -> triage
    g.add_edge(START, "triage")

    # triage -> (triage_llm or handling)
    g.add_conditional_edges(
        "triage",
        lambda s: s.get("next"),
        {
            "classify_user_intent": "classify_user_intent",
            "handling": "handling",
        },
    )

    # classify_user_intent can either:
    # - ask a clarifying question and end the run (`closed`)
    # - proceed to routing/handling when confident.
    g.add_conditional_edges(
        "classify_user_intent",
        lambda s: s.get("next"),
        {
            "closed": "closed",
            "route_by_user_intent": "route_by_user_intent",
        },
    )

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

    return g.compile()
