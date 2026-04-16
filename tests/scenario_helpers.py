"""
Shared helpers for scenario-driven integration tests.

Each integration test loads a JSON bundle (node_ref + scenarios list),
resolves the node callable, and delegates to run_scenario / _assert_expect.

Bundle JSON shapes:

A) Direct node callable:
   { "node_ref": "module.path:callable_name", "scenarios": [...] }

B) Route subgraph node wrapper:
   { "node_ref": "app.core.graph.route_handler.factory_and_nodes:make_route_subgraph",
     "route_id": "AA", "node_name": "generate", "scenarios": [...] }

route_id / node_name can also be provided inside scenario.setup to override
bundle-level defaults on a per-scenario basis.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch as mock_patch, MagicMock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

Scenario = Dict[str, Any]
NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Import helpers

def _import_from_ref(ref: str) -> Any:
    """Import 'module:attr' (attr can be dotted)."""
    if ":" not in ref:
        raise ValueError(
            f"Invalid node_ref '{ref}'. Expected format: module.path:callable_name")
    mod_name, attr_path = ref.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Route subgraph wrapper

def _extract_graph_nodes_dict(graph_obj: Any) -> Dict[str, Any]:
    for attr in ("nodes", "_nodes", "node_map", "_node_map"):
        if hasattr(graph_obj, attr):
            d = getattr(graph_obj, attr)
            if isinstance(d, dict):
                return d
    raise TypeError(
        "Could not locate a node dictionary on the graph object. "
        "Tried attributes: nodes, _nodes, node_map, _node_map."
    )


def _as_node_callable(node_obj: Any) -> NodeFn:
    if callable(node_obj):
        def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
            return node_obj(state) or {}
        return _fn

    if hasattr(node_obj, "invoke") and callable(getattr(node_obj, "invoke")):
        def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
            return node_obj.invoke(state) or {}
        return _fn

    raise TypeError(
        f"Node object is neither callable nor has .invoke(): {type(node_obj)}")


def _wrap_route_subgraph_node(graph_factory: Callable[..., Any], route_id: str, node_name: str) -> NodeFn:
    g = graph_factory(route_id)
    nodes = _extract_graph_nodes_dict(g)
    if node_name not in nodes:
        available = ", ".join(sorted(nodes.keys()))
        raise KeyError(
            f"Node '{node_name}' not found in graph. Available nodes: {available}")
    return _as_node_callable(nodes[node_name])


# ---------------------------------------------------------------------------
# Loading + resolving

def load_scenarios_json(path: str | Path) -> Tuple[Any, List[Scenario], Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Scenario bundle must be a JSON object. Got: {type(data)}")

    node_ref = data.get("node_ref")
    scenarios = data.get("scenarios")

    if not isinstance(node_ref, str) or not node_ref.strip():
        raise ValueError("'node_ref' is required and must be a non-empty string.")
    if not isinstance(scenarios, list):
        raise ValueError("'scenarios' is required and must be a list.")

    obj = _import_from_ref(node_ref)
    if not callable(obj):
        raise TypeError(f"Resolved node_ref is not callable: {node_ref}")

    bundle_meta = {
        "route_id": data.get("route_id"),
        "node_name": data.get("node_name"),
        "node_ref": node_ref,
    }
    return obj, scenarios, bundle_meta


def _resolve_node_fn(obj: Any, bundle_meta: Dict[str, Any], scenario: Scenario) -> NodeFn:
    setup = scenario.get("setup") or {}
    route_id = setup.get("route_id", bundle_meta.get("route_id"))
    node_name = setup.get("node_name", bundle_meta.get("node_name"))

    if route_id and node_name:
        return _wrap_route_subgraph_node(obj, str(route_id), str(node_name))

    def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
        return obj(state) or {}
    return _fn


# ---------------------------------------------------------------------------
# State helpers

def _apply_patch(state: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a LangGraph-style partial update to a local state dict."""
    if not patch:
        return state
    for k, v in patch.items():
        if k == "messages":
            state.setdefault("messages", [])
            state["messages"].extend(v or [])
        else:
            state[k] = v
    return state


def _last_ai_message(messages: List[BaseMessage]) -> Optional[str]:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage):
            return (m.content or "").strip()
    return None


def _matches_expect(state: Dict[str, Any], expect: Dict[str, Any]) -> bool:
    """Return True if all checks in expect pass against state."""
    st = (expect or {}).get("state") or {}
    for key, val in st.items():
        if key == "confidence_min":
            conf = state.get("confidence")
            if conf is None or float(conf) < float(val):
                return False
        elif key == "confidence_max":
            conf = state.get("confidence")
            if conf is None or float(conf) > float(val):
                return False
        else:
            if state.get(key) != val:
                return False

    action = (expect or {}).get("action")
    if action:
        last_ai = _last_ai_message(state.get("messages") or [])
        if action == "ask_clarifier":
            if not last_ai:
                return False
            if (expect.get("assistant") or {}).get("ends_with_qmark", True) and not last_ai.endswith("?"):
                return False
        elif action == "lock_route":
            if not state.get("locked_route"):
                return False
        elif action == "escalate":
            if state.get("escalated_to_human") is not True:
                return False

    a = (expect or {}).get("assistant") or {}
    if a:
        last_ai = _last_ai_message(state.get("messages") or []) or ""
        for s in a.get("contains") or []:
            if s not in last_ai:
                return False

    return True


def _assert_expect(state: Dict[str, Any], expect: Dict[str, Any]) -> None:
    if not _matches_expect(state, expect):
        last_ai = _last_ai_message(state.get("messages") or [])
        raise AssertionError(
            f"Expectation failed\n"
            f"  expected: {expect}\n"
            f"  got:"
            f"  locked_route={state.get('locked_route')}"
            f"  escalated_to_human={state.get('escalated_to_human')}"
            f"  solve_attempts={state.get('solve_attempts')}"
            f"  routing_attempts={state.get('routing_attempts')}"
            f"  confidence={state.get('confidence')}\n"
            f"  last_ai={repr(last_ai)}"
        )


# ---------------------------------------------------------------------------
# Mocks

@contextlib.contextmanager
def _apply_scenario_mocks(scenario: Scenario, bundle_meta: Dict[str, Any]):
    """Patch LLM chain and/or retriever for the duration of one node call."""
    setup = scenario.get("setup") or {}
    mocks = setup.get("mocks") or {}
    route_id = setup.get("route_id") or bundle_meta.get("route_id")

    with contextlib.ExitStack() as stack:
        llm_output = mocks.get("llm_output")
        if llm_output and route_id:
            from app.core.graph.route_handler.models import HandlerOutput
            mock_response = HandlerOutput(**llm_output)
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            stack.enter_context(
                mock_patch(
                    "app.core.graph.route_handler.chain.get_route_chain",
                    return_value=mock_chain,
                )
            )

        retriever_docs = mocks.get("retriever_docs")
        if retriever_docs is not None:
            from langchain_core.documents import Document
            docs = [
                Document(page_content=d["page_content"], metadata=d.get("metadata") or {})
                for d in retriever_docs
            ]
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = docs
            stack.enter_context(
                mock_patch(
                    "app.core.graph.route_handler.factory_and_nodes._get_route_retriever",
                    return_value=mock_retriever,
                )
            )

        # tool_router_mock: list of {name, args} dicts (or [] for no tool calls).
        # Controls which tool calls the LLM "selects" during the tool-router step.
        # Set to "raise" to simulate a tool-router failure (graceful degradation test).
        tool_router_mock = mocks.get("tool_router_mock")
        if tool_router_mock is not None:
            mock_tr = MagicMock()
            if tool_router_mock == "raise":
                mock_tr.invoke.side_effect = RuntimeError("simulated tool router failure")
            else:
                from langchain_core.messages import AIMessage as _AIMessage
                # LangChain requires each tool_call entry to have an 'id' field.
                tool_calls_with_id = [
                    {**tc, "id": tc.get("id", f"call_{i}")}
                    for i, tc in enumerate(tool_router_mock)
                ]
                mock_response = _AIMessage(content="", tool_calls=tool_calls_with_id)
                mock_tr.invoke.return_value = mock_response
            stack.enter_context(
                mock_patch(
                    "app.core.graph.route_handler.factory_and_nodes.get_tool_router_llm",
                    return_value=mock_tr,
                )
            )

        # catalog_mock: dict returned by catalog_lookup (replaces real catalog file read).
        catalog_mock = mocks.get("catalog_mock")
        if catalog_mock is not None:
            stack.enter_context(
                mock_patch(
                    "app.core.graph.route_handler.factory_and_nodes.catalog_lookup",
                    return_value=catalog_mock,
                )
            )

        yield


# ---------------------------------------------------------------------------
# Runner

def run_scenario(node_fn: NodeFn, s: Scenario, bundle_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bundle_meta = bundle_meta or {}
    state: Dict[str, Any] = {"messages": []}
    state.update((s.get("setup") or {}).get("initial_state") or {})

    print(f"\n{'=' * 80}")
    print(f"{s.get('id')}: {s.get('title')}")

    for i, turn in enumerate(s.get("turns") or [], start=1):
        user_msg_text = turn["user_msg"]
        state.setdefault("messages", []).append(HumanMessage(content=user_msg_text))

        with _apply_scenario_mocks(s, bundle_meta):
            state_update = node_fn(state) or {}
        state = _apply_patch(state, state_update)

        last_ai = _last_ai_message(state.get("messages") or [])

        print(f"\n  Turn {i} > user: {repr(user_msg_text)}")
        if last_ai:
            print(f"  assistant: {last_ai}")
        print(f"  state: locked={state.get('locked_route')}  "
              f"escalated={state.get('escalated_to_human')}  "
              f"solve_attempts={state.get('solve_attempts')}")

        exp = turn.get("expect")
        if not exp:
            continue

        if isinstance(exp, dict) and "one_of" in exp:
            options = exp["one_of"] or []
            if not any(_matches_expect(state, opt) for opt in options):
                raise AssertionError(
                    f"No expectation in one_of matched\n"
                    f"  one_of: {options}\n"
                    f"  got: locked_route={state.get('locked_route')} "
                    f"escalated_to_human={state.get('escalated_to_human')} "
                    f"solve_attempts={state.get('solve_attempts')} "
                    f"last_ai={repr(last_ai)}"
                )
        else:
            _assert_expect(state, exp)

    return state
