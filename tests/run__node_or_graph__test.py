"""
Single runner for node scenarios.

Supports two bundle modes:

A) Direct node callable (original behavior)
{
  "node_ref": "module.path:callable_name",
  "scenarios": [ ... ]
}

B) Route subgraph node wrapper (NEW)
{
  "node_ref": "app.core.graph.nodes.route_subgraph:make_route_subgraph",
  "route_id": "AA",
  "node_name": "generate",
  "scenarios": [ ... ]
}

Also allowed: route_id/node_name can be provided inside scenario.setup, and will override bundle-level.
This lets you keep one bundle and vary per-scenario if desired.

Run:
  python run_test.py path/to/bundle.json
  python run_test.py bundle1.json bundle2.json

  python tests/run__node_or_graph__test.py tests/test__node_route_classifier.scenarios.json  
   
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch as mock_patch, MagicMock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from numpy import rint

Scenario = Dict[str, Any]
NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def green(text): return f"\033[92m{text}\033[0m"
def cyan(text): return f"\033[96m{text}\033[0m"
def blue(text): return f"\033[94m{text}\033[0m"
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Route subgraph wrapper (NEW)

def _extract_graph_nodes_dict(graph_obj: Any) -> Dict[str, Any]:
    """
    Try common attribute names for node storage on StateGraph / compiled graphs.
    We keep this defensive because LangGraph internals can vary by version.
    """
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
    """
    Normalize a LangGraph/LangChain node into a callable(state)->patch.
    Many LangChain runnables expose .invoke(input).
    """
    if callable(node_obj):
        # direct python function (your closures)
        def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
            out = node_obj(state)
            return out or {}
        return _fn

    if hasattr(node_obj, "invoke") and callable(getattr(node_obj, "invoke")):
        def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
            out = node_obj.invoke(state)
            return out or {}
        return _fn

    raise TypeError(
        f"Node object is neither callable nor has .invoke(): {type(node_obj)}")


def _wrap_route_subgraph_node(graph_factory: Callable[..., Any], route_id: str, node_name: str) -> NodeFn:
    """
    Create a node_fn(state)->patch by:
      - building the route subgraph for route_id
      - extracting node callable by node_name
    """
    g = graph_factory(route_id)

    nodes = _extract_graph_nodes_dict(g)
    if node_name not in nodes:
        available = ", ".join(sorted(nodes.keys()))
        raise KeyError(
            f"Node '{node_name}' not found in graph. Available nodes: {available}")

    return _as_node_callable(nodes[node_name])


# -----------------------------------------------------------------------------
# Loading + resolving

def load_scenarios_json(path: str | Path) -> Tuple[Any, List[Scenario], Dict[str, Any]]:
    """
    Load bundle JSON.

    REQUIRED:
      { "node_ref": str, "scenarios": list[Scenario] }

    OPTIONAL (for route subgraph wrapper mode):
      { "route_id": str, "node_name": str }
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"Scenario bundle must be a JSON object. Got: {type(data)}")

    node_ref = data.get("node_ref")
    scenarios = data.get("scenarios")

    if not isinstance(node_ref, str) or not node_ref.strip():
        raise ValueError(
            "'node_ref' is required and must be a non-empty string.")
    if not isinstance(scenarios, list):
        raise ValueError("'scenarios' is required and must be a list.")

    obj = _import_from_ref(node_ref)
    if not callable(obj):
        raise TypeError(f"Resolved node_ref is not callable: {node_ref}")

    # pass through bundle-level extras
    bundle_meta = {
        "route_id": data.get("route_id"),
        "node_name": data.get("node_name"),
        "node_ref": node_ref,
    }
    return obj, scenarios, bundle_meta


def _resolve_node_fn(obj: Any, bundle_meta: Dict[str, Any], scenario: Scenario) -> NodeFn:
    """
    Resolve the actual NodeFn to run for this scenario.

    - If bundle provides route_id+node_name (or scenario.setup does), treat obj as graph_factory(route_id)
      and extract node node_name.
    - Otherwise, treat obj as direct node_fn(state)->patch.
    """
    setup = scenario.get("setup") or {}
    route_id = setup.get("route_id", bundle_meta.get("route_id"))
    node_name = setup.get("node_name", bundle_meta.get("node_name"))

    if route_id and node_name:
        return _wrap_route_subgraph_node(obj, str(route_id), str(node_name))

    # direct mode
    def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
        out = obj(state)
        return out or {}
    return _fn


# -----------------------------------------------------------------------------
# Runner

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
    """Return True if state/dialog checks pass (used for one_of)."""
    # State checks
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

    # Action checks (derive from state/messages)
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
        elif action == "noop":
            pass

    # Assistant content checks
    a = (expect or {}).get("assistant") or {}
    if a:
        last_ai = _last_ai_message(state.get("messages") or []) or ""
        for s in a.get("contains") or []:
            if s not in last_ai:
                return False

    return True


def _assert_expect(state: Dict[str, Any], expect: Dict[str, Any]) -> None:
    """Assert a single expectation; raise AssertionError with useful context."""
    if not _matches_expect(state, expect):
        last_ai = _last_ai_message(state.get("messages") or [])
        raise AssertionError(
            blue("  Expectation failed\n") +
            blue("  - expected: ") + f"{expect}\n" +
            blue("  - got: ") +
            " locked_route = " + cyan(f"{state.get('locked_route')}") +
            " estimated_route = " + cyan(f"{state.get('estimated_route')} ") +
            " escalated_to_human = " + cyan(f"{state.get('escalated_to_human')}") +
            " routing_attempts = " + cyan(f"{state.get('routing_attempts')} ") +
            " solve_attempts = " + cyan(f"{state.get('solve_attempts')} ") +
            " confidence = " + cyan(f"{state.get('confidence')}" + "\n" +
                                    blue("  - last_ai = ") +
                                    cyan(f"{repr(last_ai)}")
                                    ))


@contextlib.contextmanager
def _apply_scenario_mocks(scenario: Scenario, bundle_meta: Dict[str, Any]):
    """Patch LLM chain and/or retriever for the duration of one node call.

    Reads ``scenario.setup.mocks``:
    - ``llm_output``      → patches the route chain's ``.invoke()`` to return a
                            HandlerOutput built from the given dict.
    - ``retriever_docs``  → patches ``_get_route_retriever`` to return a mock
                            retriever whose ``.invoke()`` yields the given docs.
    """
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
                    "app.core.graph.route_handler.factory_and_nodes.get_route_chain",
                    return_value=mock_chain,
                )
            )

        retriever_docs = mocks.get("retriever_docs")
        if retriever_docs:
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=d["page_content"],
                    metadata=d.get("metadata") or {},
                )
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

        yield


def run_scenario(node_fn: NodeFn, s: Scenario, bundle_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    bundle_meta = bundle_meta or {}
    state: Dict[str, Any] = {"messages": []}
    state.update((s.get("setup") or {}).get("initial_state") or {})

    print(green("=" * 100))
    print(green(f"{s.get('id')}: {s.get('title')}"))
    print()
    for k in sorted(state.keys()):
        print(blue("  _initial_ ") + f"state.{k}: " + cyan(f"{state[k]}"))

    for i, turn in enumerate(s.get("turns") or [], start=1):
        user_msg_text = turn["user_msg"]
        state.setdefault("messages", []).append(
            HumanMessage(content=user_msg_text))

        with _apply_scenario_mocks(s, bundle_meta):
            state_update = node_fn(state) or {}
        state = _apply_patch(state, state_update)

        last_ai = _last_ai_message(state.get("messages") or [])

        # print("\n " + "." * 90)
        print()
        print(green(f"  Turn {i} >" + "-" * 89))
        print(blue("\n  _input_ ") + "User: " + cyan(f"{repr(user_msg_text)}"))
        print(green("  " + "." * 97))

        locked_route = state.get("locked_route")
        if locked_route:
            print(
                blue("  _output_ ") + "state.locked_route: " + cyan(f"{locked_route}"))
        else:
            print(
                blue("  _output_ ") + "state.estimated_route: " + cyan(f"{state.get('estimated_route')}") + "\t confidence: " +
                cyan(f"{state.get('confidence')} ")
            )
        print(blue("  _output_ ") + "state.routing_attempts: " +
              cyan(f"{state.get('routing_attempts')}"))
        print(blue("  _output_ ") + "state.solve_attempts: " +
              cyan(f"{state.get('solve_attempts')}"))

        print(blue("  _output_ ") + "state.escalated_to_human: " +
              cyan(f"{state.get('escalated_to_human')}"))
        if last_ai:
            print(blue("\n  _output_ ") + "Assistant: " + cyan(last_ai))
        print()  # blank space

        exp = turn.get("expect")
        if not exp:
            continue

        if isinstance(exp, dict) and "one_of" in exp:
            options = exp["one_of"] or []
            if not any(_matches_expect(state, opt) for opt in options):
                raise AssertionError(
                    "No expectation in one_of matched\n"
                    f" - one_of: {options}\n"
                    f" - got: locked_route={state.get('locked_route')} estimated_route={state.get('estimated_route')} "
                    f" escalated_to_human={state.get('escalated_to_human')}"
                    f" routing_attempts={state.get('routing_attempts')} solve_attempts={state.get('solve_attempts')} "
                    f" confidence={state.get('confidence')} last_ai={repr(last_ai)}"
                )
        else:
            _assert_expect(state, exp)

        print("\n")  # extra space

    return state


def run_all(obj: Any, scenarios: List[Scenario], bundle_meta: Dict[str, Any]) -> None:
    failures = 0
    for s in scenarios:
        try:
            node_fn = _resolve_node_fn(obj, bundle_meta, s)
            run_scenario(node_fn, s, bundle_meta)
        except Exception as e:
            failures += 1
            print("\n  ❌ FAILED:", s.get("id"))
            print(e)

    print("\n" + "=" * 90)
    if failures:
        raise SystemExit(f"{failures} scenario(s) failed.")
    print("✅ All scenarios passed.")


# -----------------------------------------------------------------------------
# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run node scenarios from bundle JSON files (bundle-only).")
    ap.add_argument("bundle_files", nargs="+",
                    help="Path(s) to scenario bundle JSON files.")
    args = ap.parse_args()

    total_failures = 0

    for bundle_path in args.bundle_files:
        try:
            obj, scenarios, bundle_meta = load_scenarios_json(bundle_path)

            print("\n" + "#" * 90)
            print(f"Bundle: {bundle_path}")
            print(f"node_ref: {bundle_meta.get('node_ref')}")
            if bundle_meta.get("route_id") and bundle_meta.get("node_name"):
                print(
                    f"route_id/node_name: {bundle_meta.get('route_id')} / {bundle_meta.get('node_name')}")
            print(f"Scenarios: {len(scenarios)}")

            run_all(obj, scenarios, bundle_meta)
        except SystemExit as e:
            total_failures += 1
        except Exception as e:
            total_failures += 1
            print("\n❌ FAILED BUNDLE:", bundle_path)
            print(e)

    if total_failures:
        raise SystemExit(total_failures)


if __name__ == "__main__":
    main()
