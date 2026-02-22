"""
Single runner for node scenarios.

Each scenario JSON file is an object with BOTH:
- "node_ref": "module.path:callable_name"
- "scenarios": [ ... ]   (your Scenario Spec v1 list)

Example:
{
  "node_ref": "app.core.graph.nodes.hub:node__classify_user_intent",
  "scenarios": [
    { "id": "Scenario_1", "title": "...", "setup": {...}, "turns": [...] }
  ]
}

Run:
  python utility_to_run_scenarios.py path/to/bundle.json
  python utility_to_run_scenarios.py bundle1.json bundle2.json

This stays intentionally small: node-only (no graphs), no pytest plugins, no extra abstractions.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

Scenario = Dict[str, Any]
NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]


# -----------------------------------------------------------------------------
# Loading + node resolving (BUNDLE ONLY)


def _import_from_ref(ref: str) -> Any:
    """Import 'module:attr' (attr can be dotted)."""
    if ":" not in ref:
        raise ValueError(
            f"Invalid node_ref '{ref}'. Expected format: module.path:callable_name"
        )
    mod_name, attr_path = ref.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def load_scenarios_json(path: str | Path) -> Tuple[NodeFn, List[Scenario]]:
    """Load a scenario bundle and resolve its node function.

    REQUIRED JSON schema:
      { "node_ref": str, "scenarios": list[Scenario] }
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"Scenario bundle must be a JSON object with keys 'node_ref' and 'scenarios'. Got: {type(data)}"
        )

    node_ref = data.get("node_ref")
    scenarios = data.get("scenarios")

    if not isinstance(node_ref, str) or not node_ref.strip():
        raise ValueError(
            "'node_ref' is required and must be a non-empty string.")
    if not isinstance(scenarios, list):
        raise ValueError("'scenarios' is required and must be a list.")

    fn = _import_from_ref(node_ref)
    if not callable(fn):
        raise TypeError(f"Resolved node_ref is not callable: {node_ref}")

    return fn, scenarios  # type: ignore[return-value]


# python utility_to_run_scenarios_bundle_only.py path/to/your_bundle.json
# -----------------------------------------------------------------------------
# Runner (kept almost identical to your current engine)


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
            "Expectation failed\n"
            f"- expected: {expect}\n"
            f"- got locked_route={state.get('locked_route')} estimated_route={state.get('estimated_route')} "
            f"escalated_to_human={state.get('escalated_to_human')} attempts={state.get('attempts')} "
            f"confidence={state.get('confidence')}\n"
            f"- last_ai={repr(last_ai)}"
        )


def run_scenario(node_fn: NodeFn, s: Scenario) -> Dict[str, Any]:
    state: Dict[str, Any] = {"messages": []}
    state.update((s.get("setup") or {}).get("initial_state") or {})

    print("\n" + "=" * 90)
    print(f"{s.get('id')}: {s.get('title')}")
    print()
    for k in sorted(state.keys()):
        print(f" _initial_ state.{k}: {state[k]}")

    for i, turn in enumerate(s.get("turns") or [], start=1):
        user_msg_text = turn["user_msg"]
        state.setdefault("messages", []).append(
            HumanMessage(content=user_msg_text))

        patch = node_fn(state) or {}
        state = _apply_patch(state, patch)

        last_ai = _last_ai_message(state.get("messages") or [])

        print("\n " + "." * 90)
        print(f" Turn {i}:")
        print("\n _input__ User:", repr(user_msg_text))
        print("\n" + "." * 90)

        locked_route = state.get("locked_route")
        if locked_route:
            print(" _output_ state.locked_route:", locked_route)
        else:
            print(
                f" _output_ state.estimated_route: {state.get('estimated_route')}, \t confidence: {state.get('confidence')} "
            )
        print(" _output_ state.attempts:", state.get("attempts"))
        print(" _output_ state.escalated_to_human:",
              state.get("escalated_to_human"))
        if last_ai:
            print("\n _output_ Assistant:", last_ai)

        exp = turn.get("expect")
        if not exp:
            continue

        if isinstance(exp, dict) and "one_of" in exp:
            options = exp["one_of"] or []
            if not any(_matches_expect(state, opt) for opt in options):
                raise AssertionError(
                    "No expectation in one_of matched\n"
                    f"- one_of: {options}\n"
                    f"- got: locked_route={state.get('locked_route')} estimated_route={state.get('estimated_route')} "
                    f"escalated_to_human={state.get('escalated_to_human')} attempts={state.get('attempts')} "
                    f"confidence={state.get('confidence')} last_ai={repr(last_ai)}"
                )
        else:
            _assert_expect(state, exp)

        print("\n \n")  # extra space

    return state


def run_all(node_fn: NodeFn, scenarios: List[Scenario]) -> None:
    failures = 0
    for s in scenarios:
        try:
            run_scenario(node_fn, s)
        except Exception as e:
            failures += 1
            print("\n❌ FAILED:", s.get("id"))
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
            node_fn, scenarios = load_scenarios_json(bundle_path)
            print("\n" + "#" * 90)
            print(f"Bundle: {bundle_path}")
            print(f"Scenarios: {len(scenarios)}")
            run_all(node_fn, scenarios)
        except SystemExit as e:
            # run_all uses SystemExit for failures
            total_failures += 1
        except Exception as e:
            total_failures += 1
            print("\n❌ FAILED BUNDLE:", bundle_path)
            print(e)

    if total_failures:
        raise SystemExit(total_failures)


if __name__ == "__main__":
    main()
