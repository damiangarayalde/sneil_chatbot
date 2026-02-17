"""Scenario-driven smoke tests for hub routing rules (REAL LLM).

This script is intentionally lightweight: it uses a single Scenario Spec (v1) that
models *state + dialog flow* expectations.

Notes:
- This WILL call your configured LLM for scenarios that reach the classifier chain.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.core.graph.nodes.hub import node__classify_user_intent

# ======================================================================================
# Scenario Spec (v1) — state + dialog flow
#
# scenario:
#   id: str
#   title: str
#   tags: list[str] (optional)
#   setup:
#     initial_state: dict (optional)   # starting ChatState fields (attempts, locked_route, etc.)
#   turns: list[turn]
#
# turn:
#   user_msg: str
#   expect: expect | {"one_of": [expect, ...]} (optional)
#
# expect:
#   action: "ask_clarifier" | "lock_route" | "escalate" | "noop" (optional)
#   state: dict (optional)             # expected resulting state values (after applying node output)
#     locked_route: str|None
#     estimated_route: str|None
#     escalated_to_human: bool
#     attempts: int
#     confidence_min: float
#     confidence_max: float
#   assistant: dict (optional)         # very light dialog checks (avoid exact text matching)
#     contains: list[str]
#     ends_with_qmark: bool
#
# Key idea: assert BEHAVIOR + STATE, not exact phrasing.
# ======================================================================================

Scenario = Dict[str, Any]


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


def run_scenario(s: Scenario) -> Dict[str, Any]:
    state: Dict[str, Any] = {"messages": []}
    state.update((s.get("setup") or {}).get("initial_state") or {})

    print("\n" + "=" * 90)
    print(f"{s.get('id')}: {s.get('title')}")
    # tags = s.get("tags") or []
    # if tags:
    #     print("tags:", ", ".join(tags))
    # print("\n" + "." * 90)

    for k in sorted(state.keys()):
        print(f" _initial_ state.{k}: {state[k]}")

    for i, turn in enumerate(s.get("turns") or [], start=1):
        user_msg_text = turn["user_msg"]
        state.setdefault("messages", []).append(
            HumanMessage(content=user_msg_text))

        patch = node__classify_user_intent(state) or {}
        state = _apply_patch(state, patch)

        last_ai = _last_ai_message(state.get("messages") or [])

        print("\n " + "." * 90)
        print(f" Turn {i}:")
        print("\n _input__ User:", repr(user_msg_text))
        print("\n" + "." * 90)
        # print("Outputs:")
        locked_route = state.get("locked_route")

        if locked_route:
            print(" _output_ state.locked_route:", locked_route)
        else:
            print(
                f" _output_ state.estimated_route: {state.get("estimated_route")}, \t confidence: {state.get("confidence")} ")
        print(" _output_ state.attempts:", state.get("attempts"))
        print(" _output_ state.escalated_to_human:",
              state.get("escalated_to_human"))
        # print("\n " + "." * 90)
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

        print("\n \n")  # create extra space

    return state


SCENARIOS: List[Scenario] = [
    {
        "id": "Scenario_1",
        "title": "Low info greeting -> default clarifier (no LLM)",
        "tags": ["triage", "heuristic", "no_llm"],
        "setup": {"initial_state": {"attempts": 0}},
        "turns": [
            {
                "user_msg": "hola",
                "expect": {
                    "action": "ask_clarifier",
                    "state": {
                        "locked_route": None,
                        "attempts": 0,
                        "confidence_min": 0.0,
                        "confidence_max": 0.0,
                    },
                },
            }
        ],
    },
    {
        "id": "Scenario_2",
        "title": "Explicit human request -> escalation (no LLM)",
        "tags": ["escalation", "heuristic", "no_llm"],
        "setup": {"initial_state": {"attempts": 0}},
        "turns": [
            {
                "user_msg": "quiero hablar con un humano",
                "expect": {
                    "action": "escalate",
                    "state": {"locked_route": None, "escalated_to_human": True, "attempts": 0},
                },
            }
        ],
    },
    {
        "id": "Scenario_3",
        "title": "Support + route mention -> direct lock (no LLM)",
        "tags": ["routing", "heuristic", "no_llm", "tpms"],
        "setup": {"initial_state": {"attempts": 0}},
        "turns": [
            {
                "user_msg": "soporte tpms, no conecta el sensor",
                "expect": {
                    "action": "lock_route",
                    "state": {"locked_route": "TPMS", "estimated_route": "TPMS", "attempts": 0},
                },
            }
        ],
    },
    {
        "id": "Scenario_4",
        "title": "Attempts >= max_attempts_before_handoff -> escalation (no LLM)",
        "tags": ["escalation", "heuristic", "no_llm"],
        # NOTE: assumes your hub config sets max_attempts_before_handoff <= 3
        "setup": {"initial_state": {"attempts": 3}},
        "turns": [
            {
                "user_msg": "ok",
                "expect": {
                    "action": "escalate",
                    "state": {"escalated_to_human": True, "attempts": 0},
                },
            }
        ],
    },
    {
        "id": "Scenario_5",
        "title": "LLM needed: Should route to AA ",
        "tags": ["llm", "aa"],
        "setup": {"initial_state": {"attempts": 0}},
        "turns": [
            {
                "user_msg": "Mi aire acondicionado no enfría y hace ruido.",
                "expect": {
                    "one_of": [
                        {"action": "lock_route", "state": {"locked_route": "AA"}},
                        {"action": "ask_clarifier", "state": {
                            "locked_route": None, "estimated_route": "AA"}},
                    ]
                },
            }
        ],
    },
    {
        "id": "Scenario_6",
        "title": "LLM needed: ambiguous message (should ask a clarifier, not lock)",
        "tags": ["llm", "ambiguous"],
        "setup": {"initial_state": {"attempts": 0}},
        "turns": [
            {
                "user_msg": "Tengo una consulta sobre un producto.",
                "expect": {
                    "action": "ask_clarifier",
                    "state": {"locked_route": None, "attempts": 1},
                },
            }
        ],
    },
]


def main() -> None:
    failures = 0
    for s in SCENARIOS:
        try:
            run_scenario(s)
        except Exception as e:
            failures += 1
            print("\n❌ FAILED:", s.get("id"))
            print(e)

    print("\n" + "=" * 90)
    if failures:
        raise SystemExit(f"{failures} scenario(s) failed.")
    print("✅ All scenarios passed.")


if __name__ == "__main__":
    main()
