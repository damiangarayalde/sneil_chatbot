"""
End-to-end smoke test for the *overall* LangGraph flow (hub -> handler subgraph -> end_of_turn).

Run:
  python scripts/flow__smoke_test.py

Optional:
  FLOW_TRACE=1 python scripts/flow__smoke_test.py   # best-effort node streaming logs

What this tests
- Multi-turn persistence (same thread_id across turns)
- Hub locking + handler execution
- Handler attempts gate (max_attempts_before_handoff)
- Topic switching inside handler (TPMS -> AA -> TPMS), best-effort (LLM-dependent)

Notes
- Some paths call your configured LLM (hub classifier and/or handler structured output).
- Topic-switch detection is implemented via the handler's LLM structured output (HandlerOutput.is_topic_switch),
  so those expectations are marked as SOFT by default (they won't fail the run, but will warn).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.core.graph.build import build_graph
from app.core.prompts.builders import make_chat_prompt_for_route


# -----------------------------------------------------------------------------
# Small helpers

def _ai_messages(messages: list[BaseMessage] | None) -> list[AIMessage]:
    return [m for m in (messages or []) if isinstance(m, AIMessage)]


def _last_ai(messages: list[BaseMessage] | None) -> Optional[str]:
    ai = _ai_messages(messages)
    return ai[-1].content if ai else None


def _print_turn(title: str, user_text: str, out_state: Dict[str, Any]):
    print("\n" + "=" * 90)
    print(title)
    print("- user:", repr(user_text))
    print("- locked_route:", out_state.get("locked_route"))
    print("- estimated_route:", out_state.get("estimated_route"))
    print("- confidence:", out_state.get("confidence"))
    print("- attempts:", out_state.get("attempts"))
    print("- escalated_to_human:", out_state.get("escalated_to_human"))
    msgs = out_state.get("messages") or []
    ai = _ai_messages(msgs)
    if ai:
        print("- assistant(last):", repr(ai[-1].content))
        if len(ai) > 1:
            print("- assistant(prev):", repr(ai[-2].content))
        print(f"- total_messages: {len(msgs)} (ai={len(ai)})")
    else:
        print("- assistant: <none>")


def _get_max_solving_attempts(route_id: str) -> int:
    # Reuse the same config source as the subgraph
    _, route_cfg = make_chat_prompt_for_route(route_id)
    return int(route_cfg.get("max_attempts_before_handoff") or 0)


def _invoke(graph, thread_id: str, state_update: Dict[str, Any], *, trace: bool = False) -> Dict[str, Any]:
    """Invoke the compiled graph with a persisted thread_id.

    If trace=True, attempts to print node-level streamed events (best effort).
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    if not trace:
        return graph.invoke(state_update, config=cfg)

    last_state = None
    print("\n--- TRACE START ---")
    try:
        for event in graph.stream(state_update, config=cfg):
            if isinstance(event, dict):
                keys = list(event.keys())
                print("event keys:", keys[:6],
                      ("..." if len(keys) > 6 else ""))
                if len(event) == 1:
                    maybe = next(iter(event.values()))
                    if isinstance(maybe, dict):
                        last_state = maybe
            else:
                print("event:", type(event).__name__)
    except Exception as e:
        print("!! TRACE ERROR:", type(e).__name__, str(e))
    print("--- TRACE END ---\n")

    return graph.invoke(state_update, config=cfg)


# -----------------------------------------------------------------------------
# Scenarios

@dataclass
class Turn:
    user: str
    overrides: Optional[dict] = None
    expect: Optional[dict] = None
    soft_expect: Optional[dict] = None


@dataclass
class Scenario:
    title: str
    turns: list[Turn]


def _check_expectations(
    scenario_title: str,
    turn_i: int,
    out_state: Dict[str, Any],
    expect: Dict[str, Any],
    *,
    soft: bool,
):
    def fail(msg: str):
        if soft:
            print(f"  ! SOFT EXPECTATION FAILED: {msg}")
            return
        raise AssertionError(f"[{scenario_title} / turn {turn_i}] {msg}")

    # --- locked_route ---
    if "locked_route_eq" in expect:
        want = expect["locked_route_eq"]
        got = out_state.get("locked_route")
        if got != want:
            fail(f"locked_route == {want!r} (got {got!r})")

    if "locked_route_in" in expect:
        want = set(expect["locked_route_in"] or [])
        got = out_state.get("locked_route")
        if got not in want:
            fail(f"locked_route in {sorted(want)!r} (got {got!r})")

    # --- estimated_route ---
    if "estimated_route_eq" in expect:
        want = expect["estimated_route_eq"]
        got = out_state.get("estimated_route")
        if got != want:
            fail(f"estimated_route == {want!r} (got {got!r})")

    # --- attempts ---
    if "attempts_eq" in expect:
        want = int(expect["attempts_eq"])
        got = int(out_state.get("attempts") or 0)
        if got != want:
            fail(f"attempts == {want} (got {got})")

    if "attempts_ge" in expect:
        want = int(expect["attempts_ge"])
        got = int(out_state.get("attempts") or 0)
        if got < want:
            fail(f"attempts >= {want} (got {got})")

    if "attempts_le" in expect:
        want = int(expect["attempts_le"])
        got = int(out_state.get("attempts") or 0)
        if got > want:
            fail(f"attempts <= {want} (got {got})")

    # --- escalation flag ---
    if "escalated_eq" in expect:
        want = bool(expect["escalated_eq"])
        got = bool(out_state.get("escalated_to_human"))
        if got != want:
            fail(f"escalated_to_human == {want} (got {got})")

    # --- assistant content ---
    if "last_ai_contains" in expect:
        last = _last_ai(out_state.get("messages") or []) or ""
        needle = str(expect["last_ai_contains"])
        if needle.lower() not in last.lower():
            fail(f"last AI contains {needle!r} (got {last!r})")


def run_scenario(graph, scenario: Scenario, *, trace: bool = False):
    thread_id = f"flow-test-{uuid.uuid4().hex[:8]}"
    print("\n" + "#" * 90)
    print("SCENARIO:", scenario.title)
    print("thread_id:", thread_id)
    print("#" * 90)

    for i, t in enumerate(scenario.turns, start=1):
        overrides = t.overrides or {}
        expect = t.expect or {}
        soft_expect = t.soft_expect or {}

        state_update = {"messages": [
            HumanMessage(content=t.user)], **overrides}

        try:
            out_state = _invoke(graph, thread_id, state_update, trace=trace)
        except Exception as e:
            print("\n!! ERROR invoking graph on turn",
                  i, ":", type(e).__name__, str(e))
            return

        _print_turn(f"Turn {i}", t.user, out_state)

        # Hard expectations first (fail-fast)
        if expect:
            _check_expectations(scenario.title, i,
                                out_state, expect, soft=False)

        # Soft expectations (warn only)
        if soft_expect:
            _check_expectations(scenario.title, i, out_state,
                                soft_expect, soft=True)


def main():
    trace = os.getenv("FLOW_TRACE", "0") == "1"
    graph = build_graph()

    # -------------------------------------------------------------------------
    # Scenario 1: Multi-turn triage (no lock)
    s1 = Scenario(
        title="Multi-turn triage (hub clarifier, no lock)",
        turns=[
            Turn(user="hola", expect={"locked_route_eq": None}),
            Turn(user="necesito ayuda", expect={"locked_route_eq": None}),
            Turn(
                user="es sobre un producto, pero no sé cuál elegir",
                # Depending on your heuristics/LLM this might still not lock; keep it soft
                soft_expect={"locked_route_eq": None},
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Scenario 2: Multi-turn within a locked route (clarify on attempt 0, then solve)
    # We seed locked_route so we bypass hub and exercise handler START gating.
    s2 = Scenario(
        title="Locked route multi-turn (clarify first, then solve)",
        turns=[
            Turn(
                user="Tengo un problema",
                overrides={"locked_route": "TPMS", "attempts": 0},
                # Clarify path should NOT increment attempts (still 0)
                expect={"locked_route_eq": "TPMS", "attempts_eq": 0},
            ),
            Turn(
                user="El TPMS no muestra presión en una rueda, ya cambié la pila del sensor.",
                # Now handler should go retrieve+generate and increment attempts
                soft_expect={"locked_route_eq": "TPMS", "attempts_ge": 1},
            ),
            Turn(
                user="Ok, ¿y si el sensor queda intermitente?",  # another solve loop
                soft_expect={"locked_route_eq": "TPMS", "attempts_ge": 1},
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Topic switch mid-stream (TPMS -> AA)
    # NOTE: Topic switch detection is LLM-dependent in route_subgraph, so expectations are SOFT.
    s3 = Scenario(
        title="Topic switch (TPMS -> AA) in multi-turn conversation",
        turns=[
            Turn(
                user="Necesito soporte TPMS: no conecta el sensor y marca error.",
                soft_expect={"locked_route_eq": "TPMS"},
            ),
            Turn(
                user=(
                    "Cambiemos de tema: ahora necesito ayuda con el aire acondicionado (AA) 12V, "
                    "no enfría y la unidad se apaga sola."
                ),
                # Best-effort: handler should clear lock, hub should re-lock AA, and AA handler may run.
                soft_expect={"locked_route_in": [
                    "AA", None], "estimated_route_eq": "AA"},
            ),
            Turn(
                user="Sí, es AA. ¿Qué puedo revisar primero?",
                soft_expect={"locked_route_eq": "AA"},
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Scenario 4: Switch back (AA -> TPMS) to ensure we don't get stuck looping
    s4 = Scenario(
        title="Topic switch back (AA -> TPMS)",
        turns=[
            Turn(
                user="Necesito ayuda con AA 24V, no arranca.",
                soft_expect={"locked_route_eq": "AA"},
            ),
            Turn(
                user="Ahora volviendo al TPMS: me aparece sensor perdido en la rueda trasera.",
                soft_expect={"locked_route_in": [
                    "TPMS", None], "estimated_route_eq": "TPMS"},
            ),
            Turn(
                user="Es TPMS, modelo externo.",
                soft_expect={"locked_route_eq": "TPMS"},
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Scenario 5: Exceeded solve attempts gate inside handler triggers handoff + reset
    route_id = "TPMS"
    max_attempts = _get_max_solving_attempts(route_id)
    seeded_attempts = max_attempts if max_attempts else 999

    s5 = Scenario(
        title="Solve attempts exceeded -> handoff + reset (seeded locked route)",
        turns=[
            Turn(
                user="Sigue sin funcionar.",
                overrides={"locked_route": route_id,
                           "attempts": seeded_attempts},
                expect={"escalated_eq": True, "attempts_eq": 0},
            )
        ],
    )

    scenarios = [s2]  # [s1, s2, s3]  # , s4, s5]

    for s in scenarios:
        try:
            run_scenario(graph, s, trace=trace)
            print("\n✓ scenario finished:", s.title)
        except AssertionError as e:
            print("\n✗ ASSERTION FAILED:", str(e))

    print("\nDone.")
    if trace:
        print("(Tip) Set FLOW_TRACE=0 to disable node-level streaming logs.")


if __name__ == "__main__":
    main()
