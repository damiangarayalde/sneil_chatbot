"""
Smoke tests for route_subgraph behavior (REAL LLM + optional RAG).

Run:
  python scripts/route_subgraph__smoke_test.py

What this validates (based on your spec):
1) At the start of the route handler: if solve_attempts >= max_solve_attempts:
   - Send apology + recommend human support
   - Reset solve_attempts

2) If solve_attempts == 0:
   A) If user message is vague (no clear problem): ask a clarifying question
      - Do NOT call RAG
      - Do NOT increment solve_attempts
   B) If user message contains the problem: call RAG + answer + ask for confirmation
      - Increment solve_attempts (+1)
      - Asks for confirmation must be a separate assistant message

3) If 0 < solve_attempts < max_solve_attempts:
   - Always run (B)
   
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage

from app.core.graph.nodes.route_subgraph import make_route_subgraph
from app.core.prompts.builders import make_chat_prompt_for_route


# -----------------------------------------------------------------------------
# Helpers

def _get_max_solving_attempts(route_id: str) -> int:
    # Reuse the same config source as the subgraph
    _, route_cfg = make_chat_prompt_for_route(route_id)
    return int(
        route_cfg.get("max_attempts_before_handoff")
        or 0
    )


def _last_ai_messages(messages: list, n: int = 2) -> list[AIMessage]:
    ai = [m for m in (messages or []) if isinstance(m, AIMessage)]
    return ai[-n:]


def _print_case_header(title: str, route_id: str, user_text: str, state_overrides: dict):
    print("\n" + "=" * 90)
    print(title)
    print("- route:", route_id)
    print("- user:", repr(user_text))
    if state_overrides:
        print("- state_overrides:", state_overrides)


def run_case(
    title: str,
    route_id: str,
    user_text: str,
    *,
    expect: dict | None = None,
    **state_overrides,
):
    _print_case_header(title, route_id, user_text, state_overrides)

    g = make_route_subgraph(route_id)

    # Minimal state
    state = {
        "locked_route": route_id,
        "messages": [HumanMessage(content=user_text)],
        **state_overrides,
    }

    try:
        out_state = g.invoke(state)
    except Exception as e:
        print("!! ERROR invoking subgraph:", type(e).__name__, str(e))
        return

    attempts = out_state.get("attempts")
    escalated = out_state.get("escalated_to_human")
    locked = out_state.get("locked_route")
    retrieved = out_state.get("retrieved")

    msgs = out_state.get("messages") or []
    last_two_ai = _last_ai_messages(msgs, n=2)

    print("- locked_route(out):", locked)
    print("- escalated_to_human:", escalated)
    print("- attempts(out):", attempts)
    print("- retrieved_docs:", (len(retrieved) if retrieved else 0))
    if last_two_ai:
        print("- last assistant msg:", repr(last_two_ai[-1].content))
        if len(last_two_ai) > 1:
            print("- prev assistant msg:", repr(last_two_ai[-2].content))

    # Lightweight expectations (optional)
    if expect:
        def chk(cond: bool, label: str):
            if not cond:
                raise AssertionError(f"Expectation failed: {label}")

        if "attempts_eq" in expect:
            chk(attempts == expect["attempts_eq"],
                f"attempts == {expect['attempts_eq']} (got {attempts})")

        if "escalated_eq" in expect:
            chk(bool(escalated) == bool(expect["escalated_eq"]),
                f"escalated_to_human == {expect['escalated_eq']} (got {escalated})")

        if "retrieved_is_none" in expect:
            chk((retrieved is None) == bool(expect["retrieved_is_none"]),
                f"retrieved is None == {expect['retrieved_is_none']} (got {retrieved is None})")

        if "retrieved_min_len" in expect:
            rlen = len(retrieved) if retrieved else 0
            chk(rlen >= int(expect["retrieved_min_len"]),
                f"retrieved_docs >= {expect['retrieved_min_len']} (got {rlen})")

        if "min_ai_msgs" in expect:
            ai = [m for m in msgs if isinstance(m, AIMessage)]
            chk(len(ai) >= int(expect["min_ai_msgs"]),
                f"AI messages >= {expect['min_ai_msgs']} (got {len(ai)})")

        if "last_ai_contains" in expect:
            needle = str(expect["last_ai_contains"])
            last = last_two_ai[-1].content if last_two_ai else ""
            chk(needle.lower() in (last or "").lower(),
                f"last AI contains {needle!r}")

        print("✓ expectations passed")


# -----------------------------------------------------------------------------
# Main

def main():
    route_id = "TPMS"
    max_attempts = _get_max_solving_attempts(route_id)

    print("\n" + "#" * 90)
    print("ROUTE SUBGRAPH SMOKE TEST")
    print("- route:", route_id)
    print("- max_solving_attempts(from cfg):", max_attempts)
    print("#" * 90)

    if not max_attempts:
        print(
            "\n!! max_solving_attempts is 0/missing in config for this route.\n"
            "   Case (1) 'exceeded attempts' will NOT be meaningful until you set it.\n"
        )

    # 1) Exceeded attempts => apologize + handoff + reset attempts
    # Use attempts=max_attempts to hit threshold; if config is 0 this won't do anything.
    run_case(
        "1) attempts >= max -> apology + recommend human + reset attempts",
        route_id,
        "No me funciona y ya probé de todo.",
        attempts=max_attempts if max_attempts else 999,
        expect={
            # You can tighten these once your implementation is finalized:
            "escalated_eq": True,
            "attempts_eq": 0,
            "min_ai_msgs": 1,
        },
    )

    # 2A) First attempt, vague (no clear problem) => ask clarifying question, no RAG, no increment
    run_case(
        "2A) first attempt (0), vague -> clarifying question, no RAG, no increment",
        route_id,
        "Necesito ayuda con TPMS",
        attempts=0,
        # expected: attempts stays 0, retrieved should remain None, 1 AI question
        expect={
            "attempts_eq": 0,
            "retrieved_is_none": True,
            "min_ai_msgs": 1,
        },
    )

    # 2B) First attempt, problem stated => RAG + answer + separate confirmation, attempts +1
    run_case(
        "2B) first attempt (0), problem -> RAG + answer + confirmation (separate msg), attempts +1",
        route_id,
        "El TPMS no conecta el sensor y me marca error.",
        attempts=0,
        expect={
            "attempts_eq": 1,
            # If your RAG is configured, this should be >=1. If not, set to 0 or remove.
            "retrieved_min_len": 1,
            # Must have answer + confirmation => at least 2 AI messages
            "min_ai_msgs": 2,
            # Optional: once you fix the exact confirmation text, lock it in:
            # "last_ai_contains": "¿te sirvió"
        },
    )

    # 3) Not first attempt but below max => always B, attempts increments again
    run_case(
        "3) attempts=1 (<max) -> always B, attempts becomes 2",
        route_id,
        "Sigue sin conectar. Ya reinicié y cambié la pila del sensor.",
        attempts=1,
        expect={
            "attempts_eq": 2,
            "min_ai_msgs": 2,  # answer + confirmation
        },
    )


if __name__ == "__main__":
    main()
