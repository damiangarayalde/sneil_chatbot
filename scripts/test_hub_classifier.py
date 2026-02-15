"""Smoke tests for hub routing rules (REAL LLM).

Run:
  python scripts/test_hub_rules.py

Notes:
- This will call your configured LLM for the cases that require classification.
- Make sure your environment is configured (e.g., OPENAI_API_KEY or whatever `init_llm` expects).
"""

from langchain_core.messages import HumanMessage

from app.core.graph.nodes.hub import node__classify_user_intent


def run_case(title: str, user_text: str, **state_overrides):
    state = {"messages": [HumanMessage(content=user_text)], **state_overrides}
    out = node__classify_user_intent(state) or {}

    locked = out.get("locked_route")
    escalated = out.get("escalated_to_human")
    attempts = out.get("attempts")
    msgs = out.get("messages") or []
    last = msgs[-1].content if msgs else None

    print("\n" + "=" * 90)
    print(title)
    print("- user:", repr(user_text))
    print("- locked_route:", locked)
    print("- escalated_to_human:", escalated)
    print("- routing_attempts:", attempts)
    if last:
        print("- assistant:", last)


def main():
    # deterministic branches (NO LLM)
    run_case("1) Low info greeting -> default clarifier", "hola")
    run_case("2) Explicit human request -> escalation",
             "quiero hablar con un humano")
    run_case("3) support/sale + single route mention -> direct lock",
             "soporte tpms, no conecta el sensor")
    run_case("4) attempts >= MAX_ROUTING_ATTEMPTS -> escalation",
             "ok", attempts=3)

    # these should hit the LLM
    run_case("5) LLM needed: normal AA issue (should lock AA)",
             "Mi aire acondicionado no enfría y hace ruido.")
    run_case("6) LLM needed: ambiguous message (should ask clarifier)",
             "Tengo una consulta sobre un producto.")


if __name__ == "__main__":
    main()
