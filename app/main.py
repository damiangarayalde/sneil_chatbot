from dotenv import load_dotenv
from app.graph import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()

graph = build_graph()


def run_chatbot():
    # Keep a single in-memory state across turns (REPL).
    # In production (WhatsApp/webhook), you would persist this by thread_id.
    state = {
        "messages": [],
        "phase": "triage",
        "next": None,
        "handling_channel": None,
        "confidence": None,
        "locked_route": None,
        "routing_attempts": 0,
        "triage_question": None,
        "triage_summary": None,
        "attempts": {},
        "retrieved": None,
        "answer": None,
    }

    while True:
        user_input = input("Message: ")
        if user_input.strip().lower() == "exit":
            print("Bye")
            break

        # Append the user's message to the conversation state
        state.setdefault("messages", []).append(
            HumanMessage(content=user_input))

        # Invoke the compiled graph to process the state and produce a response
        state = graph.invoke(state)

        # Display last assistant reply
        if state.get("messages"):
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
