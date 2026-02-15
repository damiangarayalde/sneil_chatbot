from dotenv import load_dotenv
from app.core.graph.build import build_graph
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
graph = build_graph()


def run_chatbot():
    # thread_id identifies a unique WhatsApp conversation
    user_id = input("Enter Phone Number (thread_id): ")
    config = {"configurable": {"thread_id": user_id}}

    print(f"--- Chat Session: {user_id} ---")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break

        # We only send the NEW message.
        # The Checkpointer pulls the previous history and phase from the DB automatically.
        input_data = {"messages": [HumanMessage(content=user_input)]}

        # We don't need to assign 'state = graph.invoke' because the DB persists it.
        # We just invoke to trigger the next turn.
        output = graph.invoke(input_data, config=config)

        # Print the last message from the assistant (only if it's actually an AIMessage)
        msgs = output.get("messages") or []

        # Find the index of the last HumanMessage
        last_human_idx = next(
            (i for i in range(len(msgs) - 1, -1, -1)
             if isinstance(msgs[i], HumanMessage)),
            None
        )

        # Print all assistant messages after that human message
        start = (last_human_idx + 1) if last_human_idx is not None else 0
        ai_after = [m for m in msgs[start:] if isinstance(m, AIMessage)]

        for j, m in enumerate(ai_after, start=1):
            content = (m.content or "").strip()
            if content:
                print(f"Assistant[{j}]: {content}")


if __name__ == "__main__":
    run_chatbot()
