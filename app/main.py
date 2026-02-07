from dotenv import load_dotenv
from app.graph import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()

# Build graph once
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

        # Print the last message from the assistant
        if output.get("messages"):
            print(f"Assistant: {output['messages'][-1].content}")


if __name__ == "__main__":
    run_chatbot()
