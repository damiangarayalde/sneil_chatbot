from dotenv import load_dotenv
from app.graph import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()

graph = build_graph()


def run_chatbot():

    state = {
        "messages": [],  # start empty
        "handling_channel": None,
        # "product_family": None,
        "confidence": None,
        # "attempts": {},  # new unused field
        # "retrieved": None,  # new unused field
        # "answer": None,  # new unused field
    }

    # Main REPL loop: accept user input, run through the graph, and print replies
    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        # Append the user's message to the conversation state
        state["messages"] = state.get(
            "messages", []) + [HumanMessage(content=user_input)]

        # Invoke the compiled graph to process the state and produce a response
        state = graph.invoke(state)

        # print(state["answer"]) the new approach uses this as output msg instead of state["messages"][-1].content

        # If the graph returned messages, display the last assistant reply
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
