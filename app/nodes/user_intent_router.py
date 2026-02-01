# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
from app.types import ChatState

# load_dotenv()

# Initialize the chat model used by the application
# llm = init_chat_model(model="gpt-4o-mini")


def node__route_by_user_intent(state: ChatState) -> ChatState:
    # Router node: returns the node-key string that identifies the chosen channel agent node (sales or support)

    pf = state.get("handling_channel") or "unknown"
    print(f'Routing based on handling_channel in state: {pf}')

    # Update the state with the chosen next node key so the conditional
    # edges selector can read it. Must return a dict (state update).
    chosen = f"handle__{pf}"  # if pf in subgraphs else "handle__unknown"
    return {"next": chosen}
