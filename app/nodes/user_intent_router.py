from app.types import ChatState


def node__route_by_user_intent(state: ChatState) -> ChatState:
    """Router node: chooses the handler node key.

    Expected upstream behavior:
    - `node__classify_user_intent` sets `locked_route` when ready to proceed.

    If `locked_route` is missing, we bounce back to triage to recover.
    """
    locked = state.get("locked_route")
    if not locked:
        print("---> Inside: node__route_by_user_intent .....missing locked_route -> returning to triage\n")
        return {"next": "triage"}

    chosen = f"handle__{locked}"
    print(
        f"---> Inside: node__route_by_user_intent ..... locked_route={locked} -> {chosen}\n")
    return {"next": chosen}
