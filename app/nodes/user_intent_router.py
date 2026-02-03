from app.types import ChatState


def node__route_by_user_intent(state: ChatState) -> ChatState:
    """
    Router node: decides which handler node to run next.
    uses `handling_channel` in state to pick the route.
    If `locked_route` is set, uses that instead.
    """

    # Prefer locked route if present
    locked = state.get("locked_route")
    if locked:
        chosen = f"handle__{locked}"
        print(f"Routing using locked_route in state: {locked} -> {chosen}")
        return {"next": chosen}

    pf = state.get("handling_channel") or "unknown"
    print(f"Routing based on handling_channel in state: {pf}")

    chosen = f"handle__{pf}"
    return {"next": chosen}
