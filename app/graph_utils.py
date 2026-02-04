from __future__ import annotations

from typing import Callable
from app.types import ChatState


def wrap_node(name: str, fn: Callable[[ChatState], ChatState]) -> Callable[[ChatState], ChatState]:
    """Wrap a node function (or compiled subgraph callable) to print concise debug logs.

    Prints: node name + small state summary on entry, and the returned partial update on exit.
    Keep this lightweight so it can stay enabled during iterative development.
    """

    def _fmt_state_summary(s: ChatState) -> str:
        return (
            f"phase={s.get('phase')!r} next={s.get('next')!r} locked_route={s.get('locked_route')!r} "
            f"routing_attempts={s.get('routing_attempts')!r} handling_channel={s.get('handling_channel')!r} "
            f"confidence={s.get('confidence')!r}"
        )

    def wrapper(state: ChatState) -> ChatState:
        try:
            print(f"[GRAPH] Enter node: {name} | {_fmt_state_summary(state)}")
        except Exception:
            print(f"[GRAPH] Enter node: {name}")

        # Support both callables and compiled graphs exposing `.invoke`
        try:
            if hasattr(fn, "invoke"):
                result = fn.invoke(state)
            else:
                result = fn(state)
        except TypeError:
            if hasattr(fn, "invoke"):
                result = fn.invoke(state)
            else:
                raise

        try:
            print(f"[GRAPH] Exit  node: {name} -> {result}")
        except Exception:
            print(f"[GRAPH] Exit  node: {name}")

        return result

    return wrapper
