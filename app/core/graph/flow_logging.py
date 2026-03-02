from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import BaseMessage
from app.core.graph.state import ChatState

# Routing helpers (to predict the "next" node for clearer logs).
# NOTE: These are pure functions and safe to call here.
from app.core.graph.nodes import end_turn_node_name

# ---- Logging config ----
_MAX_STR = 110   # max chars shown for any string-ish value
_MAX_KVS = 7     # max extra key/values shown per line


def _ansi_enabled() -> bool:
    # Respect NO_COLOR (https://no-color.org/)
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


ANSI = _ansi_enabled()
RESET = "\033[0m" if ANSI else ""
BOLD = "\033[1m" if ANSI else ""
DIM = "\033[90m" if ANSI else ""  # grey/dim


def _style_current(label: str) -> str:
    # Current step in bold (or uppercase if no ANSI)
    if ANSI:
        return f"{BOLD}{label}{RESET}"
    return label.upper()


def _style_inactive(label: str) -> str:
    # Non-current steps greyed out (or plain if no ANSI)
    if ANSI:
        return f"{DIM}{label}{RESET}"
    return label


def _non_empty(v: Any) -> bool:
    return not (v is None or v == "" or v == [] or v == {})


def _handle_label(s: Dict[str, Any]) -> str:
    locked = s.get("locked_route")
    if _non_empty(locked):
        return f"handle__{locked}"
    return "handle__{route}?"


def _render_flow_bar(current_node: str, merged_state: Dict[str, Any]) -> str:
    """Render the current architecture flow:

    START -> classifier -> handle__{route}? -> finalize_turn -> END

    - START/END are always inactive (dim), since they are not executed as nodes.
    - The handler slot is filled with handle__<locked_route> once the classifier locks.
    """
    classifier_node = "classifier"
    classifier_label = "triage"

    finalize_name = end_turn_node_name()
    handler = _handle_label(merged_state)
    if current_node.startswith("handle__"):
        handler = current_node

    labels = ["START", classifier_node, handler, finalize_name, "END"]

    if current_node == classifier_node:
        current_label = classifier_node
    elif current_node.startswith("handle__"):
        current_label = handler
    elif current_node == finalize_name:
        current_label = finalize_name
    else:
        current_label = current_node

    styled: List[str] = []
    for lab in labels:
        if lab in ("START", "END"):
            styled.append(_style_inactive(lab))
        elif lab == current_label:
            styled.append(_style_current(lab))
        else:
            styled.append(_style_inactive(lab))

    return " -> ".join(styled)


def _delta_normalize_messages(input_state: ChatState, result: Dict[str, Any]) -> Dict[str, Any]:
    """If a runnable/subgraph returns a FULL messages list, convert it to a delta list.

    Some compiled subgraphs return the entire state; we want logs (and downstream reducers)
    to behave as "append-only" by returning just the new messages.
    """
    if "messages" not in result:
        return result

    in_msgs = input_state.get("messages") or []
    out_msgs = result.get("messages") or []

    if not isinstance(in_msgs, list) or not isinstance(out_msgs, list):
        return result
    if not out_msgs:
        return result
    if not all(isinstance(m, BaseMessage) for m in out_msgs):
        return result

    # Heuristic: if the output starts with the input history, treat it as full-state return
    if len(out_msgs) >= len(in_msgs):
        try:
            prefix_matches = True
            for i in range(len(in_msgs)):
                if out_msgs[i] is not in_msgs[i] and repr(out_msgs[i]) != repr(in_msgs[i]):
                    prefix_matches = False
                    break
            if prefix_matches:
                updated = dict(result)
                updated["messages"] = out_msgs[len(in_msgs):]
                return updated
        except Exception:
            return result

    return result


def wrap_node(name: str, fn: Callable[[ChatState], ChatState]) -> Callable[[ChatState], ChatState]:
    """Wrap a node (or compiled subgraph) and print a one-line flow log per execution."""

    def wrapper(state: ChatState) -> ChatState:
        # Snapshot "before"
        in_locked = state.get("locked_route")

        # Execute
        if hasattr(fn, "invoke"):
            result = fn.invoke(state)  # type: ignore[attr-defined]
        else:
            result = fn(state)

        # If this runnable/subgraph returned full state, keep only delta messages
        if isinstance(result, dict):
            result = _delta_normalize_messages(state, result)

        # Merge state+result for display and prediction
        merged: Dict[str, Any] = dict(state)
        if isinstance(result, dict):
            merged.update(result)

        flow_bar = _render_flow_bar(name, merged)

        # Tail: signal-heavy routing
        tail_parts: List[str] = []

        out_locked = merged.get("locked_route")

        if _non_empty(out_locked):
            tail_parts.append(f"Locked_route={out_locked}")

        conf = merged.get("confidence")
        if isinstance(conf, (float, int)):
            tail_parts.append(f"confidence={float(conf):.3f}")

        ra = merged.get("attempts")
        if isinstance(ra, int) and ra != 0:
            tail_parts.append(f"attempts={ra}")

        tail = ("  " + " ".join(tail_parts[:_MAX_KVS])) if tail_parts else ""
        print(f"[ {flow_bar} ] {tail}")

        return result

    return wrapper
