import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from app.core.graph.state import ChatState
from app.core.graph.nodes.route_subgraph import make_route_subgraph
from app.core.graph.flow_logging import wrap_node
from app.core.utils import get_routes
from app.core.persistence import get_sqlite_checkpointer


"""
Dev API for quickly testing a single route-subgraph with a WhatsApp-style UI.

This file mirrors the approach in api_test_route_subgraph.py, but uses an
outsourced HTML file (chatbot_ui_mockup.html) that is loaded from disk at runtime.
"""

load_dotenv()

TEST_KEY = os.getenv("TEST_KEY", "")
TEST_ROUTE = os.getenv("TEST_ROUTE", "TPMS").strip() or "TPMS"

ROUTES = set(get_routes())
if TEST_ROUTE not in ROUTES:
    raise RuntimeError(
        f"Invalid TEST_ROUTE={TEST_ROUTE!r}. Valid routes: {sorted(ROUTES)}")

CHECKPOINTER = get_sqlite_checkpointer()


def build_route_only_graph(route_id: str, checkpointer: Optional[Any] = None):
    subgraph = make_route_subgraph(route_id)

    g = StateGraph(ChatState)
    node_name = f"handle__{route_id}"
    g.add_node(node_name, wrap_node(node_name, subgraph))

    g.add_edge(START, node_name)
    g.add_edge(node_name, END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


graph = build_route_only_graph(TEST_ROUTE, checkpointer=CHECKPOINTER)

app = FastAPI()


# ---- HTML loading (outsourced UI) ----
_THIS_DIR = Path(__file__).resolve().parent
_HTML_PATH = _THIS_DIR / "chatbot_ui_mockup.html"


def render_page() -> str:
    """
    Load and return the HTML UI.
    Uses simple placeholder replacement (no f-strings, no brace escaping issues).
    """
    if not _HTML_PATH.exists():
        return (
            "<h2>Missing UI file</h2>"
            f"<p>Expected to find: <code>{_HTML_PATH}</code></p>"
            "<p>Copy <code>chatbot_ui_mockup.html</code> next to <code>dev_api.py</code>.</p>"
        )
    html = _HTML_PATH.read_text(encoding="utf-8")
    return html.replace("{{TEST_ROUTE}}", TEST_ROUTE)
# -------------------------------------


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render_page())


def extract_assistant_text(output: dict) -> str:
    """
    Collect all AI messages produced after the latest HumanMessage in the
    returned state (helps when the graph emits multiple AI messages per turn).
    """
    msgs = output.get("messages") or []

    last_human_idx = next(
        (i for i in range(len(msgs) - 1, -1, -1)
         if isinstance(msgs[i], HumanMessage)),
        None,
    )
    start = (last_human_idx + 1) if last_human_idx is not None else 0
    ai_after = [m for m in msgs[start:] if isinstance(m, AIMessage)]

    parts = []
    for m in ai_after:
        content = (m.content or "").strip()
        if content:
            parts.append(content)

    return "\n\n".join(parts).strip()


def _make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _reset_thread_state(thread_id: str) -> None:
    """
    Clears persisted state for this thread (messages + attempts) so the UI's
    /reset button is a true "fresh chat" for the same thread_id.
    """
    config = _make_config(thread_id)
    clean_state = {
        "messages": [],
        "locked_route": TEST_ROUTE,
        "attempts": 0,
    }

    if hasattr(graph, "update_state"):
        try:
            graph.update_state(config, clean_state)  # type: ignore
            return
        except TypeError:
            # Some langgraph versions require as_node=START
            graph.update_state(config, clean_state,
                               as_node=START)  # type: ignore
            return


@app.post("/chat")
async def chat(req: Request):
    payload = await req.json()

    # Optional shared secret for local exposure (ngrok, etc.)
    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = (payload.get("text") or "").strip()
    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    config = _make_config(thread_id)

    # This dev UI is meant to test ONE route, so we keep locked_route fixed.
    input_data = {
        "locked_route": TEST_ROUTE,
        "attempts": 1,
        "messages": [HumanMessage(content=text)],
    }

    output = graph.invoke(input_data, config=config)
    answer = extract_assistant_text(output) or "(no assistant output)"
    return {"answer": answer}


@app.post("/reset")
async def reset(req: Request):
    payload = await req.json()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"
    _reset_thread_state(thread_id)
    return {"ok": True}


def main():
    # Update module path if this file lives elsewhere in your repo.
    uvicorn.run(
        "app.interfaces.dev_api:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )


if __name__ == "__main__":
    main()
