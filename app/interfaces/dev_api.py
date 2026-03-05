import os

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

from langchain_core.messages import HumanMessage, AIMessage

from app.core.utils import get_routes
from app.core.persistence import get_sqlite_checkpointer
from app.core.graph.build import build_graph  # Changed to full graph like cli.py
from app.interfaces.chatbot_ui_mockup_helpers import (
    extract_assistant_text,
    make_config,
    render_page,
    validate_route,
)


"""
Dev API for quickly testing the full chatbot graph with a WhatsApp-style UI.

This mirrors cli.py's full chatbot behavior but with a web UI and reset functionality.
"""

load_dotenv()

TEST_KEY = os.getenv("TEST_KEY", "")
# Removed TEST_ROUTE as full graph handles routing dynamically

CHECKPOINTER = get_sqlite_checkpointer()

graph = build_graph()  # Changed to full graph like cli.py

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    # Generalized for full graph; update HTML template if needed to remove route-specific elements
    return HTMLResponse(render_page(__file__, "Full Chatbot"))


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

    config = make_config(thread_id)

    # Mimic cli.py: only send messages, no locked_route or attempts
    input_data = {
        "messages": [HumanMessage(content=text)],
    }

    output = graph.invoke(input_data, config=config)

    # Extract assistant text like cli.py: find AIMessages after the last HumanMessage
    msgs = output.get("messages") or []
    last_human_idx = next(
        (i for i in range(len(msgs) - 1, -1, -1)
         if isinstance(msgs[i], HumanMessage)),
        None
    )
    start = (last_human_idx + 1) if last_human_idx is not None else 0
    ai_after = [m for m in msgs[start:] if isinstance(m, AIMessage)]
    answer = "\n".join(m.content.strip()
                       for m in ai_after if m.content.strip()) or "(no assistant output)"

    return {"answer": answer}


@app.post("/reset")
async def reset(req: Request):
    payload = await req.json()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"
    # Reset for full graph: delete all checkpoints for this thread_id
    CHECKPOINTER.conn.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    CHECKPOINTER.conn.commit()
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
