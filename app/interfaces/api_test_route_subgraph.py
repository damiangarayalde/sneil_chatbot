import os

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

from langchain_core.messages import HumanMessage

from app.core.persistence import get_sqlite_checkpointer
from app.core.utils import get_routes
from app.interfaces.chatbot_ui_mockup_helpers import (
    build_route_only_graph,
    extract_assistant_text,
    make_config,
    render_page,
    reset_thread_state,
    validate_route,
)


load_dotenv()

TEST_KEY = os.getenv("TEST_KEY", "")
TEST_ROUTE = "TPMS"

ROUTES = set(get_routes())
TEST_ROUTE = validate_route(TEST_ROUTE, ROUTES)

CHECKPOINTER = get_sqlite_checkpointer()

graph = build_route_only_graph(TEST_ROUTE, checkpointer=CHECKPOINTER)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render_page(__file__, TEST_ROUTE))


@app.post("/chat")
async def chat(req: Request):
    payload = await req.json()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = (payload.get("text") or "").strip()
    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    config = make_config(thread_id)

    input_data = {
        "locked_route": TEST_ROUTE,
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
    reset_thread_state(graph, thread_id, locked_route=TEST_ROUTE, attempts=0)
    return {"ok": True}


def main():
    uvicorn.run(
        "app.interfaces.api_test_route_subgraph:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
