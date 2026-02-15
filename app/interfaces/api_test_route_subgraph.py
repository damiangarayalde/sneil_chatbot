import os
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


load_dotenv()

# Optional shared secret; if set, clients must include {"key": "..."} in /chat payload.
TEST_KEY = os.getenv("TEST_KEY", "")

# Hardcoded: always run TPMS handler.
TEST_ROUTE = "TPMS"

ROUTES = set(get_routes())
if TEST_ROUTE not in ROUTES:
    raise RuntimeError(
        f"Invalid TEST_ROUTE={TEST_ROUTE!r}. Valid routes: {sorted(ROUTES)}"
    )

# Persistent (SQLite) checkpointer (cli.py semantics)
CHECKPOINTER = get_sqlite_checkpointer()


def build_route_only_graph(route_id: str, checkpointer: Optional[Any] = None):
    """Build a minimal graph that ALWAYS runs a single route subgraph.

    With a checkpointer + thread_id, this behaves like cli.py:
      - client sends only the NEW message
      - previous state/messages are restored automatically using thread_id
    """
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

# NOTE: This is an f-string (because of {TEST_ROUTE} below),
# so ALL literal braces in HTML/JS/CSS are escaped as {{}}.
PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dev Route Subgraph</title>
  <style>
    :root {{
      --bg: #000000;
      --panel: #1f1f1f;
      --input: #2a2a2a;
      --border: #3a3a3a;
      --text: #eaeaea;
      --muted: #bdbdbd;
      --btn: #333333;
      --btnHover: #444444;
      --danger: #5a2a2a;
      --dangerHover: #6a3333;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      margin: 0;
      padding: 0;
    }}

    .wrap {{
      max-width: 860px;
      margin: 0 auto;
      padding: 16px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    h2 {{
      margin: 8px 0 0;
      font-weight: 650;
      font-size: 20px;
    }}

    .muted {{
      color: var(--muted);
      margin: 0;
      font-size: 13px;
      line-height: 1.35;
    }}

    .topbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: flex-end;
    }}

    .field {{
      flex: 1 1 220px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 180px;
    }}

    label {{
      font-size: 12px;
      color: var(--muted);
    }}

    #thread {{
      background: var(--input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--text);
      outline: none;
      width: 100%;
    }}

    #thread::placeholder {{
      color: var(--muted);
    }}

    #log {{
      white-space: pre-wrap;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
      flex: 1 1 auto;
      min-height: 320px;
    }}

    #f {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}

    #m {{
      flex: 1 1 220px;
      background: var(--input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 12px;
      color: var(--text);
      outline: none;
      min-width: 180px;
    }}

    #m::placeholder {{
      color: var(--muted);
    }}

    button {{
      background: var(--btn);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      cursor: pointer;
      flex: 0 0 auto;
      width: auto;
    }}

    button:hover {{
      background: var(--btnHover);
    }}

    #resetBtn {{
      background: var(--danger);
    }}

    #resetBtn:hover {{
      background: var(--dangerHover);
    }}

    /* Mobile tweaks */
    @media (max-width: 520px) {{
      .wrap {{
        padding: 12px;
        gap: 10px;
      }}
      h2 {{
        font-size: 18px;
      }}
      #log {{
        padding: 12px;
        min-height: 260px;
      }}
      button {{
        width: 100%;
      }}
      #f {{
        gap: 8px;
      }}
    }}
  </style>
</head>

<body>
  <div class="wrap">
    <div>
      <h2>Route Subgraph Test — {TEST_ROUTE}</h2>
      <p class="muted">
        History-enabled (cli.py style): send only the NEW message; prior state is restored using <code>thread_id</code>.
      </p>
    </div>

    <div class="topbar">
      <div class="field">
        <label for="thread">thread_id (Phone Number)</label>
        <input id="thread" autocomplete="off" placeholder="e.g. +54911..." />
      </div>
      <div class="field" style="flex: 0 0 180px;">
        <label>&nbsp;</label>
        <button id="resetBtn" type="button">Reset chat</button>
      </div>
    </div>

    <div id="log"></div>

    <form id="f">
      <input id="m" autocomplete="off" placeholder="Type a message..." />
      <button id="sendBtn">Send</button>
    </form>
  </div>

<script>
const log = document.getElementById("log");
const f = document.getElementById("f");
const m = document.getElementById("m");
const thread = document.getElementById("thread");
const resetBtn = document.getElementById("resetBtn");

function addLine(who, text) {{
  log.textContent += "\\n" + who + ": " + text + "\\n";
  log.scrollTop = log.scrollHeight;
}}

function getThreadId() {{
  const v = (thread.value || "").trim();
  return v || "dev-thread";
}}

function persistThreadId() {{
  try {{
    localStorage.setItem("dev_thread_id", (thread.value || "").trim());
  }} catch (e) {{}}
}}

(function init() {{
  try {{
    const saved = localStorage.getItem("dev_thread_id");
    if (saved && !thread.value) thread.value = saved;
  }} catch (e) {{}}
  m.focus();
}})();

thread.addEventListener("change", persistThreadId);
thread.addEventListener("blur", persistThreadId);

f.onsubmit = async (e) => {{
  e.preventDefault();
  const text = (m.value || "").trim();
  if (!text) return;

  persistThreadId();

  addLine("You", text);
  m.value = "";
  m.focus();

  // Avoid JS object-literal braces as much as possible; still fine either way.
  const body = JSON.stringify(Object.fromEntries([
    ["text", text],
    ["thread_id", getThreadId()],
  ]));

  const r = await fetch("/chat", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body
  }});

  const j = await r.json();
  addLine("E-Neil", j.answer || JSON.stringify(j));
}};

resetBtn.onclick = async () => {{
  persistThreadId();

  const body = JSON.stringify(Object.fromEntries([
    ["thread_id", getThreadId()],
  ]));

  const r = await fetch("/reset", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body
  }});

  if (!r.ok) {{
    let j = {{}};
    try {{ j = await r.json(); }} catch (e) {{}}
    addLine("System", "Reset failed: " + (j.detail || r.status));
    return;
  }}

  log.textContent = "";
  addLine("System", "Chat reset (state + history cleared for this thread_id).");
  m.focus();
}};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


def extract_assistant_text(output: dict) -> str:
    """Return AI text emitted after the last HumanMessage (same logic as cli.py)."""
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
    """Reset state/history for a thread by overwriting a clean slate via the checkpointer."""
    config = _make_config(thread_id)

    # Keep it minimal and safe: only fields we know exist in this test.
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
            # Some versions require as_node=
            graph.update_state(config, clean_state,
                               as_node=START)  # type: ignore
            return

    # If update_state isn't available, there's no reliable cross-version way to purge.
    # We leave it as a no-op instead of risking DB corruption.
    return


@app.post("/chat")
async def chat(req: Request):
    payload = await req.json()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = (payload.get("text") or "").strip()
    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    config = _make_config(thread_id)

    # cli.py semantics: send only the NEW message; DB restores prior state automatically.
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
    uvicorn.run(
        "app.interfaces.api_test_route_subgraph:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
