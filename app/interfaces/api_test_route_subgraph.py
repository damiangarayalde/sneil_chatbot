import os

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


def build_route_only_graph(route_id: str):
    """Build a minimal graph that ALWAYS runs a single route subgraph.

    This bypasses the hub/high-level routing graph. It is intentionally stateless:
    each request starts from a fresh state (no thread/session tracking).
    """

    subgraph = make_route_subgraph(route_id)

    g = StateGraph(ChatState)
    node_name = f"handle__{route_id}"  # keep the same naming convention
    g.add_node(node_name, wrap_node(node_name, subgraph))

    g.add_edge(START, node_name)
    g.add_edge(node_name, END)

    # No checkpointer: every call starts new
    return g.compile()


graph = build_route_only_graph(TEST_ROUTE)


app = FastAPI()

PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
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
    }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, \"Apple Color Emoji\",\"Segoe UI Emoji\";
      max-width: 820px;
      margin: 40px auto;
      padding: 0 16px;
    }}

    h2 {{ margin: 0 0 8px; font-weight: 650; }}
    .muted {{ color: var(--muted); margin: 0 0 14px; font-size: 14px; }}

    #log {{
      white-space: pre-wrap;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      height: 460px;
      overflow: auto;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
    }}

    #f {{
      margin-top: 12px;
      display: flex;
      gap: 10px;
    }}

    #m {{
      flex: 1;
      background: var(--input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 12px;
      color: var(--text);
      outline: none;
    }}
    #m::placeholder {{ color: var(--muted); }}

    button {{
      background: var(--btn);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      cursor: pointer;
    }}
    button:hover {{ background: var(--btnHover); }}
  </style>
</head>

<body>
  <h2>Route Subgraph Test — {TEST_ROUTE}</h2>
  <p class=\"muted\">Stateless: this bypasses the hub and runs only the TPMS route handler subgraph (retrieve → generate). Each call starts from a new state.</p>

  <div id=\"log\"></div>

  <form id=\"f\">
    <input id=\"m\" autocomplete=\"off\" placeholder=\"Type a message...\" />
    <button>Send</button>
  </form>

<script>
const log = document.getElementById(\"log\");
const f = document.getElementById(\"f\");
const m = document.getElementById(\"m\");

function addLine(who, text){{
  log.textContent += `\n${{who}}: ${{text}}\n`;
  log.scrollTop = log.scrollHeight;
}}

f.onsubmit = async (e) => {{
  e.preventDefault();
  const text = m.value.trim();
  if (!text) return;

  addLine(\"You\", text);
  m.value = \"\";

  const body = {{ text }};

  // If you set TEST_KEY on the server, uncomment:
  // body.key = \"mysharedsecret\";

  const r = await fetch(\"/chat\", {{
    method:\"POST\",
    headers:{{\"Content-Type\":\"application/json\"}},
    body: JSON.stringify(body)
  }});

  const j = await r.json();
  addLine(\"E-Neil\", j.answer || JSON.stringify(j));
}};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


def extract_assistant_text(output: dict) -> str:
    """Return AI text emitted after the last HumanMessage."""
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


@app.post("/chat")
async def chat(req: Request):
    payload = await req.json()
    text = (payload.get("text") or "").strip()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    # Force the route lock for this handler-only test API.
    # This mirrors the minimal state used in test_route_subgraph.py.
    input_data = {
        "locked_route": TEST_ROUTE,
        "messages": [HumanMessage(content=text)],
    }

    # No session/thread tracking: each request is a fresh run.
    output = graph.invoke(input_data)

    answer = extract_assistant_text(output) or "(no assistant output)"
    return {"answer": answer}


def main():
    # Console entry point: sneil-chatbot-api-test-route
    uvicorn.run(
        "app.interfaces.api_test_route_subgraph:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
