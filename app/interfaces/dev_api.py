import os
import uuid
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

from langchain_core.messages import HumanMessage, AIMessage

from app.core.graph.build import build_graph

TEST_KEY = os.getenv("TEST_KEY", "")  # optional shared secret


# --- Same init pattern as cli.py ---
load_dotenv()
graph = build_graph()
# ----------------------------------


app = FastAPI()

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dev Chat</title>
  <style>
    :root {
      --bg: #000000;         /* app background */
      --panel: #1f1f1f;      /* chat window */
      --input: #2a2a2a;      /* free text input */
      --border: #3a3a3a;
      --text: #eaeaea;
      --muted: #bdbdbd;
      --btn: #333333;
      --btnHover: #444444;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      max-width: 820px;
      margin: 40px auto;
      padding: 0 16px;
    }

    h2 { margin: 0 0 14px; font-weight: 650; }

    #log {
      white-space: pre-wrap;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      height: 460px;
      overflow: auto;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
    }

    #f {
      margin-top: 12px;
      display: flex;
      gap: 10px;
    }

    #m {
      flex: 1;
      background: var(--input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 12px;
      color: var(--text);
      outline: none;
    }
    #m::placeholder { color: var(--muted); }

    button {
      background: var(--btn);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      cursor: pointer;
    }
    button:hover { background: var(--btnHover); }
  </style>
</head>

<body>
  <h2>E-Neil Test Chat</h2>

  <div id="log"></div>

  <form id="f">
    <input id="m" autocomplete="off" placeholder="Type a message..." />
    <button>Send</button>
  </form>

<script>
const log = document.getElementById("log");
const f = document.getElementById("f");
const m = document.getElementById("m");

let thread_id = localStorage.getItem("thread_id");
if (!thread_id) {
  thread_id = crypto.randomUUID();
  localStorage.setItem("thread_id", thread_id);
}

function addLine(who, text){
  log.textContent += `\\n${who}: ${text}\\n`;
  log.scrollTop = log.scrollHeight;
}

f.onsubmit = async (e) => {
  e.preventDefault();
  const text = m.value.trim();
  if (!text) return;

  addLine("You", text);
  m.value = "";

  const body = { thread_id, text };

  // If you set TEST_KEY on the server, uncomment:
  // body.key = "mysharedsecret";

  const r = await fetch("/chat", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });

  const j = await r.json();
  addLine("E-Neil", j.answer || JSON.stringify(j));
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


def extract_assistant_text(output: dict) -> str:
    msgs = output.get("messages") or []

    last_human_idx = next(
        (i for i in range(len(msgs) - 1, -1, -1)
         if isinstance(msgs[i], HumanMessage)),
        None
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
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    text = (payload.get("text") or "").strip()

    if TEST_KEY and payload.get("key") != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=text)]}
    output = graph.invoke(input_data, config=config)

    answer = extract_assistant_text(output) or "(no assistant output)"
    return {"thread_id": thread_id, "answer": answer}


def main():
    # Console entry point: sneil-chatbot-api
    uvicorn.run("app.interfaces.dev_api:app",
                host="127.0.0.1", port=8000, reload=True)
