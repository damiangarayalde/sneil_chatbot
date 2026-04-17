"""
WhatsApp webhook adapter — FastAPI APIRouter.

Receives incoming message events from the Baileys bridge (or, in the future,
directly from Meta's WhatsApp Business API) and runs them through the same
LangGraph pipeline used by dev_api.py.

Current payload (Baileys bridge):
    POST /whatsapp/incoming
    { "from": "5491155551234", "text": "hola", "timestamp": 1234567890 }
    → { "response": "bot reply text" }

Future payload (Meta Business API — swap in when ready):
    POST /whatsapp/incoming
    { "object": "whatsapp_business_account", "entry": [ { "changes": [ {
        "value": {
            "messages": [ { "from": "...", "text": { "body": "..." }, "timestamp": "..." } ],
            "metadata": { "phone_number_id": "..." }
        }
    } ] } ] }

    GET /whatsapp/incoming   ← Meta token-verification handshake
    ?hub.mode=subscribe&hub.challenge=123&hub.verify_token=<WA_VERIFY_TOKEN>

Thread ID convention: "wa_{phone_number}"
    e.g. phone "5491155551234" → thread_id "wa_5491155551234"
    Consistent with the Meta format (same field name: "from").
    Deterministic and persistent across WA Web reconnects and API migration.

Auth: X-Bridge-Secret header must match WA_BRIDGE_SECRET env var (if set).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from langchain_core.messages import HumanMessage

from app.core.logging_config import get_logger, set_request_id
from app.interfaces.chatbot_ui_mockup_helpers import extract_assistant_text, make_config

_logger = get_logger("sneil.whatsapp")

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

# Injected by dev_api.py during lifespan startup (see set_graph).
_graph: Any = None

def set_graph(graph: Any) -> None:
    """Called by dev_api.py once the async graph is ready."""
    global _graph
    _graph = graph


def _check_auth(secret: str | None) -> None:
    """Reject requests whose X-Bridge-Secret doesn't match WA_BRIDGE_SECRET.

    Read lazily so load_dotenv() in dev_api.py has already run by request time.
    """
    expected = os.getenv("WA_BRIDGE_SECRET", "")
    if expected and secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Baileys bridge endpoint ───────────────────────────────────────────────────

@router.post("/incoming")
async def incoming(
    req: Request,
    x_bridge_secret: str | None = Header(default=None),
):
    """
    Receive a message from the Baileys bridge, run it through the graph,
    and return the bot's reply synchronously.

    The Baileys bridge POSTs here and waits (up to 65s) for the response,
    then sends it back to the WhatsApp user.

    ── Meta migration note ──────────────────────────────────────────────────
    When Meta webhooks replace the Baileys bridge:
      1. Parse Meta's nested payload instead (see module docstring above).
      2. Add the GET /incoming endpoint below for the verification handshake.
      3. The graph invocation block below stays IDENTICAL.
    ─────────────────────────────────────────────────────────────────────────
    """
    _check_auth(x_bridge_secret)

    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised yet")

    payload = await req.json()

    # ── Parse payload (Baileys bridge format) ─────────────────────────────
    # META FORMAT (future): phone = payload["entry"][0]["changes"][0]["value"]
    #                                       ["messages"][0]["from"]
    #               text  = ...["messages"][0]["text"]["body"]
    phone = str(payload.get("from") or "").strip()
    text  = str(payload.get("text") or "").strip()

    if not phone or not text:
        raise HTTPException(status_code=400, detail="Missing 'from' or 'text'")

    thread_id  = f"wa_{phone}"
    request_id = uuid.uuid4().hex[:8]
    set_request_id(request_id)
    t0 = time.perf_counter()

    _logger.info(
        "incoming message",
        extra={"phone": phone, "thread_id": thread_id, "text_preview": text[:80]},
    )

    try:
        output = await _graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config=make_config(thread_id),
        )
    except Exception as exc:
        _logger.error("graph error", extra={"thread_id": thread_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))

    answer = extract_assistant_text(output) or "(sin respuesta)"

    dt_ms = round((time.perf_counter() - t0) * 1000, 1)
    _logger.info(
        "reply ready",
        extra={
            "thread_id": thread_id,
            "duration_ms": dt_ms,
            "route": output.get("locked_route"),
            "confidence": output.get("confidence"),
            "answer_preview": answer[:80],
        },
    )

    return {"response": answer}


# ── Meta verification handshake (stub — activate when Meta is ready) ──────────
#
# @router.get("/incoming")
# async def verify_webhook(
#     hub_mode: str = Query(alias="hub.mode"),
#     hub_challenge: str = Query(alias="hub.challenge"),
#     hub_verify_token: str = Query(alias="hub.verify_token"),
# ):
#     """Meta sends a GET request to verify the webhook URL before enabling delivery."""
#     expected = os.getenv("WA_VERIFY_TOKEN", "")
#     if hub_mode == "subscribe" and hub_verify_token == expected:
#         return PlainTextResponse(hub_challenge)
#     raise HTTPException(status_code=403, detail="Verification failed")
