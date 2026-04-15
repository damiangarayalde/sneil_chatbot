import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.logging_config import configure_logging, get_logger, set_request_id
from app.core.utils import get_routes, load_cfg
from app.core.persistence import (
    get_db_path,
    get_sqlite_checkpointer,
    delete_old_threads,
    get_db_stats,
)
from app.core.graph.build import build_graph
from app.interfaces.chatbot_ui_mockup_helpers import (
    extract_assistant_text,
    make_config,
    render_page,
    validate_route,
)

configure_logging()
_logger = get_logger("sneil.api")


"""
Dev API for quickly testing the full chatbot graph with a WhatsApp-style UI.

This mirrors cli.py's full chatbot behavior but with a web UI and reset functionality.
"""

load_dotenv()

TEST_KEY = os.getenv("TEST_KEY", "")

# Sync checkpointer — used only for direct SQL ops (reset, health, TTL cleanup).
CHECKPOINTER = get_sqlite_checkpointer()

# Graph is set during lifespan startup once AsyncSqliteSaver is open.
graph = None

# In-memory job store: job_id -> {"status": "pending"|"done"|"error", "result": ...}
_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    if not TEST_KEY:
        _logger.warning(
            "TEST_KEY is not set — the /chat and /reset endpoints are unprotected. "
            "Set TEST_KEY in your environment to require authentication."
        )

    # Build the graph with an async checkpointer so ainvoke works correctly.
    async with AsyncSqliteSaver.from_conn_string(get_db_path()) as async_checkpointer:
        graph = build_graph(checkpointer=async_checkpointer)

        cfg = load_cfg()
        ttl_hours = cfg.get("THREAD_TTL_HOURS", 24)
        try:
            deleted = delete_old_threads(CHECKPOINTER, ttl_hours)
            if deleted > 0:
                _logger.info(
                    "startup cleanup completed",
                    extra={"deleted_threads": deleted, "ttl_hours": ttl_hours},
                )
        except Exception as e:
            _logger.error("startup cleanup failed", extra={"error": str(e)})

        yield  # app runs here; AsyncSqliteSaver stays open for the process lifetime


app = FastAPI(lifespan=lifespan)


def _check_auth(x_test_key: str | None):
    """Raise 401 if TEST_KEY is configured and the header does not match."""
    if TEST_KEY and x_test_key != TEST_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _run_graph(job_id: str, text: str, thread_id: str, config: dict):
    """Background task: invoke the graph and store the result in _jobs."""
    request_id = uuid.uuid4().hex[:8]
    set_request_id(request_id)
    t0 = time.perf_counter()
    _logger.info("job start", extra={"job_id": job_id, "thread_id": thread_id})

    try:
        input_data = {"messages": [HumanMessage(content=text)]}
        output = await graph.ainvoke(input_data, config=config)

        msgs = output.get("messages") or []
        last_human_idx = next(
            (i for i in range(len(msgs) - 1, -1, -1)
             if isinstance(msgs[i], HumanMessage)),
            None,
        )
        start = (last_human_idx + 1) if last_human_idx is not None else 0
        ai_after = [m for m in msgs[start:] if isinstance(m, AIMessage)]
        answer = (
            "\n".join(m.content.strip() for m in ai_after if m.content.strip())
            or "(no assistant output)"
        )

        state_info = {
            "locked_route": output.get("locked_route"),
            "estimated_route": output.get("estimated_route"),
            "confidence": output.get("confidence"),
            "routing_attempts": output.get("routing_attempts"),
            "solve_attempts": output.get("solve_attempts"),
            "max_solve_attempts": output.get("max_solve_attempts"),
            "escalated_to_human": output.get("escalated_to_human"),
            "retrieved_count": len(output.get("retrieved") or []),
        }

        dt_ms = round((time.perf_counter() - t0) * 1000, 1)
        _logger.info(
            "job complete",
            extra={
                "job_id": job_id,
                "thread_id": thread_id,
                "duration_ms": dt_ms,
                "route": state_info.get("locked_route"),
                "confidence": state_info.get("confidence"),
            },
        )

        _jobs[job_id] = {"status": "done", "result": {
            "answer": answer, "state": state_info}}

    except Exception as exc:
        _logger.error("job failed", extra={
                      "job_id": job_id, "error": str(exc)})
        _jobs[job_id] = {"status": "error", "result": {"detail": str(exc)}}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render_page(__file__, "Full Chatbot"))


@app.post("/chat", status_code=202)
async def chat(req: Request, x_test_key: str | None = Header(default=None)):
    _check_auth(x_test_key)

    payload = await req.json()

    text = (payload.get("text") or "").strip()
    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending", "result": None}

    config = make_config(thread_id)
    asyncio.create_task(_run_graph(job_id, text, thread_id, config))

    return {"job_id": job_id, "status": "pending"}


@app.get("/result/{job_id}")
async def get_result(job_id: str, x_test_key: str | None = Header(default=None)):
    _check_auth(x_test_key)

    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "pending":
        return {"job_id": job_id, "status": "pending"}

    # Clean up completed/errored jobs after retrieval
    result = _jobs.pop(job_id)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["result"]["detail"])

    return {"job_id": job_id, "status": "done", **result["result"]}


@app.post("/reset")
async def reset(req: Request, x_test_key: str | None = Header(default=None)):
    _check_auth(x_test_key)

    payload = await req.json()
    thread_id = (payload.get("thread_id") or "").strip() or "dev-thread"
    # Reset for full graph: delete all checkpoints for this thread_id
    CHECKPOINTER.conn.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
    )
    CHECKPOINTER.conn.commit()
    return {"ok": True}


@app.get("/health")
async def health():
    """Health check endpoint with database statistics (Gap 4).

    Returns:
        - status: "ok" if healthy
        - db: Database statistics (size_bytes, active_thread_count, total_checkpoints)
        - max_history_messages: Configured message history limit
        - thread_ttl_hours: Configured thread TTL
    """
    try:
        stats = get_db_stats(CHECKPOINTER)
        cfg = load_cfg()

        return {
            "status": "ok",
            "db": stats,
            "max_history_messages": cfg.get("MAX_HISTORY_MESSAGES", 20),
            "thread_ttl_hours": cfg.get("THREAD_TTL_HOURS", 24),
        }
    except Exception as e:
        _logger.error("health check failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(e)}")


def main():
    uvicorn.run(
        "app.interfaces.dev_api:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        reload_dirs=["app"],
        reload_includes=["*.py"],
    )


if __name__ == "__main__":
    main()
