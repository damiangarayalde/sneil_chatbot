"""
Central logging configuration (Gap 3 — Structured Logging).

Usage
-----
Call ``configure_logging()`` once at process startup (dev_api.py, cli.py).
In the API layer, call ``set_request_id(str(uuid.uuid4()))`` at the start of
each request so every downstream log line carries it automatically.

Environment variables
---------------------
LOG_LEVEL   INFO (default) | DEBUG | WARNING | ERROR | CRITICAL
LOG_FORMAT  json (default when non-TTY) | pretty (default when TTY)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Request-ID context variable
# ---------------------------------------------------------------------------

_REQUEST_ID_VAR: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Bind *request_id* to the current async-task / thread context."""
    _REQUEST_ID_VAR.set(request_id)


def get_request_id() -> str:
    """Return the request ID bound to the current context, or ``'-'``."""
    return _REQUEST_ID_VAR.get()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

# Extra keys that nodes and wrappers may pass via ``extra={...}``
_STRUCTURED_KEYS = frozenset({
    "node",
    "route",
    "confidence",
    "duration_ms",
    "thread_id",
    "retrieved_count",
    "catalog_matches",
})


class _JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log record (stdout-friendly)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _REQUEST_ID_VAR.get(),
            "msg": record.getMessage(),
        }
        for key in _STRUCTURED_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _PrettyFormatter(logging.Formatter):
    """Human-readable one-liner that includes request_id."""

    _FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    _DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        rid = _REQUEST_ID_VAR.get()
        base = super().format(record)
        # Append any structured extras that are present
        extras = {k: getattr(record, k) for k in _STRUCTURED_KEYS if getattr(record, k, None) is not None}
        suffix = ("  " + "  ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""
        return f"[{rid}] {base}{suffix}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure the root logger.  Safe to call more than once (idempotent)."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt_env = os.getenv("LOG_FORMAT", "").lower()
    use_json = fmt_env == "json" or (fmt_env != "pretty" and not sys.stdout.isatty())

    formatter: logging.Formatter
    if use_json:
        formatter = _JsonFormatter()
    else:
        formatter = _PrettyFormatter(fmt=_PrettyFormatter._FMT, datefmt=_PrettyFormatter._DATE)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers unless we are in DEBUG mode
    if level > logging.DEBUG:
        for noisy in ("httpcore", "httpx", "openai", "langchain", "langgraph", "chromadb", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
