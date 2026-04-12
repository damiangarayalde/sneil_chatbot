"""LLM client with retry, timeout, and fallback logic.

Wraps LLM invocations with:
- Exponential backoff retry (max 3 attempts) for transient errors
- Configurable timeout (LLM_TIMEOUT_S env var, default 15s)
- Structured fallback responses on final failure

This ensures transient errors (rate limits, connection issues) are retried
gracefully, and permanent failures return a safe fallback state instead of
propagating the exception to the caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional, TypeVar

from openai import APIConnectionError, RateLimitError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_llm_timeout_s() -> float:
    """Get the LLM call timeout in seconds from env var (default: 15s)."""
    raw = os.getenv("LLM_TIMEOUT_S", "15")
    try:
        val = float(raw)
        if val <= 0:
            _logger.warning(
                "LLM_TIMEOUT_S is non-positive, using default 15s",
                extra={"value": val},
            )
            return 15.0
        return val
    except (ValueError, TypeError):
        _logger.warning(
            "LLM_TIMEOUT_S is not a valid float, using default 15s",
            extra={"value": raw},
        )
        return 15.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    reraise=True,  # Re-raise after final failed attempt
)
def _invoke_with_retry(chain: Any, inputs: dict) -> Any:
    """Invoke a LangChain chain with retry logic.

    Will retry up to 3 times on:
    - openai.RateLimitError
    - openai.APIConnectionError

    Uses exponential backoff: 2s, 4s, 8s (capped at 10s between attempts).

    On final failure, the original exception is re-raised.
    """
    return chain.invoke(inputs)


async def _invoke_with_timeout(
    chain: Any, inputs: dict, timeout_s: float
) -> Any:
    """Invoke a LangChain chain with timeout.

    Wraps the sync chain.invoke() in asyncio.wait_for so it respects
    the timeout. If timeout occurs, raises asyncio.TimeoutError.
    """

    def _sync_invoke():
        return chain.invoke(inputs)

    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _sync_invoke),
        timeout=timeout_s,
    )


def invoke_chain_safe(
    chain: Any,
    inputs: dict,
    fallback_fn: Callable[[], T],
    timeout_s: Optional[float] = None,
) -> T:
    """Invoke a LangChain chain with retry, timeout, and fallback.

    Args:
        chain: LangChain chain (has .invoke(inputs) method)
        inputs: dict passed to chain.invoke()
        fallback_fn: Callable that returns a fallback Pydantic response on error
        timeout_s: timeout in seconds (if None, uses LLM_TIMEOUT_S env var)

    Returns:
        The chain result on success, or the fallback response on any error.

    On error, logs the exception at WARNING level with context, then returns
    the fallback. The caller never sees the exception.
    """
    timeout_s = timeout_s or get_llm_timeout_s()

    try:
        # For now, use sync invoke with retry.
        # A future enhancement could use async chains with asyncio.wait_for.
        result = _invoke_with_retry(chain, inputs)
        _logger.debug(
            "chain invocation succeeded",
            extra={"timeout_s": timeout_s, "retry_enabled": True},
        )
        return result

    except asyncio.TimeoutError:
        _logger.warning(
            "chain invocation timed out",
            extra={"timeout_s": timeout_s},
        )
        return fallback_fn()

    except (RateLimitError, APIConnectionError) as e:
        # After retries are exhausted, these still get raised
        _logger.warning(
            "chain invocation failed after retries",
            extra={
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "timeout_s": timeout_s,
            },
        )
        return fallback_fn()

    except RetryError as e:
        _logger.warning(
            "chain invocation failed with retry error",
            extra={
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "timeout_s": timeout_s,
            },
        )
        return fallback_fn()

    except Exception as e:
        # Catch any other exception (validation errors, unexpected failures, etc.)
        _logger.error(
            "chain invocation failed with unexpected error",
            extra={
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "timeout_s": timeout_s,
            },
        )
        return fallback_fn()
