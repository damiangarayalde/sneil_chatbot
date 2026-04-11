"""LLM provider abstraction for mock / real switching.

Set the environment variable ``LLM_MOCK=true`` (or ``1`` / ``yes``) before
importing any graph modules to activate the mock layer.  All LLM chains will
then read canned responses from JSON fixture files instead of calling OpenAI,
so the full scenario test suite runs with zero API calls and no key required.

Fixture directory
-----------------
``tests/fixtures/mock_llm/<OutputClassName>.json``

Each file is a JSON array of entries evaluated top-to-bottom:

.. code-block:: json

    [
        {"match": "keyword present in user_text", "output": { ... }},
        {"output": { ... }}   // fallback — no "match" key
    ]

The first entry whose ``match`` value appears (case-insensitive substring) in
the incoming ``user_text`` wins.  The last entry should always be a fallback
with no ``match`` key so every input resolves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Type

from pydantic import BaseModel

# Resolved relative to this file so the path works regardless of cwd.
_FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "mock_llm"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_mock_mode() -> bool:
    """Return True when LLM_MOCK env var is set to a truthy value."""
    return os.getenv("LLM_MOCK", "").lower() in ("1", "true", "yes")


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    """Return the real LangChain LLM.

    Only call this when *not* in mock mode — it requires OPENAI_API_KEY.
    """
    from app.core.utils import init_llm  # local import avoids circular deps
    return init_llm(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# MockChain
# ---------------------------------------------------------------------------

class MockChain:
    """Drop-in replacement for a LangChain ``prompt | llm.with_structured_output(Schema)`` chain.

    It reads fixture JSON from ``tests/fixtures/mock_llm/<OutputClass>.json``
    and returns canned Pydantic instances without touching the network.

    The ``.invoke(inputs)`` signature mirrors what LangGraph nodes pass —
    a dict with at least a ``user_text`` key.
    """

    def __init__(self, output_cls: Type[BaseModel]) -> None:
        self._output_cls = output_cls
        fixture_path = _FIXTURES_DIR / f"{output_cls.__name__}.json"
        if fixture_path.exists():
            self._fixtures: list[dict] = json.loads(
                fixture_path.read_text(encoding="utf-8")
            )
        else:
            self._fixtures = []

    def invoke(self, inputs: dict, **kwargs) -> BaseModel:
        """Return the first fixture whose ``match`` string appears in ``user_text``."""
        user_text = (inputs.get("user_text") or "").lower()

        # 1) Match entries
        for entry in self._fixtures:
            match_str = (entry.get("match") or "").lower()
            if match_str and match_str in user_text:
                return self._output_cls(**entry["output"])

        # 2) Fallback — first entry without a match key
        for entry in self._fixtures:
            if not entry.get("match"):
                return self._output_cls(**entry["output"])

        raise RuntimeError(
            f"MockChain: no fixture matched for {self._output_cls.__name__}. "
            f"user_text={user_text!r}. "
            f"Add a fallback entry (no 'match' key) to "
            f"{_FIXTURES_DIR / (self._output_cls.__name__ + '.json')}"
        )
