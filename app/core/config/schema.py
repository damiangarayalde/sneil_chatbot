"""
Pydantic schema for config.yaml (Gap 8 — Config validation).

At startup, the raw YAML dict is passed through :func:`validate_cfg`.
A misconfigured value (e.g. a negative ``max_chars``, a missing
``prompt_file``, or an out-of-range threshold) raises immediately so
the process never starts in a broken state.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class WhatsAppLinks(BaseModel):
    tech: str
    sales: str


class HeuristicsConfig(BaseModel):
    mentions: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    disambiguation_question: Optional[str] = None
    clarifier: Optional[str] = None
    question: Optional[str] = None

    model_config = {"extra": "allow"}


class ClassifierConfig(BaseModel):
    max_chars: int = Field(gt=0)
    preferred_chars: Optional[list[int]] = None
    whatsapp: Optional[WhatsAppLinks] = None
    prompt_file: str
    max_attempts_before_handoff: int = Field(ge=1)
    route_lock_threshold: float = Field(ge=0.0, le=1.0)


class RouteConfig(BaseModel):
    max_chars: int = Field(gt=0)
    preferred_chars: Optional[list[int]] = None
    whatsapp: Optional[WhatsAppLinks] = None
    prompt_file: str
    max_attempts_before_handoff: int = Field(ge=1)
    heuristics: Optional[HeuristicsConfig] = None
    special: Optional[dict[str, Any]] = None

    model_config = {"extra": "allow"}


class AppConfig(BaseModel):
    """Top-level config model.

    ``CLASSIFIER`` is validated as :class:`ClassifierConfig`.
    Every other top-level key is validated as a :class:`RouteConfig`.
    Unknown sub-keys inside route sections are tolerated (forward compat).
    """

    CLASSIFIER: ClassifierConfig
    MAX_HISTORY_MESSAGES: int = Field(default=20, ge=1)
    THREAD_TTL_HOURS: int = Field(default=24, ge=1)

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _validate_routes(cls, data: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        for key, value in data.items():
            if key in ("CLASSIFIER", "MAX_HISTORY_MESSAGES", "THREAD_TTL_HOURS"):
                continue
            if not isinstance(value, dict):
                errors.append(
                    f"Route '{key}': expected a mapping, got {type(value).__name__}"
                )
                continue
            try:
                RouteConfig(**value)
            except Exception as exc:
                errors.append(f"Route '{key}': {exc}")
        if errors:
            raise ValueError(
                "config.yaml validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return data


def validate_cfg(raw: dict[str, Any]) -> None:
    """Validate *raw* (the result of ``yaml.safe_load``) against :class:`AppConfig`.

    Raises :class:`pydantic.ValidationError` (or :class:`ValueError`) on the
    first structural or constraint violation so the process hard-fails at startup
    rather than producing a cryptic error deep inside a graph node.
    """
    AppConfig(**raw)
