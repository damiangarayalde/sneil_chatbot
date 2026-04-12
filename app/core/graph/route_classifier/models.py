from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.core.utils import get_routes

ALLOWED_ROUTES = set(get_routes())


class ClassifierOutput(BaseModel):
    """Structured output model used to enforce the classifier's response shape."""

    estimated_route: Literal["TPMS", "AA", "CLIMATIZADOR", "UNKNOWN"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como un route_id válido (ej: TPMS, AA, CLIMATIZADOR). Use UNKNOWN as fallback when LLM fails.",
    )
    confidence: float = Field(..., ge=0, le=1)
    clarifying_question: Optional[str] = Field(
        None,
        description="If confidence is low, ask ONE short clarifying question that would most improve routing.",
    )

    @field_validator("estimated_route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        v = (v or "").strip()
        # Allow UNKNOWN as a special value for fallback cases
        if v == "UNKNOWN":
            return v
        if v not in ALLOWED_ROUTES:
            raise ValueError(f"Invalid estimated_route: {v}")
        return v
