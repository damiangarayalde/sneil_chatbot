from __future__ import annotations

from pydantic import BaseModel, Field


class HandlerOutput(BaseModel):
    """Route handler output.

    - If the user switched product/topic, we want to clear lock so classifier can re-route.
    - Otherwise we send back an answer.
    """

    is_topic_switch: bool = Field(
        description=(
            "True ONLY if the user clearly switched to a completely different product category "
            "(e.g. from AA to TPMS or CALDERA). "
            "Keep False for any question that is still about the current product, "
            "including price, availability, specs, or installation queries."
        )
    )
    answer: str = Field(
        description=(
            "The response to the user. "
            "Required and non-empty when is_topic_switch is False. "
            "May include a brief transition phrase when is_topic_switch is True."
        )
    )
    increment_solve_attempts: bool = Field(
        description="True if the user expressed the proposed solution was unsuccessful in solving his problem."
    )
