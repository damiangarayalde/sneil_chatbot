from __future__ import annotations

from pydantic import BaseModel, Field


class HandlerOutput(BaseModel):
    """Route handler output.

    - If the user switched product/topic, we want to clear lock so hub can re-route.
    - Otherwise we send back an answer.
    """

    is_topic_switch: bool = Field(
        description="True if the user changed the topic to a different product."
    )
    answer: str = Field(
        description="The response to the user. Empty if is_topic_switch is True."
    )
