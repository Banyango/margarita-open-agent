from dataclasses import dataclass
from typing import Literal

from margarita_open_agent.core.models.tool_call_event import (
    ToolCallCallingMetadata,
    ToolCallDoneMetadata,
)


@dataclass
class StreamEvent:
    """A single event emitted during a streaming LLM response.

    Attributes:
        type: One of ``"thinking"``, ``"content"``, ``"status"``, or ``"tool_call"``.
        text: Human-readable text for this event.
        metadata: Optional structured data. For ``"tool_call"`` events this is either
            a :class:`ToolCallCallingMetadata` or :class:`ToolCallDoneMetadata` instance.
    """

    type: Literal["thinking", "content", "status", "tool_call"]
    text: str
    metadata: ToolCallCallingMetadata | ToolCallDoneMetadata | None = None
