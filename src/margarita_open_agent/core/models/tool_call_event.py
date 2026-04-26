from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolCallCallingMetadata:
    """Metadata for a tool_call StreamEvent when the tool is being invoked.

    Attributes:
        name: The name of the tool being called.
        arguments: The arguments passed to the tool.
    """

    name: str
    arguments: dict


@dataclass
class ToolCallDoneMetadata:
    """Metadata for a tool_call StreamEvent when the tool has finished.

    Attributes:
        name: The name of the tool that was called.
        arguments: The arguments that were passed to the tool.
        result: The output returned by the tool.
        state: Always ``"done"``.
    """

    name: str
    arguments: dict
    result: str | None
    state: Literal["done"] = "done"
    success: bool = False
