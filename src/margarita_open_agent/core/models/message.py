from typing import TypedDict, Literal


class ToolCallFunction(TypedDict):
    name: str
    arguments: dict


class ToolCall(TypedDict):
    function: ToolCallFunction


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    thinking: bool
    tool_calls: list[ToolCall]
