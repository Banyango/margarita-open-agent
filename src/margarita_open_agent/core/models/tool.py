from typing import TypedDict, Literal


class ToolParameters(TypedDict, total=False):
    type: str
    properties: dict
    required: list[str]


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: ToolParameters | None


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: ToolFunction


class ToolCallRequest(TypedDict):
    name: str
    arguments: dict
