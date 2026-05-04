# Margarita Open Agent

A Python framework for building LLM-powered agents backed by [Ollama](https://ollama.com/). It provides a clean session-based API, a built-in tool registry, and a permission/user-input callback system so you can wire the agent into any interface.

---

## Features

- **Session-based conversations** – `AgentSession` manages the full message history and drives multi-turn conversations automatically.
- **Streaming & async support** – Stream responses token-by-token with `send_and_stream_async`, or await the final reply with `send_and_wait_async`.
- **Built-in tools** – Ready-made tools the LLM can invoke:
  - `ask_user` – ask a clarifying question back to the user
  - `file_read` / `file_write` / `file_find` – read, write and search files
  - `code_search` – search source code
  - `command_runner` – execute shell commands
- **Custom tools** – Register your own tools by providing a `ToolDefinition` and a `UserToolCallbackHandler`.
- **Permission hooks** – Every sensitive operation (file writes, shell commands, …) is gated behind a `PermissionCallbackHandler` that you control.
- **Dependency injection** – Internals are wired with [wireup](https://github.com/maldoinc/wireup); easy to extend and test.

---

## Requirements

- Python ≥ 3.12
- [Ollama](https://ollama.com/) running locally (default backend)

---

## Installation

```bash
# clone the repo
git clone https://github.com/your-org/margarita-open-agent.git
cd margarita-open-agent

# install with uv (recommended)
uv sync
```

---

## Quick Start

```python
import asyncio
from margarita_open_agent.core.models.permissions import PermissionsRequest, UserInputRequest
from margarita_open_agent.core.interfaces import (
  PermissionCallbackHandler,
  UserInputCallbackHandler,
  UserToolCallbackHandler,
)
from margarita_open_agent.core.sessions.session import AgentSession


class PermissionsImpl(PermissionCallbackHandler):
  async def __call__(self, request: PermissionsRequest) -> bool:
    answer = input(f"Allow {request.kind}? [y/N] ").strip().lower()
    return answer == "y"


class UserInputImpl(UserInputCallbackHandler):
  async def __call__(self, request: UserInputRequest) -> str:
    return input(f"[Agent asks] {request.question}\n> ").strip()


async def main():
  session = AgentSession(
    model="llama:latest",
    system_message="You are a helpful assistant.",
    additional_tools=[],
    on_user_input_request=UserInputImpl(),
    on_permission_request=PermissionsImpl(),
    on_custom_tool_request=None,
  )

  reply = await session.send_and_wait_async("Hello! What can you do?")
  print(reply)


asyncio.run(main())
```

See `main.py` for a more complete example including a custom tool.

---

## Custom Tools

```python
from margarita_open_agent.core.models.tool import ToolDefinition, ToolFunction, ToolCallRequest
from margarita_open_agent.core.interfaces import UserToolCallbackHandler

MY_TOOL = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="my_tool",
        description="Does something useful.",
        parameters=None,
    ),
)

class MyToolHandler(UserToolCallbackHandler):
    async def __call__(self, request: ToolCallRequest) -> str:
        if request["name"] == "my_tool":
            return "Tool result goes here."
        return ""
```

Pass `additional_tools=[MY_TOOL]` and `on_custom_tool_request=MyToolHandler()` when constructing `AgentSession`.

---

## Development

```bash
# run unit tests
uv run pytest tests/unit

# run integration tests (requires Ollama)
uv run pytest tests/integration

# lint & format
uv run ruff check .
uv run ruff format .
```

---

## Project Structure

```
src/margarita_open_agent/
├── session.py          # AgentSession – main entry point
├── container.py        # DI container setup
├── core/
│   ├── llm.py          # LLM client abstraction
│   ├── interfaces.py   # Callback handler protocols
│   └── models/         # Pydantic models (messages, tools, events, …)
└── libs/
    ├── ollama/         # Ollama backend implementation
    ├── tools/          # Built-in tool implementations & registry
    ├── permissions/    # Permission gate handler
    └── user_input/     # User-input handler
```

---

## License

MIT
