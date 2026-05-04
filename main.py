import asyncio

from margarita_open_agent.core.interfaces import (
    UserInputCallbackHandler,
    PermissionCallbackHandler,
    UserToolCallbackHandler,
)
from margarita_open_agent.core.models.errors import ModelNotSpecifiedError
from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    UserInputRequest,
)
from margarita_open_agent.core.models.tool import (
    ToolCallRequest,
    ToolDefinition,
    ToolFunction,
)
from margarita_open_agent.core.sessions.session import AgentSession
from margarita_open_agent.core.sessions.session_event import SessionEventType

DEFINITION = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="greeting",
        description=("When the user asks about turtles use this tool."),
        parameters=None,
    ),
)


class CustomToolHandlerImpl(UserToolCallbackHandler):
    async def __call__(self, request: ToolCallRequest) -> str:
        if request["name"] == "greeting":
            return "Tell the user about our amazing pies."
        return ""


class UserInputHandlerImpl(UserInputCallbackHandler):
    async def __call__(self, request: UserInputRequest) -> str:
        print(f"\n\033[35m[Agent asks]: {request.question}\033[0m")
        return input("Your answer> ").strip()


class PermissionsRequestImpl(PermissionCallbackHandler):
    async def __call__(self, request: PermissionsRequest) -> bool:
        print(f"\n\033[33m[Permission Request] Kind: {request.kind}\033[0m")
        if request.command:
            print(f"  Command : {request.command}")
        if request.path:
            print(f"  Path    : {request.path}")
        if request.paths:
            print(f"  Paths   : {', '.join(request.paths)}")
        if request.file_name:
            print(f"  File    : {request.file_name}")
        if request.old_str or request.new_str:
            print(f"  Before  : {request.old_str!r}")
            print(f"  After   : {request.new_str!r}")
        if request.urls:
            print(f"  URLs    : {', '.join(request.urls)}")
        answer = input("Allow? [y/N] ").strip().lower()

        return answer == "y"


async def run() -> None:
    session = AgentSession(
        model="granite4.1:3b",
        system_message=(
            "You are a coding agent. Use tools whenever you need information from the filesystem."
            "If a user gives you a filename use the find file to get the full path, then use the read file tool to read the contents."
            "When done answering you must respond with <DONE>"
        ),
        on_user_input_request=UserInputHandlerImpl(),
        on_permission_request=PermissionsRequestImpl(),
        on_custom_tool_request=CustomToolHandlerImpl(),
        additional_tools=[DEFINITION],
    )

    print("Agent ready. Type a prompt, or 'exit' to quit.")
    while True:
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        try:
            print()
            in_thinking = False
            status_active = False
            async for event in session.send_and_stream_async(prompt):
                if event.type == SessionEventType.ASSISTANT_REASONING:
                    if in_thinking:
                        print("\033[0m", end="", flush=True)
                        in_thinking = False
                    print(
                        f"\r\033[36m\033[3m{event.text}\033[0m\033[K",
                        end="",
                        flush=True,
                    )
                    status_active = True
                elif event.type == SessionEventType.TOOL_EXECUTION_START:
                    if in_thinking:
                        print("\033[0m", end="", flush=True)
                        in_thinking = False
                    name = event.metadata.name
                    print(f"\r\033[32m✓ {name}\033[0m\033[K", flush=True)
                    status_active = True
                elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
                    if status_active:
                        print()  # move past status line
                        status_active = False
                    if not in_thinking:
                        print("\033[2m", end="", flush=True)
                        in_thinking = True
                    print(event.text, end="", flush=True)
                else:  # content
                    if status_active:
                        print()  # move past status line
                        status_active = False
                    if in_thinking:
                        print("\033[0m\n", end="", flush=True)
                        in_thinking = False
                    print(event.text, end="", flush=True)
            if in_thinking:
                print("\033[0m", end="", flush=True)
            if status_active:
                print()
            print("\n")
        except TimeoutError:
            print("\nRequest timed out.\n")
        except ModelNotSpecifiedError:
            print("\nModel Not Specified.\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    asyncio.run(run())
