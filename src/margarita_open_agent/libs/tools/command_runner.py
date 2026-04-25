import asyncio

from wireup import injectable

from margarita_open_agent.core.interfaces import PermissionCallbackHandler
from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    PermissionRequestKind,
)
from margarita_open_agent.core.models.tool import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
)

DEFINITION: ToolDefinition = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="command_runner",
        description="Execute a shell command and return its output. stderr is combined with stdout.",
        parameters=ToolParameters(
            type="object",
            properties={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            required=["command"],
        ),
    ),
)


@injectable
class CommandRunnerTool:
    def __init__(self, permission_service: PermissionCallbackHandler):
        self.permission_service = permission_service

    async def execute(self, arguments: dict) -> str:
        """Execute a shell command and return its combined stdout/stderr output.

        Args:
            arguments: A dict containing the key "command" with the shell command to run.

        Returns:
            The combined stdout/stderr output as a string, or an error message if the
            command exits with a non-zero status.
        """
        command: str = arguments["command"]

        if not await self.permission_service(
            PermissionsRequest(
                kind=PermissionRequestKind.COMMAND,
                command=command,
                tool_name="command_runner",
            )
        ):
            return "Request was denied by the user"

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace").strip()

        if proc.returncode != 0:
            return f"Command failed (exit {proc.returncode}):\n{output}"

        return output or "(no output)"
