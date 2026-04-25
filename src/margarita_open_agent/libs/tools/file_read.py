from pathlib import Path

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
        name="file_read",
        description="Read the contents of a file at the given path.",
        parameters=ToolParameters(
            type="object",
            properties={
                "path": {
                    "type": "string",
                    "description": "The path of the file to read.",
                },
            },
            required=["path"],
        ),
    ),
)


@injectable
class FileReadTool:
    def __init__(self, permission_service: PermissionCallbackHandler):
        self.permission_service = permission_service

    async def execute(self, arguments: dict) -> str:
        """Read and return the contents of a file.

        Args:
            arguments: A dict containing the key "path" with the file path to read.

        Returns:
            The file contents as a string, or an error message string on failure.
        """
        path = Path(arguments["path"])

        if not await self.permission_service(
            PermissionsRequest(
                kind=PermissionRequestKind.FILE_READ,
                paths=[str(path.absolute())],
                tool_name="file_read",
            )
        ):
            return "Request was denied by the user"

        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except IsADirectoryError:
            return f"Error: path is a directory: {path}"
        except OSError as e:
            return f"Error reading {path}: {e}"
