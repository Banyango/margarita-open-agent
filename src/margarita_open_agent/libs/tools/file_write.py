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
        name="file_write",
        description=(
            "Edit a file by replacing an exact string with a new one. "
            "The old_str must appear exactly once in the file. "
            "If the file does not exist and old_str is empty, the file is created with new_str as its content."
        ),
        parameters=ToolParameters(
            type="object",
            properties={
                "path": {
                    "type": "string",
                    "description": "The path of the file to edit or create.",
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to find and replace. Leave empty to create a new file.",
                },
                "new_str": {
                    "type": "string",
                    "description": "The string to replace old_str with.",
                },
            },
            required=["path", "old_str", "new_str"],
        ),
    ),
)


@injectable
class FileWriteTool:
    def __init__(self, permission_service: PermissionCallbackHandler):
        self.permission_service = permission_service

    async def execute(self, arguments: dict) -> str:
        """Edit or create a file by replacing a unique old string with a new string.

        Args:
            arguments: A dict with keys path, old_str, and new_str.

        Returns:
            A human-readable status or error string.
        """
        path = Path(arguments["path"])
        old_str: str = arguments["old_str"]
        new_str: str = arguments["new_str"]

        if not path.exists():
            if old_str:
                return f"Error: file not found: {path}"
            if not await self.permission_service(
                PermissionsRequest(
                    kind=PermissionRequestKind.FILE_WRITE,
                    paths=[str(path.absolute())],
                    new_str=new_str,
                    tool_name="file_write",
                )
            ):
                return "Request was denied by the user"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_str, encoding="utf-8")
            return f"Created {path}"

        if not await self.permission_service(
            PermissionsRequest(
                kind=PermissionRequestKind.FILE_WRITE,
                paths=[str(path.absolute())],
                old_str=old_str,
                new_str=new_str,
                tool_name="file_write",
            )
        ):
            return "Request was denied by the user"

        content = path.read_text(encoding="utf-8")
        count = content.count(old_str)

        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return f"Error: old_str found {count} times in {path} — must be unique"

        path.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited {path}"
