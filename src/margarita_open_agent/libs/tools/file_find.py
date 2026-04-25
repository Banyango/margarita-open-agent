from pathlib import Path

from wireup import injectable

from margarita_open_agent.core.models.tool import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
)

DEFINITION = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="file_find",
        description=(
            "Find the path of a filename. Use this when a file doesn't have a path specified."
        ),
        parameters=ToolParameters(
            type="object",
            properties={
                "directory": {
                    "type": "string",
                    "description": "The root directory to search from.",
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match filenames against, e.g. '*.py' "
                        "or '**/*test*.py'. The search is always recursive."
                    ),
                },
            },
            required=["directory", "pattern"],
        ),
    ),
)


@injectable
class FileFindTool:
    async def execute(self, arguments: dict) -> str:
        """Search for files matching a glob pattern under a directory.

        Args:
            arguments: A dict with keys:
                - ``directory``: Root path to search from.
                - ``pattern``: Glob pattern (always searched recursively).

        Returns:
            A newline-separated string of matching file paths, or an error message.
        """
        directory = Path(arguments["directory"])
        pattern = arguments["pattern"]

        try:
            matches = sorted(str(p) for p in directory.rglob(pattern))
            return "\n".join(matches) if matches else "No files found."
        except NotADirectoryError:
            return f"Error: not a directory: {directory}"
        except OSError as e:
            return f"Error searching {directory}: {e}"
