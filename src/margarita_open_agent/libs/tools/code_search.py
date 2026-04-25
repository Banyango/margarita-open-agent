import asyncio
import shutil

from wireup import injectable

from margarita_open_agent.core.models.tool import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
)

DEFINITION: ToolDefinition = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="code_search",
        description="Search for a pattern across files using ripgrep (rg). Returns matching lines with file names and line numbers.",
        parameters=ToolParameters(
            type="object",
            properties={
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to current directory.",
                },
                "file_type": {
                    "type": "string",
                    "description": "Limit search to files of this type (e.g. 'py', 'js', 'go').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive. Defaults to false.",
                },
            },
            required=["pattern"],
        ),
    ),
)

_MAX_LINES = 50


@injectable
class CodeSearchTool:
    async def execute(self, arguments: dict) -> str:
        """Search for a regex pattern across files and return matching lines.

        Args:
            arguments: A dict containing:
                - pattern: The regex pattern to search for (required).
                - path: File or directory path to search (defaults to current directory).
                - file_type: Optional file extension (e.g. 'py') to limit the search.
                - case_sensitive: Whether the search should be case-sensitive.

        Returns:
            A human-readable string with matches or an error message.
        """
        pattern: str = arguments["pattern"]
        path: str = arguments.get("path", ".")
        file_type: str | None = arguments.get("file_type")
        case_sensitive: bool = arguments.get("case_sensitive", False)

        if shutil.which("rg"):
            return await self._search_rg(pattern, path, file_type, case_sensitive)
        return await self._search_grep(pattern, path, file_type, case_sensitive)

    @staticmethod
    async def _search_rg(
        pattern: str, path: str, file_type: str | None, case_sensitive: bool
    ) -> str:
        args = ["rg", "--line-number", "--with-filename", "--color=never"]
        if not case_sensitive:
            args.append("--ignore-case")
        if file_type:
            args += ["--type", file_type]
        args += [pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode == 1:
            return "No matches found"
        if proc.returncode != 0:
            return f"Search failed (exit {proc.returncode})"

        return _truncate(stdout.decode(errors="replace").strip())

    async def _search_grep(
        self, pattern: str, path: str, file_type: str | None, case_sensitive: bool
    ) -> str:
        args = ["grep", "-rn", "--include=*"]
        if not case_sensitive:
            args.append("-i")
        if file_type:
            args[2] = f"--include=*.{file_type}"
        args += [pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode == 1:
            return "No matches found"
        if proc.returncode != 0:
            return f"Search failed (exit {proc.returncode})"

        return _truncate(stdout.decode(errors="replace").strip())


def _truncate(output: str) -> str:
    if not output:
        return "No matches found"
    lines = output.splitlines()
    if len(lines) > _MAX_LINES:
        return (
            "\n".join(lines[:_MAX_LINES])
            + f"\n... ({len(lines) - _MAX_LINES} more lines)"
        )
    return output
