from wireup import injectable

from margarita_open_agent.core.interfaces import UserToolCallbackHandler
from margarita_open_agent.core.models.tool import ToolCallRequest
from margarita_open_agent.core.models.tool import ToolDefinition
from margarita_open_agent.libs.tools import ask_user as ask_user_module
from margarita_open_agent.libs.tools import code_search as code_search_module
from margarita_open_agent.libs.tools import command_runner as command_runner_module
from margarita_open_agent.libs.tools import file_find as file_find_module
from margarita_open_agent.libs.tools import file_read as file_read_module
from margarita_open_agent.libs.tools import file_write as file_write_module
from margarita_open_agent.libs.tools.ask_user import AskUserTool
from margarita_open_agent.libs.tools.code_search import CodeSearchTool
from margarita_open_agent.libs.tools.command_runner import CommandRunnerTool
from margarita_open_agent.libs.tools.file_find import FileFindTool
from margarita_open_agent.libs.tools.file_read import FileReadTool
from margarita_open_agent.libs.tools.file_write import FileWriteTool


@injectable
class ToolRegistry:
    def __init__(
        self,
        ask_user: AskUserTool,
        code_search: CodeSearchTool,
        command_runner: CommandRunnerTool,
        file_find: FileFindTool,
        file_read: FileReadTool,
        file_write: FileWriteTool,
        user_tool_handler: UserToolCallbackHandler,
    ):
        self.user_tool_handler = user_tool_handler
        self._tool_registry = {
            "ask_user": ask_user,
            "code_search": code_search,
            "command_runner": command_runner,
            "file_find": file_find,
            "file_read": file_read,
            "file_write": file_write,
        }

    @staticmethod
    def get_tool_definitions() -> list[ToolDefinition]:
        """Return the tool definitions for all registered tools.

        Returns:
            A list of ToolDefinition objects describing each available tool.
        """
        return [
            ask_user_module.DEFINITION,
            code_search_module.DEFINITION,
            command_runner_module.DEFINITION,
            file_find_module.DEFINITION,
            file_read_module.DEFINITION,
            file_write_module.DEFINITION,
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a registered tool by name with the provided arguments.

        Args:
            name: The tool name to execute.
            arguments: A dictionary of arguments to pass to the tool.

        Returns:
            The output string returned by the tool.
        """
        if name not in self._tool_registry:
            return await self.user_tool_handler(
                ToolCallRequest(name=name, arguments=arguments)
            )

        tool = self._tool_registry[name]

        return await tool.execute(arguments)
