from wireup import injectable

from margarita_open_agent.core.interfaces import UserInputCallbackHandler
from margarita_open_agent.core.models.permissions import UserInputRequest
from margarita_open_agent.core.models.tool import (
    ToolDefinition,
    ToolFunction,
    ToolParameters,
)

DEFINITION: ToolDefinition = ToolDefinition(
    type="function",
    function=ToolFunction(
        name="ask_user",
        description=(
            "Ask the user a question and wait for their response. "
            "Use this tool whenever you need clarification or additional information from the user."
        ),
        parameters=ToolParameters(
            type="object",
            properties={
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            required=["question"],
        ),
    ),
)


@injectable
class AskUserTool:
    def __init__(self, user_input_handler: UserInputCallbackHandler):
        self.user_input_handler = user_input_handler

    async def execute(self, arguments: dict) -> str:
        """Ask the user a question and return their response.

        Args:
            arguments: A dict containing the key ``question``.

        Returns:
            The user's response string.
        """
        question = arguments.get("question", "")

        response = await self.user_input_handler(UserInputRequest(question=question))

        if response is None:
            return ""

        return str(response)
