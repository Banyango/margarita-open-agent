from wireup import injectable

from margarita_open_agent.core.interfaces import UserToolCallbackHandler
from margarita_open_agent.core.models.tool import ToolCallRequest

# This is overridden by the passed in handler.


@injectable(as_type=UserToolCallbackHandler)
class UserToolCallbackHandlerImpl(UserToolCallbackHandler):
    async def __call__(self, request: ToolCallRequest) -> str:
        pass
