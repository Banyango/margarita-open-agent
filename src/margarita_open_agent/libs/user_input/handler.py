from wireup import injectable

from margarita_open_agent.core.interfaces import UserInputCallbackHandler


@injectable(as_type=UserInputCallbackHandler)
class DefaultUserInputCallbackHandler(UserInputCallbackHandler):
    async def __call__(self, request: UserInputCallbackHandler) -> str:
        return "No input handler available so ignore user input"
