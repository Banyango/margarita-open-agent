
import os
from openai import AsyncOpenAI
from wireup import injectable


@injectable
def openai_client() -> AsyncOpenAI:
    """Factory that provides a configured AsyncOpenAI client instance.

    Reads API key from either OPEN_AI_API_KEY (existing env name in repo)
    or the standard OPENAI_API_KEY environment variable.

    Returns:
        An instance of openai.AsyncOpenAI.
    """
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if api_key:
        return AsyncOpenAI(api_key=api_key)

    return AsyncOpenAI()
