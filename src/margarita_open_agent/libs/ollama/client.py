import os

from ollama import AsyncClient
from wireup import injectable


@injectable
def ollama_client() -> AsyncClient:
    """Factory that provides a configured AsyncClient instance for Ollama.

    The function is registered with the wireup injector so other components
    can depend on AsyncClient.

    Returns:
        An instance of ollama.AsyncClient.
    """
    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        return AsyncClient()

    return AsyncClient(
        headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")}
    )
