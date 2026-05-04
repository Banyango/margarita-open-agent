from abc import abstractmethod, ABC
from collections.abc import AsyncIterator

from margarita_open_agent.core.models.message import Message
from margarita_open_agent.core.models.stream_event import StreamEvent
from margarita_open_agent.core.models.tool import ToolDefinition


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self, model: str, messages: list[Message], tools: list[ToolDefinition]
    ) -> Message:
        """Run a chat call against the underlying LLM implementation.

        Args:
            model: The LLM model enum to select a specific model/config.
            messages: Conversation history provided to the model.
            tools: Optional list of tools the model may call.

        Returns:
            A Message representing the model's reply. Concrete implementations
            must implement this method.
        """

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[StreamEvent]:
        """Stream the assistant reply as ``StreamEvent`` objects.

        Yields both ``"thinking"`` events (chain-of-thought tokens) and
        ``"content"`` events (visible reply tokens).

        Args:
            model: The LLM model enum to select a specific model/config.
            messages: Conversation history provided to the model.
            tools: Optional list of tools the model may call.

        Yields:
            :class:`StreamEvent` instances with ``type`` set to either
            ``"thinking"`` or ``"content"``.
        """
