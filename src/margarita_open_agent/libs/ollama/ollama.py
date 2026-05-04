import dataclasses
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from ollama import AsyncClient
from wireup import injectable

from margarita_open_agent.core.llm import LLMClient
from margarita_open_agent.core.models.errors import ModelNotSpecifiedError
from margarita_open_agent.core.models.message import Message
from margarita_open_agent.core.models.stream_event import StreamEvent
from margarita_open_agent.core.models.tool import ToolDefinition
from margarita_open_agent.core.models.tool_call_event import ToolCallCallingMetadata
from margarita_open_agent.core.sessions.session_event import SessionEventType


@dataclass
class ModelConfig:
    think: bool | Literal["low", "medium", "high"]
    model: str | None = None


OLLAMA_MODEL_CONFIG: dict[str, ModelConfig] = {
    "granite4.1": ModelConfig(think=False),
    "gemma4": ModelConfig(think=True),
    "deepseek-r1": ModelConfig(think=True),
    "deepseek-v3.1": ModelConfig(think=True),
    "deepseek-v3.2": ModelConfig(think=True),
    "deepseek-v4-flash": ModelConfig(think=True),
    "qwen3": ModelConfig(think=True),
    "qwen3.5": ModelConfig(think=True),
    "qwen3.6": ModelConfig(think=True),
    "qwen3-next": ModelConfig(think=True),
    "glm-4.6": ModelConfig(think=True),
    "glm-4.7": ModelConfig(think=True),
    "glm-5": ModelConfig(think=True),
    "kimi-k2-thinking": ModelConfig(think=True),
    "kimi-k2.5-thinking": ModelConfig(think=True),
    "kimi-k2.6-thinking": ModelConfig(think=True),
    "minimax-m2": ModelConfig(think=True),
    "minimax-m2.5": ModelConfig(think=True),
    "minimax-m2.7": ModelConfig(think=True),
    "nemotron-3-nano": ModelConfig(think=True),
    "nemotron-3-super": ModelConfig(think=True),
    "nemotron-cascade": ModelConfig(think=True),
    "nemotron-cascade-2": ModelConfig(think=True),
    "magistral": ModelConfig(think=True),
    "gemini-3-flash-preview": ModelConfig(think=True),
    "gpt-oss": ModelConfig(think=True),
    "gpt-oss-safeguard": ModelConfig(think=True),
}


@injectable(as_type=LLMClient)
class OllamaLLMClient(LLMClient):
    def __init__(self, client: AsyncClient):
        """Initialize the Ollama LLM client wrapper.

        Args:
            client: An instance of ollama.AsyncClient used to perform chats.
        """

        self.client = client

    async def chat(
        self, model: str, messages: list[Message], tools: list[ToolDefinition]
    ) -> Message:
        """Send messages to the Ollama client and convert the response to a Message.

        Args:
            model: The LLMModelEnum indicating which model configuration to use.
            messages: Conversation history as a list of Message objects.
            tools: Optional list of ToolDefinition objects available to the model.

        Returns:
            A Message object containing the assistant reply and any tool call info.
        """
        if model == "" or model is None:
            raise ModelNotSpecifiedError()

        ollama_model = ModelConfig(model=model, think=False)

        response = await self.client.chat(
            messages=messages,
            tools=tools or None,
            **dataclasses.asdict(ollama_model),
            stream=False,
        )
        msg = response["message"]
        return Message(
            role=msg.role,
            content=msg.content or "",
            tool_calls=[
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in (msg.tool_calls or [])
            ],
        )

    async def stream(
        self, model: str, messages: list[Message], tools: list[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        """Stream content and thinking chunks from the Ollama model token by token.

        Args:
            model: The LLMModelEnum indicating which model configuration to use.
            messages: Conversation history as a list of Message objects.
            tools: Optional list of ToolDefinition objects available to the model.

        Yields:
            :class:`StreamEvent` with ``type="thinking"`` for chain-of-thought
            tokens and ``type="content"`` for visible reply tokens.
        """
        if model == "" or model is None:
            raise ModelNotSpecifiedError()

        model_split = model.split(":")
        if len(model_split) != 2:
            raise ModelNotSpecifiedError()

        ollama_model = OLLAMA_MODEL_CONFIG[model_split[0]]
        ollama_model.model = model

        async for chunk in await self.client.chat(
            messages=messages,
            tools=tools or None,
            **dataclasses.asdict(ollama_model),
            stream=True,
        ):
            msg = chunk.get("message", {})
            thinking = msg.get("thinking", "")
            content = msg.get("content", "")
            # todo might consider pooling here.
            if thinking:
                yield StreamEvent(
                    type=SessionEventType.ASSISTANT_REASONING_DELTA, text=thinking
                )
            if content:
                yield StreamEvent(
                    type=SessionEventType.ASSISTANT_STREAMING_DELTA, text=content
                )
            if chunk.message.tool_calls:
                for tool_call in chunk.message.tool_calls:
                    yield StreamEvent(
                        type=SessionEventType.TOOL_REQUESTED,
                        text=f"Tool call: {tool_call.function.name}",
                        metadata=ToolCallCallingMetadata(
                            tool_call_id=str(uuid.uuid1()),
                            name=tool_call.function.name,
                            arguments=dict(tool_call.function.arguments),
                        ),
                    )
