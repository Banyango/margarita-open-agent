import dataclasses
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from ollama import AsyncClient
from wireup import injectable

from margarita_open_agent.core.llm import LLMClient
from margarita_open_agent.core.models.llm_model_enum import LLMModelEnum
from margarita_open_agent.core.models.message import Message
from margarita_open_agent.core.models.stream_event import StreamEvent
from margarita_open_agent.core.models.tool import ToolDefinition
from margarita_open_agent.core.models.tool_call_event import ToolCallCallingMetadata
from margarita_open_agent.session_event import SessionEventType


@dataclass
class ModelConfig:
    model: str
    think: bool | Literal["low", "medium", "high"]


OLLAMA_MODEL_CONFIG: dict[LLMModelEnum, ModelConfig] = {
    # Gemma
    LLMModelEnum.GEMMA_4_LATEST: ModelConfig(model="gemma4:latest", think=True),
    LLMModelEnum.GEMMA_4_E2B: ModelConfig(model="gemma4:e2b", think=True),
    LLMModelEnum.GEMMA_4_E4B: ModelConfig(model="gemma4:e4b", think=True),
    LLMModelEnum.GEMMA_4_26B: ModelConfig(model="gemma4:26b", think=True),
    LLMModelEnum.GEMMA_4_31B: ModelConfig(model="gemma4:31b", think=True),
    LLMModelEnum.GEMMA_4_31B_CLOUD: ModelConfig(model="gemma4:31b-cloud", think=True),
    # DeepSeek
    LLMModelEnum.DEEPSEEK_R1_7B: ModelConfig(model="deepseek-r1:7b", think=True),
    LLMModelEnum.DEEPSEEK_V3_1_671B: ModelConfig(
        model="deepseek-v3.1:671b", think=True
    ),
    LLMModelEnum.DEEPSEEK_V3_2_CLOUD: ModelConfig(
        model="deepseek-v3.2:cloud", think=True
    ),
    LLMModelEnum.DEEPSEEK_V4_FLASH_CLOUD: ModelConfig(
        model="deepseek-v4-flash:cloud", think=True
    ),
    # Qwen3
    LLMModelEnum.QWEN3_8B: ModelConfig(model="qwen3:8b", think=True),
    LLMModelEnum.QWEN3_5_LATEST: ModelConfig(model="qwen3.5:latest", think=True),
    LLMModelEnum.QWEN3_5_0_8B: ModelConfig(model="qwen3.5:0.8b", think=True),
    LLMModelEnum.QWEN3_5_2B: ModelConfig(model="qwen3.5:2b", think=True),
    LLMModelEnum.QWEN3_5_4B: ModelConfig(model="qwen3.5:4b", think=True),
    LLMModelEnum.QWEN3_5_9B: ModelConfig(model="qwen3.5:9b", think=True),
    LLMModelEnum.QWEN3_5_27B: ModelConfig(model="qwen3.5:27b", think=True),
    LLMModelEnum.QWEN3_5_35B: ModelConfig(model="qwen3.5:35b", think=True),
    LLMModelEnum.QWEN3_5_122B: ModelConfig(model="qwen3.5:122b", think=True),
    LLMModelEnum.QWEN3_5_CLOUD: ModelConfig(model="qwen3.5:cloud", think=True),
    LLMModelEnum.QWEN3_5_397B_CLOUD: ModelConfig(
        model="qwen3.5:397b-cloud", think=True
    ),
    LLMModelEnum.QWEN3_6_LATEST: ModelConfig(model="qwen3.6:latest", think=True),
    LLMModelEnum.QWEN3_6_27B: ModelConfig(model="qwen3.6:27b", think=True),
    LLMModelEnum.QWEN3_6_35B: ModelConfig(model="qwen3.6:35b", think=True),
    LLMModelEnum.QWEN3_6_27B_CODING_MXFP8: ModelConfig(
        model="qwen3.6:27b-coding-mxfp8", think=True
    ),
    LLMModelEnum.QWEN3_6_27B_CODING_NVFP4: ModelConfig(
        model="qwen3.6:27b-coding-nvfp4", think=True
    ),
    LLMModelEnum.QWEN3_6_27B_CODING_BF16: ModelConfig(
        model="qwen3.6:27b-coding-bf16", think=True
    ),
    LLMModelEnum.QWEN3_6_27B_MLX_BF16: ModelConfig(
        model="qwen3.6:27b-mlx-bf16", think=True
    ),
    LLMModelEnum.QWEN3_6_35B_A3B_CODING_MXFP8: ModelConfig(
        model="qwen3.6:35b-a3b-coding-mxfp8", think=True
    ),
    LLMModelEnum.QWEN3_6_35B_A3B_CODING_NVFP4: ModelConfig(
        model="qwen3.6:35b-a3b-coding-nvfp4", think=True
    ),
    LLMModelEnum.QWEN3_6_35B_A3B_CODING_BF16: ModelConfig(
        model="qwen3.6:35b-a3b-coding-bf16", think=True
    ),
    LLMModelEnum.QWEN3_6_35B_A3B_MLX_BF16: ModelConfig(
        model="qwen3.6:35b-a3b-mlx-bf16", think=True
    ),
    LLMModelEnum.QWEN3_NEXT_LATEST: ModelConfig(model="qwen3-next:latest", think=True),
    LLMModelEnum.QWEN3_NEXT_80B: ModelConfig(model="qwen3-next:80b", think=True),
    LLMModelEnum.QWEN3_NEXT_80B_CLOUD: ModelConfig(
        model="qwen3-next:80b-cloud", think=True
    ),
    LLMModelEnum.QWEN3_VL_8B: ModelConfig(model="qwen3-vl:8b", think=True),
    # GLM
    LLMModelEnum.GLM_4_6_CLOUD: ModelConfig(model="glm-4.6:cloud", think=True),
    LLMModelEnum.GLM_4_7_CLOUD: ModelConfig(model="glm-4.7:cloud", think=True),
    LLMModelEnum.GLM_4_7_FLASH: ModelConfig(model="glm-4.7-flash:latest", think=True),
    LLMModelEnum.GLM_4_7_FLASH_Q4_K_M: ModelConfig(
        model="glm-4.7-flash:q4_K_M", think=True
    ),
    LLMModelEnum.GLM_4_7_FLASH_Q8_0: ModelConfig(
        model="glm-4.7-flash:q8_0", think=True
    ),
    LLMModelEnum.GLM_4_7_FLASH_BF16: ModelConfig(
        model="glm-4.7-flash:bf16", think=True
    ),
    LLMModelEnum.GLM_5_CLOUD: ModelConfig(model="glm-5:cloud", think=True),
    LLMModelEnum.GLM_5_1_CLOUD: ModelConfig(model="glm-5.1:cloud", think=True),
    # Kimi
    LLMModelEnum.KIMI_K2_THINKING_CLOUD: ModelConfig(
        model="kimi-k2-thinking:cloud", think=True
    ),
    LLMModelEnum.KIMI_K2_5_CLOUD: ModelConfig(model="kimi-k2.5:cloud", think=True),
    LLMModelEnum.KIMI_K2_6_CLOUD: ModelConfig(model="kimi-k2.6:cloud", think=True),
    # MiniMax
    LLMModelEnum.MINIMAX_M2_CLOUD: ModelConfig(model="minimax-m2:cloud", think=True),
    LLMModelEnum.MINIMAX_M2_5_CLOUD: ModelConfig(
        model="minimax-m2.5:cloud", think=True
    ),
    LLMModelEnum.MINIMAX_M2_7_CLOUD: ModelConfig(
        model="minimax-m2.7:cloud", think=True
    ),
    # Nemotron
    LLMModelEnum.NEMOTRON_3_NANO_LATEST: ModelConfig(
        model="nemotron-3-nano:latest", think=True
    ),
    LLMModelEnum.NEMOTRON_3_NANO_4B: ModelConfig(
        model="nemotron-3-nano:4b", think=True
    ),
    LLMModelEnum.NEMOTRON_3_NANO_30B: ModelConfig(
        model="nemotron-3-nano:30b", think=True
    ),
    LLMModelEnum.NEMOTRON_3_NANO_30B_CLOUD: ModelConfig(
        model="nemotron-3-nano:30b-cloud", think=True
    ),
    LLMModelEnum.NEMOTRON_3_SUPER_LATEST: ModelConfig(
        model="nemotron-3-super:latest", think=True
    ),
    LLMModelEnum.NEMOTRON_3_SUPER_120B: ModelConfig(
        model="nemotron-3-super:120b", think=True
    ),
    LLMModelEnum.NEMOTRON_3_SUPER_CLOUD: ModelConfig(
        model="nemotron-3-super:cloud", think=True
    ),
    LLMModelEnum.NEMOTRON_CASCADE_2_LATEST: ModelConfig(
        model="nemotron-cascade-2:latest", think=True
    ),
    LLMModelEnum.NEMOTRON_CASCADE_2_30B: ModelConfig(
        model="nemotron-cascade-2:30b", think=True
    ),
    # Mistral
    LLMModelEnum.MAGISTRAL_24B: ModelConfig(model="magistral:24b", think=True),
    # Google
    LLMModelEnum.GEMINI_3_FLASH_PREVIEW_LATEST: ModelConfig(
        model="gemini-3-flash-preview:latest", think=True
    ),
    LLMModelEnum.GEMINI_3_FLASH_PREVIEW_CLOUD: ModelConfig(
        model="gemini-3-flash-preview:cloud", think=True
    ),
    # OpenAI
    LLMModelEnum.GPT_OSS_20B: ModelConfig(model="gpt-oss:20b", think=True),
    LLMModelEnum.GPT_OSS_SAFEGUARD_LATEST: ModelConfig(
        model="gpt-oss-safeguard:latest", think=True
    ),
    LLMModelEnum.GPT_OSS_SAFEGUARD_20B: ModelConfig(
        model="gpt-oss-safeguard:20b", think=True
    ),
    LLMModelEnum.GPT_OSS_SAFEGUARD_120B: ModelConfig(
        model="gpt-oss-safeguard:120b", think=True
    ),
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
        self, model: LLMModelEnum, messages: list[Message], tools: list[ToolDefinition]
    ) -> Message:
        """Send messages to the Ollama client and convert the response to a Message.

        Args:
            model: The LLMModelEnum indicating which model configuration to use.
            messages: Conversation history as a list of Message objects.
            tools: Optional list of ToolDefinition objects available to the model.

        Returns:
            A Message object containing the assistant reply and any tool call info.
        """

        ollama_model = self._get_model_config(model)
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
        self, model: LLMModelEnum, messages: list[Message], tools: list[ToolDefinition]
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
        ollama_model = self._get_model_config(model)
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
                yield StreamEvent(type=SessionEventType.ASSISTANT_REASONING_DELTA, text=thinking)
            if content:
                yield StreamEvent(type=SessionEventType.ASSISTANT_STREAMING_DELTA, text=content)
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


    @staticmethod
    def _get_model_config(model: LLMModelEnum) -> ModelConfig:
        """Return the ModelConfig for the given LLMModelEnum.

        Args:
            model: The LLMModelEnum value identifying the desired model.

        Returns:
            The corresponding ModelConfig for the provided model.

        Raises:
            ValueError: If the model is not configured in OLLAMA_MODEL_CONFIG.
        """

        if model not in OLLAMA_MODEL_CONFIG:
            raise ValueError(f"Unsupported model: {model}")

        return OLLAMA_MODEL_CONFIG[model]
