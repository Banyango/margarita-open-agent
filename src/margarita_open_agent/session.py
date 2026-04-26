import asyncio
from collections.abc import AsyncIterator

from margarita_open_agent.core.interfaces import (
    PermissionCallbackHandler,
    UserInputCallbackHandler,
    UserToolCallbackHandler,
)
from margarita_open_agent.libs.tools.registry import ToolRegistry

from margarita_open_agent.container import container
from margarita_open_agent.core.llm import LLMClient
from margarita_open_agent.core.models.llm_model_enum import LLMModelEnum
from margarita_open_agent.core.models.message import Message
from margarita_open_agent.core.models.stream_event import StreamEvent
from margarita_open_agent.core.models.tool import ToolDefinition
from margarita_open_agent.core.models.tool_call_event import (
    ToolCallCallingMetadata,
    ToolCallDoneMetadata,
)
from margarita_open_agent.session_event import SessionEventType


class AgentSession:
    """Represents an agent session which runs prompts against an LLM and manages messages and tool calls.

    The session stores the conversation history, invokes the configured LLM client, and
    executes tools via the provided tool executor callback. Consumers should use
    :meth:`send_and_wait_async` to send a prompt and await the assistant's final reply.
    """

    def __init__(
        self,
        model: LLMModelEnum,
        system_message: str,
        additional_tools: list[ToolDefinition],
        on_user_input_request: UserInputCallbackHandler,
        on_permission_request: PermissionCallbackHandler,
        on_custom_tool_request: UserToolCallbackHandler | None,
    ):
        self.model = model
        self.system_message = system_message
        self.additional_tools = additional_tools
        self.on_user_input_request = on_user_input_request
        self.on_permission_request = on_permission_request
        self.on_custom_tool_request = on_custom_tool_request
        self._messages: list[Message] = [Message(role="system", content=system_message)]

    async def send_and_wait_async(self, prompt: str, timeout: int = None) -> str:
        """Send a prompt to the agent and wait for the assistant's final response.

        This method is a thin wrapper around the internal asynchronous runner and
        supports an optional timeout (in seconds). It returns the assistant's
        content string when the conversation step completes.

        Args:
            prompt: The user prompt to send to the agent.
            timeout: Optional timeout in seconds for the entire operation.

        Returns:
            The assistant's reply content as a string.
        """

        return await asyncio.wait_for(self._run_async(prompt), timeout=timeout)

    async def send_and_stream_async(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Send a prompt and stream the assistant's reply as ``StreamEvent`` objects.

        Tool calls are handled transparently via non-streaming chat round-trips.
        The final plain-text response is then streamed, yielding both
        ``"thinking"`` and ``"content"`` events.

        Args:
            prompt: The user prompt to send to the agent.

        Yields:
            :class:`StreamEvent` instances with ``type`` set to either
            ``"thinking"`` or ``"content"``.
        """
        container.override.set(PermissionCallbackHandler, self.on_permission_request)
        container.override.set(UserInputCallbackHandler, self.on_user_input_request)

        if self.on_custom_tool_request:
            container.override.set(UserToolCallbackHandler, self.on_custom_tool_request)

        llm_client = await container.get(LLMClient)
        tool_executor = await container.get(ToolRegistry)

        tools = tool_executor.get_tool_definitions() + self.additional_tools

        self._messages.append(Message(role="user", content=prompt))

        while True:
            yield StreamEvent(type="status", text="Thinking...")
            response = await llm_client.chat(self.model, self._messages, tools)

            if not response.get("tool_calls"):
                full_content = ""
                async for event in llm_client.stream(self.model, self._messages, tools):
                    if event.type == "content":
                        full_content += event.text
                    yield event
                self._messages.append(Message(role="assistant", content=full_content))
                return

            self._messages.append(response)

            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                yield StreamEvent(
                    type=SessionEventType.TOOL_EXECUTION_START,
                    text=f"Calling tool: {name}",
                    metadata=ToolCallCallingMetadata(name=name, arguments=arguments),
                )

                try:
                    result = await tool_executor.execute(name, arguments)

                    yield StreamEvent(
                        type=SessionEventType.TOOL_EXECUTION_COMPLETE,
                        text=f"Tool {name} finished",
                        metadata=ToolCallDoneMetadata(
                            name=name, arguments=arguments, result=result, success=True
                        ),
                    )
                    self._messages.append(Message(role="tool", content=result))
                finally:
                    yield StreamEvent(
                        type=SessionEventType.TOOL_EXECUTION_COMPLETE,
                        text=f"Tool {name} execution failed",
                        metadata=ToolCallDoneMetadata(
                            name=name, arguments=arguments, success=False, result=None
                        ),
                    )

    async def _run_async(self, prompt: str) -> str:
        """Internal runner that sends the prompt to the LLM, handles tool calls,

        Args:
            prompt: The user prompt to process.

        Returns:
            The assistant's final reply content as a string.
        """
        container.override.set(PermissionCallbackHandler, self.on_permission_request)
        container.override.set(UserInputCallbackHandler, self.on_user_input_request)
        if self.on_custom_tool_request:
            container.override.set(UserToolCallbackHandler, self.on_custom_tool_request)

        llm_client = await container.get(LLMClient)
        tool_executor = await container.get(ToolRegistry)

        self._messages.append(Message(role="user", content=prompt))

        tools = tool_executor.get_tool_definitions() + self.additional_tools

        while True:
            response = await llm_client.chat(self.model, self._messages, tools)

            if not response.get("tool_calls"):
                self._messages.append(response)
                return response.get("content", "")

            self._messages.append(response)

            for tool_call in response["tool_calls"]:
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                result = await tool_executor.execute(name, arguments)
                self._messages.append(Message(role="tool", content=result))
