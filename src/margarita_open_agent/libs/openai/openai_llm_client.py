import json
import uuid
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from wireup import injectable

from margarita_open_agent.core.llm import LLMClient
from margarita_open_agent.core.models.errors import ModelNotSpecifiedError
from margarita_open_agent.core.models.message import Message
from margarita_open_agent.core.models.stream_event import StreamEvent
from margarita_open_agent.core.models.tool import ToolDefinition
from margarita_open_agent.core.models.tool_call_event import ToolCallCallingMetadata
from margarita_open_agent.core.sessions.session_event import SessionEventType


@injectable(as_type=LLMClient, qualifier="openai")
class OpenAILLMClient(LLMClient):
    def __init__(self, client: AsyncOpenAI):
        """Initialize the OpenAI LLM client wrapper.

        Args:
            client: An instance of openai.OpenAI used to perform chat and streaming.
        """

        self.client = client

    async def chat(
        self, model: str, messages: list[Message], tools: list[ToolDefinition]
    ) -> Message:
        """Send messages to the OpenAI client and convert the response to a Message.

        This implementation uses the OpenAI Python SDK's chat.completions.create
        (via client.chat.completions.create) and supports function-calling tools
        by passing them as `functions` and returning tool_calls if present.
        """
        if model == "" or model is None:
            raise ModelNotSpecifiedError()

        # Convert internal Message objects to OpenAI chat format
        openai_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or m.get("thinking") or ""
            # Internal 'tool' role corresponds to OpenAI's 'function' role.
            if role == "tool":
                # include the tool name if available
                openai_messages.append(
                    {"role": "function", "name": m.get("tool_name"), "content": content}
                )
            else:
                openai_messages.append({"role": role, "content": content})

        # Convert tool definitions to OpenAI functions
        functions = []
        for t in (tools or []):
            func = t.get("function")
            if not func:
                continue
            functions.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters"),
                }
            )

        # Use the async client's create method directly
        response = await self.client.chat.completions.create(
            model=model, messages=openai_messages, functions=functions or None
        )

        # OpenAI returns choices; take the first
        choice = response.choices[0]
        message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)

        # If the model returned a function_call, expose it via tool_calls
        tool_calls = []
        if message is not None and (getattr(message, "function_call", None) or (isinstance(message, dict) and message.get("function_call"))):
            fc = getattr(message, "function_call", None) or (message.get("function_call") if isinstance(message, dict) else None)
            # arguments may be a JSON string
            args = fc.get("arguments") if isinstance(fc, dict) else getattr(fc, "arguments", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}

            name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", None)
            tool_calls.append({"function": {"name": name, "arguments": args or {}}})

        return Message(
            role=getattr(message, "role", "assistant"),
            content=getattr(message, "content", "") or "",
            tool_calls=tool_calls,
        )

    async def stream(
        self, model: str, messages: list[Message], tools: list[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        """Stream content events from OpenAI. OpenAI streams partial deltas which
        we map to ASSISTANT_STREAMING_DELTA. Function-calls are surfaced as TOOL_REQUESTED
        events when a final function call is produced.
        """
        if model == "" or model is None:
            raise ModelNotSpecifiedError()

        openai_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or m.get("thinking") or ""
            if role == "tool":
                openai_messages.append(
                    {"role": "function", "name": m.get("tool_name"), "content": content}
                )
            else:
                openai_messages.append({"role": role, "content": content})

        functions = []
        for t in (tools or []):
            func = t.get("function")
            if not func:
                continue
            functions.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters"),
                }
            )

        # Use the streaming API. The async client returns an async context manager
        # that yields chunks; iterate with `async for` over the stream contents.
        async with self.client.chat.completions.stream(
            model=model, messages=openai_messages, functions=functions or None
        ) as stream:
            async for chunk in stream:
                # Each chunk may contain delta content. Normalize access for both
                # dict-like and attribute-like SDK objects.
                choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else [])
                for choice in choices:
                    delta = getattr(choice, "delta", None) or (choice.get("delta") if isinstance(choice, dict) else {})

                    # content delta (some SDKs use 'content' others 'text')
                    content = None
                    if isinstance(delta, dict):
                        content = delta.get("content") or delta.get("text")
                    else:
                        content = getattr(delta, "content", None) or getattr(delta, "text", None)

                    if content:
                        yield StreamEvent(
                            type=SessionEventType.ASSISTANT_STREAMING_DELTA, text=content
                        )

                    # function_call may be present in delta
                    function_call = None
                    if isinstance(delta, dict):
                        function_call = delta.get("function_call")
                    else:
                        function_call = getattr(delta, "function_call", None)

                    if function_call:
                        # If function_call contains name/arguments, surface as tool request
                        if isinstance(function_call, dict):
                            name = function_call.get("name")
                            arguments = function_call.get("arguments")
                        else:
                            name = getattr(function_call, "name", None)
                            arguments = getattr(function_call, "arguments", None)

                        # arguments may be a JSON string
                        if isinstance(arguments, str):
                            try:
                                args_parsed = json.loads(arguments)
                            except Exception:
                                args_parsed = {"raw": arguments}
                        else:
                            args_parsed = arguments or {}

                        yield StreamEvent(
                            type=SessionEventType.TOOL_REQUESTED,
                            text=f"Tool call: {name}",
                            metadata=ToolCallCallingMetadata(
                                tool_call_id=str(uuid.uuid1()),
                                name=name,
                                arguments=args_parsed,
                            ),
                        )
