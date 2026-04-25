"""
Integration tests — require a running Ollama instance with gemma3 pulled.

Run with:
    uv run pytest tests/integration -m integration -v
"""

import pytest
from unittest.mock import AsyncMock, patch
from ollama import AsyncClient

from margarita_open_agent.core.models.llm_model_enum import LLMModelEnum
from margarita_open_agent.libs.ollama.ollama import OllamaLLMClient
from margarita_open_agent.libs.tools import registry
from margarita_open_agent.session import AgentSession


def make_session(llm_client: OllamaLLMClient) -> tuple[AgentSession, patch]:
    """Return a session wired to the given client, with container.get patched."""
    mock_container = patch("margarita_open_agent.session.container")
    container = mock_container.start()
    container.get = AsyncMock(return_value=llm_client)
    session = AgentSession(
        model=LLMModelEnum.GEMMA_4,
        system_message="You are a coding agent. Use tools whenever you need information from the filesystem.",
        additional_tools=registry.get_definitions(),
        tool_executor=registry.execute,
        on_user_input_request=AsyncMock(),
        on_permission_request=AsyncMock(),
    )
    return session, mock_container


@pytest.fixture
def llm_client() -> OllamaLLMClient:
    return OllamaLLMClient(AsyncClient())


@pytest.mark.integration
async def test_basic_chat(llm_client):
    """Model responds without using any tools."""
    response = await llm_client.chat(
        LLMModelEnum.GEMMA_4,
        [{"role": "user", "content": "Reply with exactly one word: hello"}],
        [],
    )
    assert response["role"] == "assistant"
    assert response["content"].strip()


@pytest.mark.integration
async def test_file_read_tool(llm_client, tmp_path):
    """Agent reads a file via file_read and includes the contents in its reply."""
    secret = tmp_path / "secret.txt"
    secret.write_text("xkcd-correct-horse-battery-staple")

    session, patcher = make_session(llm_client)
    try:
        result = await session.send_and_wait_async(
            f"Read the file at {secret} and tell me exactly what it contains.",
            timeout=120,
        )
    finally:
        patcher.stop()

    assert "xkcd-correct-horse-battery-staple" in result


@pytest.mark.integration
async def test_file_write_tool(llm_client, tmp_path):
    """Agent creates a new file via file_write."""
    target = tmp_path / "output.txt"

    session, patcher = make_session(llm_client)
    try:
        await session.send_and_wait_async(
            f"Create a file at {target} containing only the text: agent-was-here",
            timeout=120,
        )
    finally:
        patcher.stop()

    assert target.exists()
    assert "agent-was-here" in target.read_text()


@pytest.mark.integration
async def test_command_runner_tool(llm_client):
    """Agent runs a shell command and reports the output."""
    session, patcher = make_session(llm_client)
    try:
        result = await session.send_and_wait_async(
            "Run the command `echo integration-test-marker` and tell me what it printed.",
            timeout=120,
        )
    finally:
        patcher.stop()

    assert "integration-test-marker" in result


@pytest.mark.integration
async def test_code_search_tool(llm_client):
    """Agent searches the codebase and reports what it found."""
    session, patcher = make_session(llm_client)
    try:
        result = await session.send_and_wait_async(
            "Search the src directory for the text 'LLMModelEnum' and tell me which files contain it.",
            timeout=120,
        )
    finally:
        patcher.stop()

    assert result.strip()
