import pytest
from unittest.mock import AsyncMock

from margarita_open_agent.core.models.permissions import UserInputRequest
from margarita_open_agent.libs.tools.ask_user import AskUserTool


def _create_tool(response: str | None = "some response") -> AskUserTool:
    handler = AsyncMock(return_value=response)
    return AskUserTool(user_input_handler=handler)


@pytest.mark.asyncio
async def test_execute_should_return_user_response_when_handler_returns_a_string():
    # Arrange
    tool = _create_tool(response="yes, proceed")

    # Act
    result = await tool.execute({"question": "Should I continue?"})

    # Assert
    assert result == "yes, proceed"


@pytest.mark.asyncio
async def test_execute_should_return_empty_string_when_handler_returns_none():
    # Arrange
    tool = _create_tool(response=None)

    # Act
    result = await tool.execute({"question": "Are you there?"})

    # Assert
    assert result == ""


@pytest.mark.asyncio
async def test_execute_should_return_empty_string_when_question_is_missing():
    # Arrange
    tool = _create_tool(response=None)

    # Act
    result = await tool.execute({})

    # Assert
    assert result == ""


@pytest.mark.asyncio
async def test_execute_should_call_handler_with_correct_request_when_question_is_provided():
    # Arrange
    handler = AsyncMock(return_value="ok")
    tool = AskUserTool(user_input_handler=handler)

    # Act
    await tool.execute({"question": "What is your name?"})

    # Assert
    handler.assert_called_once_with(UserInputRequest(question="What is your name?"))


@pytest.mark.asyncio
async def test_execute_should_coerce_non_string_response_to_string_when_handler_returns_non_string():
    # Arrange
    handler = AsyncMock(return_value=42)
    tool = AskUserTool(user_input_handler=handler)

    # Act
    result = await tool.execute({"question": "Pick a number"})

    # Assert
    assert result == "42"
