import pytest
from unittest.mock import AsyncMock

from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    PermissionRequestKind,
)
from margarita_open_agent.libs.tools.command_runner import CommandRunnerTool


def _create_tool(permission_granted: bool = True) -> CommandRunnerTool:
    permission_service = AsyncMock(return_value=permission_granted)
    tool = CommandRunnerTool(permission_service=permission_service)
    return tool


@pytest.mark.asyncio
async def test_execute_should_return_output_when_command_succeeds():
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"command": "echo hello"})

    # Assert
    assert result == "hello"


@pytest.mark.asyncio
async def test_execute_should_return_no_output_when_command_produces_no_output():
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"command": "true"})

    # Assert
    assert result == "(no output)"


@pytest.mark.asyncio
async def test_execute_should_return_error_message_when_command_fails():
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"command": "exit 1"})

    # Assert
    assert result.startswith("Command failed (exit 1)")


@pytest.mark.asyncio
async def test_execute_should_combine_stderr_with_stdout_when_command_writes_to_stderr():
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"command": "echo error_output >&2"})

    # Assert
    assert "error_output" in result


@pytest.mark.asyncio
async def test_execute_should_return_denied_message_when_permission_is_rejected():
    # Arrange
    tool = _create_tool(permission_granted=False)

    # Act
    result = await tool.execute({"command": "echo hello"})

    # Assert
    assert result == "Request was denied by the user"


@pytest.mark.asyncio
async def test_execute_should_request_permission_with_correct_details_when_called():
    # Arrange
    permission_service = AsyncMock(return_value=True)
    tool = CommandRunnerTool(permission_service=permission_service)

    # Act
    await tool.execute({"command": "echo hello"})

    # Assert
    permission_service.assert_called_once_with(
        PermissionsRequest(
            kind=PermissionRequestKind.COMMAND,
            command="echo hello",
            tool_name="command_runner",
        )
    )
