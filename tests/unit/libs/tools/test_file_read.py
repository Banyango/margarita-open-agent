import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from margarita_open_agent.libs.tools.file_read import FileReadTool
from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    PermissionRequestKind,
)


def _create_tool(permission_granted: bool) -> FileReadTool:
    permission_service = AsyncMock(return_value=permission_granted)
    return FileReadTool(permission_service=permission_service)


@pytest.mark.asyncio
async def test_execute_should_return_file_contents_when_permission_is_granted(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "hello.txt"
    file.write_text("hello world", encoding="utf-8")
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(file)})

    # Assert
    assert result == "hello world"


@pytest.mark.asyncio
async def test_execute_should_request_permission_with_correct_metadata_when_called(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "hello.txt"
    file.write_text("data", encoding="utf-8")
    permission_service = AsyncMock(return_value=True)
    tool = FileReadTool(permission_service=permission_service)

    # Act
    await tool.execute({"path": str(file)})

    # Assert
    permission_service.assert_called_once_with(
        PermissionsRequest(
            kind=PermissionRequestKind.FILE_READ,
            paths=[str(file.absolute())],
            tool_name="file_read",
        )
    )


@pytest.mark.asyncio
async def test_execute_should_return_denied_message_when_permission_is_denied(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "secret.txt"
    file.write_text("secret", encoding="utf-8")
    tool = _create_tool(permission_granted=False)

    # Act
    result = await tool.execute({"path": str(file)})

    # Assert
    assert result == "Request was denied by the user"


@pytest.mark.asyncio
async def test_execute_should_return_error_message_when_file_not_found():
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": "/nonexistent/path/file.txt"})

    # Assert
    assert "file not found" in result


@pytest.mark.asyncio
async def test_execute_should_return_error_message_when_path_is_a_directory(
    tmp_path: Path,
):
    # Arrange
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(tmp_path)})

    # Assert
    assert "path is a directory" in result


@pytest.mark.asyncio
async def test_execute_should_return_error_message_when_os_error_occurs(tmp_path: Path):
    # Arrange
    file = tmp_path / "file.txt"
    file.write_text("data", encoding="utf-8")
    tool = _create_tool(permission_granted=True)

    # Act
    with patch.object(Path, "read_text", side_effect=OSError("disk error")):
        result = await tool.execute({"path": str(file)})

    # Assert
    assert "Error reading" in result
    assert "disk error" in result
