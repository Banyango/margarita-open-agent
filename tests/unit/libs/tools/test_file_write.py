import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from margarita_open_agent.libs.tools.file_write import FileWriteTool
from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    PermissionRequestKind,
)


def _create_tool(permission_granted: bool) -> FileWriteTool:
    permission_service = AsyncMock(return_value=permission_granted)
    return FileWriteTool(permission_service=permission_service)


@pytest.mark.asyncio
async def test_execute_should_create_file_when_file_does_not_exist_and_old_str_is_empty(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "new_file.txt"
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute(
        {"path": str(file), "old_str": "", "new_str": "hello world"}
    )

    # Assert
    assert result == f"Created {file}"
    assert file.read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_execute_should_create_nested_file_when_parent_dirs_do_not_exist(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "a" / "b" / "new_file.txt"
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(file), "old_str": "", "new_str": "nested"})

    # Assert
    assert result == f"Created {file}"
    assert file.read_text(encoding="utf-8") == "nested"


@pytest.mark.asyncio
async def test_execute_should_return_error_when_file_does_not_exist_and_old_str_is_not_empty(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "missing.txt"
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute(
        {"path": str(file), "old_str": "something", "new_str": "other"}
    )

    # Assert
    assert result == f"Error: file not found: {file}"


@pytest.mark.asyncio
async def test_execute_should_replace_old_str_with_new_str_when_old_str_exists_exactly_once(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "edit.txt"
    file.write_text("foo bar baz", encoding="utf-8")
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(file), "old_str": "bar", "new_str": "qux"})

    # Assert
    assert result == f"Edited {file}"
    assert file.read_text(encoding="utf-8") == "foo qux baz"


@pytest.mark.asyncio
async def test_execute_should_return_error_when_old_str_is_not_found_in_file(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "edit.txt"
    file.write_text("foo bar baz", encoding="utf-8")
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(file), "old_str": "zzz", "new_str": "qux"})

    # Assert
    assert result == f"Error: old_str not found in {file}"


@pytest.mark.asyncio
async def test_execute_should_return_error_when_old_str_appears_more_than_once(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "edit.txt"
    file.write_text("foo foo foo", encoding="utf-8")
    tool = _create_tool(permission_granted=True)

    # Act
    result = await tool.execute({"path": str(file), "old_str": "foo", "new_str": "bar"})

    # Assert
    assert result == f"Error: old_str found 3 times in {file} — must be unique"


@pytest.mark.asyncio
async def test_execute_should_return_denied_message_when_permission_is_denied_for_edit(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "edit.txt"
    file.write_text("original", encoding="utf-8")
    tool = _create_tool(permission_granted=False)

    # Act
    result = await tool.execute(
        {"path": str(file), "old_str": "original", "new_str": "replaced"}
    )

    # Assert
    assert result == "Request was denied by the user"
    assert file.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_execute_should_return_denied_message_when_permission_is_denied_for_create(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "new_file.txt"
    tool = _create_tool(permission_granted=False)

    # Act
    result = await tool.execute({"path": str(file), "old_str": "", "new_str": "hello"})

    # Assert
    assert result == "Request was denied by the user"
    assert not file.exists()


@pytest.mark.asyncio
async def test_execute_should_request_permission_with_correct_metadata_when_editing(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "edit.txt"
    file.write_text("old content", encoding="utf-8")
    permission_service = AsyncMock(return_value=True)
    tool = FileWriteTool(permission_service=permission_service)

    # Act
    await tool.execute(
        {"path": str(file), "old_str": "old content", "new_str": "new content"}
    )

    # Assert
    permission_service.assert_called_once_with(
        PermissionsRequest(
            kind=PermissionRequestKind.FILE_WRITE,
            paths=[str(file.absolute())],
            old_str="old content",
            new_str="new content",
            tool_name="file_write",
        )
    )


@pytest.mark.asyncio
async def test_execute_should_request_permission_with_correct_metadata_when_creating(
    tmp_path: Path,
):
    # Arrange
    file = tmp_path / "new_file.txt"
    permission_service = AsyncMock(return_value=True)
    tool = FileWriteTool(permission_service=permission_service)

    # Act
    await tool.execute({"path": str(file), "old_str": "", "new_str": "content"})

    # Assert
    permission_service.assert_called_once_with(
        PermissionsRequest(
            kind=PermissionRequestKind.FILE_WRITE,
            paths=[str(file.absolute())],
            new_str="content",
            tool_name="file_write",
        )
    )
