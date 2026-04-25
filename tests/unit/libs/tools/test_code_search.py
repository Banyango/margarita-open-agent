import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from margarita_open_agent.libs.tools.code_search import CodeSearchTool


def _create_tool() -> CodeSearchTool:
    return CodeSearchTool()


def _make_proc(stdout: bytes, returncode: int) -> MagicMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    return proc


@pytest.mark.asyncio
async def test_execute_should_return_matches_when_rg_finds_results():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"src/foo.py:10:def hello():", returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        # Act
        result = await tool.execute({"pattern": "hello"})

    # Assert
    assert "src/foo.py:10:def hello():" in result


@pytest.mark.asyncio
async def test_execute_should_return_no_matches_found_when_rg_exits_with_1():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"", returncode=1)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        # Act
        result = await tool.execute({"pattern": "nonexistent"})

    # Assert
    assert result == "No matches found"


@pytest.mark.asyncio
async def test_execute_should_return_error_message_when_rg_exits_with_non_zero():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"", returncode=2)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        # Act
        result = await tool.execute({"pattern": "hello"})

    # Assert
    assert result == "Search failed (exit 2)"


@pytest.mark.asyncio
async def test_execute_should_pass_ignore_case_flag_when_case_sensitive_is_false():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"match", returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
    ):
        # Act
        await tool.execute({"pattern": "hello", "case_sensitive": False})

    # Assert
    args = mock_exec.call_args[0]
    assert "--ignore-case" in args


@pytest.mark.asyncio
async def test_execute_should_not_pass_ignore_case_flag_when_case_sensitive_is_true():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"match", returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
    ):
        # Act
        await tool.execute({"pattern": "hello", "case_sensitive": True})

    # Assert
    args = mock_exec.call_args[0]
    assert "--ignore-case" not in args


@pytest.mark.asyncio
async def test_execute_should_pass_file_type_flag_when_file_type_is_provided():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"match", returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/rg"),
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
    ):
        # Act
        await tool.execute({"pattern": "hello", "file_type": "py"})

    # Assert
    args = mock_exec.call_args[0]
    assert "--type" in args
    assert "py" in args


@pytest.mark.asyncio
async def test_execute_should_use_grep_when_rg_is_not_available():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"src/foo.py:10:def hello():", returncode=0)

    with (
        patch("shutil.which", return_value=None),
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
    ):
        # Act
        result = await tool.execute({"pattern": "hello"})

    # Assert
    args = mock_exec.call_args[0]
    assert args[0] == "grep"
    assert "src/foo.py:10:def hello():" in result


@pytest.mark.asyncio
async def test_execute_should_return_no_matches_found_when_grep_exits_with_1():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"", returncode=1)

    with (
        patch("shutil.which", return_value=None),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        # Act
        result = await tool.execute({"pattern": "nonexistent"})

    # Assert
    assert result == "No matches found"


@pytest.mark.asyncio
async def test_execute_should_filter_by_file_extension_when_file_type_is_provided_via_grep():
    # Arrange
    tool = _create_tool()
    proc = _make_proc(b"match", returncode=0)

    with (
        patch("shutil.which", return_value=None),
        patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec,
    ):
        # Act
        await tool.execute({"pattern": "hello", "file_type": "py"})

    # Assert
    args = mock_exec.call_args[0]
    assert "--include=*.py" in args
