import pytest

from margarita_open_agent.libs.tools.file_find import FileFindTool


@pytest.mark.asyncio
async def test_execute_should_return_matching_files_when_pattern_matches(tmp_path):
    # Arrange
    (tmp_path / "foo.py").write_text("foo")
    (tmp_path / "bar.py").write_text("bar")
    (tmp_path / "baz.txt").write_text("baz")
    tool = FileFindTool()

    # Act
    result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"})

    # Assert
    assert str(tmp_path / "bar.py") in result
    assert str(tmp_path / "foo.py") in result
    assert "baz.txt" not in result


@pytest.mark.asyncio
async def test_execute_should_return_no_files_found_when_pattern_does_not_match(
    tmp_path,
):
    # Arrange
    (tmp_path / "foo.txt").write_text("foo")
    tool = FileFindTool()

    # Act
    result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"})

    # Assert
    assert result == "No files found."


@pytest.mark.asyncio
async def test_execute_should_search_recursively_when_files_are_in_subdirectories(
    tmp_path,
):
    # Arrange
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "nested.py").write_text("nested")
    tool = FileFindTool()

    # Act
    result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"})

    # Assert
    assert str(subdir / "nested.py") in result


@pytest.mark.asyncio
async def test_execute_should_return_no_files_found_when_directory_does_not_exist(
    tmp_path,
):
    # Arrange
    missing = tmp_path / "nonexistent"
    tool = FileFindTool()

    # Act
    result = await tool.execute({"directory": str(missing), "pattern": "*.py"})

    # Assert
    assert result == "No files found."


@pytest.mark.asyncio
async def test_execute_should_return_no_files_found_when_path_is_not_a_directory(
    tmp_path,
):
    # Arrange
    file_path = tmp_path / "a_file.txt"
    file_path.write_text("content")
    tool = FileFindTool()

    # Act
    result = await tool.execute({"directory": str(file_path), "pattern": "*.py"})

    # Assert
    assert result == "No files found."
