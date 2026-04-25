from dataclasses import dataclass
from enum import StrEnum


class PermissionRequestKind(StrEnum):
    """Discriminator for different types of permission requests"""

    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    URL_ACCESS = "url_access"
    COMMAND = "command"
    CUSTOM_TOOL_CALL = "custom_tool_call"


@dataclass
class PermissionsRequest:
    kind: PermissionRequestKind
    """Permission kind discriminator"""

    paths: list[str] | None = None
    """File paths that may be read or written by the command"""

    urls: list[str] | None = None
    """URLs that may be accessed by the command"""

    command: str | None = None
    """Command that may be accessed"""

    tool_call_id: str | None = None
    """Tool call ID that triggered this permission request"""

    old_str: str | None = None
    """Unified diff showing the proposed changes"""

    new_str: str | None = None
    """Unified diff showing the proposed changes"""

    file_name: str | None = None
    """Path of the file being written to"""

    path: str | None = None
    """Path of the file or directory being read"""

    tool_name: str | None = None
    """Internal name of the MCP tool

    Name of the custom tool

    Name of the tool the hook is gating
    """


@dataclass
class UserInputRequest:
    question: str | None = None
    """Question that will be shown to the user"""
