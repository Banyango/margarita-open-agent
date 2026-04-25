from abc import ABC, abstractmethod

from margarita_open_agent.core.models.permissions import (
    PermissionsRequest,
    UserInputRequest,
)
from margarita_open_agent.core.models.tool import ToolCallRequest


class PermissionCallbackHandler(ABC):
    @abstractmethod
    async def __call__(self, request: PermissionsRequest) -> bool:
        """Ask the user for permission to perform an action.

        Args:
            request (PermissionsRequest): The permission request containing details about the action.

        Returns:
            True if the permission is granted, False otherwise.
        """


class UserInputCallbackHandler(ABC):
    @abstractmethod
    async def __call__(self, request: UserInputRequest) -> str:
        """Ask the user for permission to perform an action.

        Args:
            request (UserInputRequest): The permission request containing details about the action.
        """


class UserToolCallbackHandler(ABC):
    @abstractmethod
    async def __call__(self, request: ToolCallRequest) -> str:
        """This is called when a user added tool call is requested.

        Args:
            request (ToolCallRequest): The tool call request containing details about the tool and arguments.
        """
