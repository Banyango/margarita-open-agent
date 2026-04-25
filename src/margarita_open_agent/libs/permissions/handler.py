from wireup import injectable

from margarita_open_agent.core.interfaces import PermissionCallbackHandler
from margarita_open_agent.core.models.permissions import PermissionsRequest


@injectable(as_type=PermissionCallbackHandler)
class DefaultPermissionCallbackHandler(PermissionCallbackHandler):
    async def __call__(self, request: PermissionsRequest) -> bool:
        return True
