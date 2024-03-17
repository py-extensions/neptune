import uuid
from enum import Enum
from typing import Any, ClassVar


class SupportedProtocol(Enum):
    """Supported protocols."""

    HTTPS: str = "https"
    TCP: str = "tcp"
    UDP: str = "udp"


class Protocol:
    NAME: ClassVar[SupportedProtocol]
    CONTEXT: ClassVar[dict[str, dict[str, Any]]] = {}

    def generate_context(self):
        """Generates new context."""

        context_id = str(uuid.uuid4())

        self.CONTEXT[context_id] = {
            "id": context_id,
            "protocol": self.NAME,
        }

        return self.CONTEXT[context_id]

    def delete_context(self, key):
        """Deletes context by key."""

        self.CONTEXT.pop(key, None)
