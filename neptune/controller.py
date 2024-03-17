from typing import Any

from neptune.middleware.middleware import apply_middleware
from neptune.middleware.pointer import PointerMiddleware
from neptune.middleware.truncated import TruncatedMiddleware
from neptune.model.dns.message import DNSMessage


class DNSController:
    def __init__(self, config):
        self.config = config
        self.ctx = {}

    @apply_middleware(TruncatedMiddleware)
    @apply_middleware(PointerMiddleware)
    def process_message(self, message: DNSMessage, ctx: dict[str, Any]) -> DNSMessage:
        """Processes message and returns a response message."""

        return message
