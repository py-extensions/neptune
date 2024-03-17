import typing

from neptune.middleware.middleware import Middleware

if typing.TYPE_CHECKING:
    from neptune.model.dns.message import DNSMessage


class PointerMiddleware(Middleware):
    """Middleware that shortens messages by replacing reused domain names with pointers."""

    def process_before(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        return message

    def process_after(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        return message
