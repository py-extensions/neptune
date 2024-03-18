import typing

from neptune.middleware.middleware import Middleware

if typing.TYPE_CHECKING:
    from neptune.model.dns.message import DNSMessage


class TruncatedMiddleware(Middleware):
    """Middleware that waits for second part of message to arrive before processing it.

    Also if the output message is too long, it will be truncated.
    """

    def process_before(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        if message.is_truncated:
            return None
        return message

    def process_after(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        return message
