import typing
from abc import ABC, abstractmethod
from functools import wraps

if typing.TYPE_CHECKING:
    from neptune.controller import DNSController
    from neptune.model.dns.message import DNSMessage


class Middleware(ABC):
    def __init__(self, ctx: dict):
        self.ctx = ctx

    @abstractmethod
    def process_before(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        pass

    @abstractmethod
    def process_after(self, message: "DNSMessage", *args, **kwargs) -> "DNSMessage":
        pass


def apply_middleware(middleware_cls: type[Middleware]):
    """Apply middleware to a controller method."""

    def decorator(func):
        @wraps(func)
        def wrapper(
            controller: "DNSController", message: "DNSMessage", ctx: dict, *args, **kwargs
        ):
            middleware = middleware_cls(ctx=ctx)

            message = middleware.process_before(message, *args, **kwargs)

            result = func(controller, message, ctx, *args, **kwargs)

            result = middleware.process_after(result, *args, **kwargs)

            return result

        return wrapper

    return decorator
