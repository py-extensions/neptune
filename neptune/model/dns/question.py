from typing import Annotated, Any, Self, Union

from pydantic import Field

from neptune.model.dns.common import (
    BaseDNSModel,
    SupportedQueryClasses,
)
from neptune.model.dns.data import SupportedQueryTypes
from neptune.types import DomainName, Pointer, uint16


class DNSMessageQuestion(BaseDNSModel):
    """A DNS dns question."""

    domain: Annotated[
        Union[Pointer, DomainName],
        Field(description="The domain name being queried.", alias="qname"),
    ]

    type: Annotated[
        uint16,
        SupportedQueryTypes,
        Field(description="Type of the question.", alias="qtype"),
    ]

    cls: Annotated[
        uint16,
        SupportedQueryClasses,
        Field(description="Class of the question.", alias="qclass"),
    ]

    @classmethod
    def get_mocked(cls, override: dict[str, Any] | None = None) -> Self:
        """Get a mocked instance of question."""

        return cls(
            domain="example.com",
            type=1,
            cls=1,
            **(override or {}),
        )
