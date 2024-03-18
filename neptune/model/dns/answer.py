from typing import Annotated, Any, Self, Union

from pydantic import Field

from neptune.model.dns.common import BaseDNSModel, SupportedQueryClasses
from neptune.model.dns.data import DATA_MAP, DNSDataType, SupportedQueryTypes
from neptune.types import DomainName, Pointer, int16, uint16, uint32


class DNSMessageAnswer(BaseDNSModel):
    """A DNS dns answer."""

    name: Annotated[
        Union[Pointer, DomainName],
        Field(description="The domain name that was queried."),
    ]

    type: Annotated[
        uint16,
        SupportedQueryTypes,
        Field(description="Type of the answer.", alias="TYPE"),
    ]

    cls: Annotated[uint16, SupportedQueryClasses, Field(description="Class of the answer.")]

    ttl: uint32 = Field(..., description="Time to live for the answer.", alias="TTL")

    data_length: int16 = Field(..., description="Length of the data.", alias="RDLENGTH")

    data: DNSDataType = Field(
        ...,
        description="Data.",
        alias="rdata",
        json_schema_extra={
            "length": "RDLENGTH",
            "$type": {
                "getter": "get_rdata_type",
                "arg": "TYPE",
            },
        },
    )

    @classmethod
    def get_rdata_type(cls, type: uint16) -> DNSDataType:
        """Get the data type for a given type."""

        return DATA_MAP[type.value]

    @classmethod
    def get_mocked(cls, override: dict[str, Any] | None = None) -> Self:
        """Get a mocked instance of answer."""

        from neptune.model.dns.data import A, IPv4Address

        kwargs = {
            "name": "example.com",
            "type": 1,
            "cls": 1,
            "ttl": 3600,
            "data_length": 4,
            "data": A(data=IPv4Address(address="127.0.0.1")),
        }

        kwargs.update(override or {})

        return cls(**kwargs)
