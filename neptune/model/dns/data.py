from enum import Enum
from typing import ClassVar, Union

from pydantic import Field

from neptune.model.dns.common import BaseDNSModel
from neptune.types import UNDEFINED, DomainName, IPv4Address, IPv6Address, String, int32


class DNSData(BaseDNSModel):
    """Base class for DNS data."""

    ID: ClassVar[int]

    LENGTH: ClassVar = UNDEFINED


class CNAME(DNSData):
    """A canonical name."""

    ID: ClassVar[int] = 5

    data: DomainName = Field(..., description="The canonical name.")


class A(DNSData):
    """An A record."""

    ID: ClassVar[int] = 1

    data: IPv4Address = Field(..., description="The IPv4 address.")


class NS(DNSData):
    """A name server."""

    ID: ClassVar[int] = 2

    data: DomainName = Field(..., description="The name server.")


class SOA(DNSData):
    """A start of authority."""

    ID: ClassVar[int] = 6

    name_server: DomainName = Field(..., description="The primary name server.", alias="mname")
    mailbox: DomainName = Field(
        ..., description="The responsible authority's mailbox.", alias="rname"
    )
    serial: int32 = Field(..., description="The serial number of the zone.")
    refresh: int32 = Field(..., description="The refresh interval.")
    retry: int32 = Field(..., description="The retry interval.")
    expire: int32 = Field(..., description="The expiration limit.")
    minimum: int32 = Field(..., description="The minimum TTL.")


class PTR(DNSData):
    """A pointer record."""

    ID: ClassVar[int] = 12

    data: DomainName = Field(..., description="The domain name.")


class MX(DNSData):
    """A mail exchange."""

    ID: ClassVar[int] = 15

    preference: int32 = Field(..., description="The preference value.")
    exchange: DomainName = Field(..., description="The mail exchange server.")


class TXT(DNSData):
    """A text record."""

    ID: ClassVar[int] = 16

    data: String = Field(..., description="The text data.")


class AAAA(DNSData):
    """An AAAA record."""

    ID: ClassVar[int] = 28

    data: IPv6Address = Field(..., description="The IPv6 address.")


class Option(DNSData):
    """An option record."""

    ID: ClassVar[int] = 41

    data: String = Field(..., description="The option data.")


# Special helpers
DNSDataType = Union[*[cls for cls in DNSData.__subclasses__()]]


SupportedQueryTypes = Enum(
    "SupportedQueryTypes", {cls.__name__: cls for cls in DNSData.__subclasses__()}
)

DATA_MAP = {cls.ID: cls for cls in DNSData.__subclasses__()}
