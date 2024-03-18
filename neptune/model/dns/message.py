from typing import Any, List, Optional, Self

from pydantic import Field

from neptune.model.dns.answer import DNSMessageAnswer
from neptune.model.dns.common import BaseDNSModel
from neptune.model.dns.question import DNSMessageQuestion
from neptune.types import boolean, int3, uint4, uint16


class DNSMessage(BaseDNSModel):
    """A DNS dns."""

    id: uint16 = Field(
        ...,
        description=(
            "A 16 bit identifier assigned by the program that generates any kind of query. "
            "This identifier is copied the corresponding reply "
            "and can be used by the requester to match up replies to outstanding queries."
        ),
        alias="ID",
    )

    is_response: boolean = Field(
        ...,
        description=(
            "A one bit field that specifies whether this dns is a query (0), or a response (1)."
        ),
        alias="QR",
    )

    type: uint4 = Field(
        ...,
        description=(
            "A four bit field that specifies kind of query in this dns. "
            "This value is set by the originator of a query and copied into the response."
        ),
        alias="CODE",
    )

    is_authoritative: boolean = Field(
        ...,
        description=(
            "Authoritative Answer - this bit is valid in responses, "
            "and specifies that the responding name server is an authority "
            "for the domain name in question section."
        ),
        alias="AA",
    )

    is_truncated: boolean = Field(
        ...,
        description=(
            "Truncation - specifies that this dns was truncated due to "
            "length greater than that permitted on the transmission channel."
        ),
        alias="TC",
    )

    is_recursion_desired: boolean = Field(
        ...,
        description=(
            "Recursion Desired - this bit may be set in a query and is copied into the response. "
            "If RD is set, it directs the name server to pursue the query recursively."
        ),
        alias="RD",
    )

    is_recursion_available: boolean = Field(
        ...,
        description=(
            "Recursion Available - this be is set or cleared in a response, "
            "and denotes whether recursive query support is available in the name server."
        ),
        alias="RA",
    )

    reserved: int3 = Field(
        ...,
        description=("Reserved for future use. Must be zero in all queries and responses."),
        alias="Z",
    )

    response_code: uint4 = Field(
        ...,
        description=(
            "Response code - this 4 bit field is set as part of responses. "
            "The values have the following interpretation:"
        ),
        alias="Status",
    )

    question_count: uint16 = Field(
        ...,
        description=(
            "An unsigned 16 bit integer specifying the number of entries in the question section."
        ),
        alias="QDCOUNT",
    )

    answer_count: uint16 = Field(
        ...,
        description=(
            "An unsigned 16 bit integer specifying the number of resource "
            "records in the answer section."
        ),
        alias="ANCOUNT",
    )

    authority_count: uint16 = Field(
        ...,
        description=(
            "An unsigned 16 bit integer specifying the number of name server "
            "resource records in the authority records section."
        ),
        alias="NSCOUNT",
    )

    additional_count: uint16 = Field(
        ...,
        description=(
            "An unsigned 16 bit integer specifying the number of resource records "
            "in the additional records section."
        ),
        alias="ARCOUNT",
    )

    question: List[DNSMessageQuestion] = Field(
        ...,
        description="The dns questions.",
        alias="Question",
        json_schema_extra={
            "length": "QDCOUNT",
        },
    )

    answer: Optional[List[DNSMessageAnswer]] = Field(
        ...,
        description="The dns answers.",
        alias="Answer",
        json_schema_extra={
            "length": "ANCOUNT",
        },
    )

    authority: Optional[List[DNSMessageAnswer]] = Field(
        ...,
        description="The dns authorities.",
        alias="Authority",
        json_schema_extra={
            "length": "NSCOUNT",
        },
    )

    additional: Optional[List[DNSMessageAnswer]] = Field(
        ...,
        description="The dns additional records.",
        alias="Additional",
        json_schema_extra={
            "length": "ARCOUNT",
        },
    )

    @classmethod
    def get_mocked(cls, override: dict[str, Any] | None = None) -> Self:
        """Gets a mocked instance of the message."""

        return cls(
            id=1,
            is_response=True,
            type=0,
            is_authoritative=True,
            is_truncated=False,
            is_recursion_desired=True,
            is_recursion_available=True,
            reserved=0,
            response_code=0,
            question_count=1,
            answer_count=0,
            authority_count=0,
            additional_count=0,
            question=[DNSMessageQuestion.get_mocked()],
            answer=[],
            authority=[],
            additional=[],
            **(override or {}),
        )
