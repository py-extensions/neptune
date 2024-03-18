import logging
from typing import Any, ClassVar, Optional, Self, Union, cast, get_args, get_origin

from pydantic import BaseModel

from neptune.types import BaseDNSType, BitArray

LOG = logging.getLogger(__name__)


class SupportedQueryClasses:
    """Supported DNS query classes."""

    IN = 1
    CS = 2
    CH = 3
    HS = 4


class BaseDNSModel(BaseModel):
    REORDER: ClassVar[list[str] | None] = None
    """Another fields order on serialization. Use it to reorder when using computer_field."""

    class Config:
        validate_assignment = True
        populate_by_name = True

    def to_bits(self) -> BitArray:
        """Converts the object to a bitarray."""

        data = BitArray()

        model_data = self.model_dump()

        order = self.REORDER or model_data.keys()

        for field_name in order:
            value = getattr(self, field_name)

            if isinstance(value, list):
                for item in value:
                    data += item.to_bits()
            else:
                data += value.to_bits()

        return data

    @classmethod
    def _get_type(cls, field, data: dict, extra: dict) -> BaseDNSType:
        """Get the type for a given data."""

        extra = extra or {}

        if "$type" in extra:
            getter = extra["$type"]["getter"]
            arg = extra["$type"]["arg"]

            return getattr(cls, getter)(data[arg])

        return field.annotation

    @classmethod
    def _parse_field(
        cls,
        field_type: type[BaseDNSType],
        bitarray: BitArray,
        ctx: Optional[dict[str, Any]] = None,
    ) -> BaseDNSType:
        """Parses a field from a bitarray."""

        pos_backup = bitarray.pos

        if get_origin(field_type) is Union:
            for inner_type in get_args(field_type):
                try:
                    return inner_type.from_bits(bitarray, ctx=ctx)
                except ValueError:
                    bitarray.pos = pos_backup
            else:
                raise ValueError("No valid type found")

        else:
            return field_type.from_bits(bitarray, ctx=ctx)

    @classmethod
    def from_bits(cls, bitarray: BitArray, *args, **kwargs) -> Self:
        """Parses a DNS header from a bitarray."""

        data = {}
        order = (
            cls.REORDER
            or cls.model_json_schema(
                mode="serialization",
                by_alias=False,
            )["properties"].keys()
        )

        for field_name in order:
            field = cls.model_fields[field_name]
            field_type = cast(
                type[BaseDNSType],
                cls._get_type(field, data, field.json_schema_extra),
            )
            field_name = field.alias or field_name

            # Optional fields are present as Union with None with real type as first argument
            if get_origin(field_type) is Union and type(None) in get_args(field_type):
                field_type = get_args(field_type)[0]

            # List fields are present as List with real type as first argument
            if get_origin(field_type) is list:
                inner_type = get_args(field_type)[0]
                list_length_field = field.json_schema_extra["length"]
                list_length = data[list_length_field].value

                data[field_name] = [
                    cls._parse_field(
                        inner_type,
                        bitarray,
                        ctx=field.json_schema_extra,
                    )
                    for _ in range(list_length)
                ]

            else:
                data[field_name] = cls._parse_field(
                    field_type,
                    bitarray,
                    ctx=field.json_schema_extra,
                )

        return cls(**data)
