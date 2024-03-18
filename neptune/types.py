import copy
import ipaddress
import sys
from abc import abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Callable, ClassVar, Optional, ParamSpec, Self, Type, TypeVar, Union, cast

from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
from pydantic_core import core_schema

UNDEFINED = object()

P = ParamSpec("P")
R = TypeVar("R")
F = Callable[P, R]


class Endian(Enum):
    """Endianness."""

    LITTLE = "little"
    BIG = "big"


MACHINE_ENDIAN = Endian[sys.byteorder.upper()]


def with_validation(f):
    """Decorator to add validation to the instance init."""

    @wraps(f)
    def wrapper(cls, value):
        cls._validate(value, None)
        return f(cls, value)

    return wrapper


def _convert_endian(data: list[int]):
    """Converts endianness of the data."""

    step = 8

    new_data = []

    for start in range(0, len(data), step):
        new_data += [data[start : start + step]]

    return [bit for byte in reversed(new_data) for bit in byte]


def convert_machine_endian(f):
    """Converts endianness of the result bitarray to machine endian."""

    @wraps(f)
    def wrapper(
        dns_type: Union[Type["BaseDNSType"], "BaseDNSType"],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        result = f(dns_type, *args, **kwargs)

        if MACHINE_ENDIAN != dns_type.ENDIAN:
            result = _convert_endian(result)

        return result

    return wrapper


class BitArray:
    """Array of bits. Which is a list of 0s and 1s.

    Array of data is always big-endian.
    """

    def __init__(self, data: bytearray | bytes | None = None):
        self.data = self.bytes_to_bits(bytearray(data or b""))
        self.pos = 0

    @staticmethod
    def bytes_to_bits(data: Union[bytes, bytearray]) -> list[int]:
        """Converts bytes to list of individual bits."""

        bits = []

        for byte in data:
            for i in range(0, 8):
                bits.append((byte >> i) & 1)

        return bits

    @staticmethod
    def bits_to_bytes(bits: Union["BitArray", list[int]]) -> bytearray:
        """Converts list of bits to bytes."""

        if isinstance(bits, BitArray):
            bits = bits.data

        result = bytearray([])
        for step in range(0, len(bits), 8):
            byte = bits[step : step + 8]

            # Pad with zeros if needed
            if len(byte) < 8:
                byte += [0] * (8 - len(byte))

            result.append(sum([2**i * v for i, v in enumerate(byte)]))

        return result

    def peek(self, length: int, offset: int = 0) -> list[int]:
        """Peek at next bits. Keep position unchanged."""

        return self.data[self.pos + offset : self.pos + offset + length]

    def read(self, length: int) -> list[int]:
        """Read next bits. Move position."""

        result = self.peek(length)
        self.pos += length

        return result

    def read_bytes(self, length: int) -> bytearray:
        """Read next bytes. Move position."""

        return self.bits_to_bytes(self.read(length * 8))

    def seek(self, location: int) -> Self:
        """Creates a copy of the array with a different position."""

        array_copy = copy.deepcopy(self)
        array_copy.pos = location

        return array_copy

    @classmethod
    def from_bits(cls, bits: list[int]) -> Self:
        """Creates a bitarray from list of bits."""

        return cls(cls.bits_to_bytes(bits))

    def __len__(self):
        return len(self.data)

    def __iadd__(self, other: Self):
        if isinstance(other, BitArray):
            self.data += other.data
        elif isinstance(other, list):
            self.data += other
        return self


class BaseDNSType:
    LENGTH: ClassVar[int] = UNDEFINED  # Length in bits
    ENDIAN: ClassVar[Endian] = Endian.LITTLE  # Network endianness

    @with_validation
    def __init__(self, value: Any):
        self.value = value

    @classmethod
    @convert_machine_endian
    def _read_bits(cls, data: BitArray, length: Optional[int] = None) -> list[int]:
        return data.read(length or cls.LENGTH)

    @classmethod
    @convert_machine_endian
    def _peek_bits(cls, data: BitArray, length: int = None, offset: int = 0) -> list[int]:
        return data.peek(length or cls.LENGTH, offset)

    @classmethod
    @convert_machine_endian
    def _read_bytes(cls, data: BitArray, length: int) -> bytearray:
        return data.read_bytes(length)

    @classmethod
    def peek(cls, data: BitArray) -> Self:
        """Peek at next bits. Keep position unchanged."""

        return cls.from_bits(BitArray.from_bits(cls._peek_bits(data)))

    @classmethod
    @abstractmethod
    def from_bits(cls, data: BitArray, ctx: Optional[dict[str, Any]] = None) -> Self:
        """Parses a value from a bitarray."""

    @convert_machine_endian
    def to_bits(self) -> list[int]:
        """Converts the object to a bitarray."""

        return self._to_bits()

    @abstractmethod
    def _to_bits(self) -> list[int]:
        """Inner method to convert value to bits."""

    @classmethod
    @abstractmethod
    def _validate(cls, value: Any, info: core_schema.ValidationInfo) -> Any:
        """Validates a value."""

    @classmethod
    def validate(cls, value: Any, info: core_schema.ValidationInfo) -> Any:
        """Validates a value."""

        if isinstance(value, cls):
            return value

        return cls._validate(value, info)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.with_info_plain_validator_function(cls.validate)


class Integer(BaseDNSType):
    ENDIAN: ClassVar[Endian] = Endian.BIG
    SIGNED: bool = True

    @classmethod
    def max_value(cls) -> int:
        return 2 ** (cls.LENGTH - 1) - 1 if cls.SIGNED else 2**cls.LENGTH - 1

    @classmethod
    def min_value(cls) -> int:
        return -(2 ** (cls.LENGTH - 1)) if cls.SIGNED else 0

    @classmethod
    def from_bits(cls, bits: BitArray, *args, **kwargs) -> Self:
        """Converts list of bits to integer."""

        return cls(sum([2**i * v for i, v in enumerate(cls._read_bits(bits))]))

    def _to_bits(self) -> list[int]:
        """Converts integer to list of bits."""

        if not self.SIGNED:
            return [(self.value >> i) & 1 for i in range(self.LENGTH)]
        else:
            return [(self.value >> i) & 1 for i in range(self.LENGTH - 1)] + [int(self.value < 0)]

    @classmethod
    def _validate(
        cls, value: Union[int, list[int], Self], info: core_schema.ValidationInfo
    ) -> int:
        """Validates a value."""

        if isinstance(value, list):
            if len(value) != cls.LENGTH:
                raise ValueError(f"Expected {cls.LENGTH} bits, got {len(value)}.")

            if any(bit not in (0, 1) for bit in value):
                raise ValueError("Expected 0 or 1.")

            return sum([2**i * v for i, v in enumerate(reversed(value))])

        elif isinstance(value, int):
            if not cls.min_value() <= value <= cls.max_value():
                raise ValueError(
                    f"Expected {cls.min_value()} <= value <= {cls.max_value()}, got {value}."
                )
            return value

        elif isinstance(value, Integer):
            return value.value

        else:
            raise ValueError(f"Expected int or list of bits, got {type(value)}.")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.JsonSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Returns JSON schema for integer."""

        return {
            "type": "integer",
            "minimum": cls.min_value(),
            "maximum": cls.max_value(),
        }

    def __repr__(self):
        return f"{self.value}"

    def __eq__(self, other):
        if isinstance(other, Integer):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __bool__(self):
        return bool(self.value)


class UnsignedInteger(Integer):
    SIGNED = False


class Boolean(UnsignedInteger):
    LENGTH = 1

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.JsonSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "boolean"}

    @classmethod
    def _validate(
        cls,
        value: Union[int, list[int], bool],
        info: core_schema.ValidationInfo,
    ) -> int:
        if isinstance(value, bool):
            value = int(value)

        return super()._validate(value, info)


class String(BaseDNSType):
    ENCODING: str = "ascii"
    LENGTH: ClassVar[int] = UNDEFINED
    SEPARATOR: ClassVar[list[int]] = [0] * 8

    def _to_bits(self) -> list[int]:
        """Converts string to list of bits."""

        return BitArray.bytes_to_bits(self.value.encode(self.ENCODING))

    @classmethod
    def from_bits(cls, data: BitArray, ctx: Optional[dict[str, Any]] = None) -> Self:
        """Parses a string from a bitarray."""

        result = ""

        ctx = ctx or {}

        length = ctx.get("length", None)

        if length is None:
            length = int8.from_bits(data).value

        result += cls._read_bytes(data, length).decode(cls.ENCODING)

        return cls(result)

    @classmethod
    def _validate(cls, value: Union[str, bytes, Self], info: core_schema.ValidationInfo) -> str:
        if isinstance(value, cls):
            value = value.value

        elif isinstance(value, bytes):
            value = value.decode(cls.ENCODING)

        elif not isinstance(value, str):
            raise ValueError(f"Expected str, got {type(value)}.")

        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, *args):
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.JsonSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "format": "byte" if cls.ENCODING == "ascii" else "binary",
        }

    def __repr__(self):
        return f"{self.value}"


class StringWithPrefix(String):
    """String with length prefix in front."""

    def _to_bits(self) -> list[int]:
        """Converts string to list of bits."""

        return int8(len(self.value))._to_bits() + super()._to_bits()

    @classmethod
    def from_bits(cls, data: BitArray, ctx: Optional[dict[str, Any]] = None) -> Self:
        """Parses a string from a bitarray."""

        length = int8.from_bits(BitArray(data.read_bytes(1))).value
        return super().from_bits(data, ctx={"length": length})


class ASCIIString(String):
    ENCODING: str = "ascii"


class ASCIIStringWithPrefix(ASCIIString, StringWithPrefix):
    pass


class UTF8String(String):
    ENCODING: str = "utf-8"


class UTF8StringWithPrefix(UTF8String, StringWithPrefix):
    pass


class DomainName(UTF8String):

    def to_bits(self) -> list[int]:
        """Converts domain name to list of bits."""

        result = []

        for part in self.value.split("."):
            result.extend(uint8(len(part)).to_bits())
            result.extend(BitArray.bytes_to_bits(part.encode(self.ENCODING)))

        return result

    @classmethod
    def from_bits(cls, data: BitArray, ctx: Optional[dict[str, Any]] = None) -> Self:
        """Parses a domain name from a bitarray."""

        result = ""

        while uint8.peek(data) != 0:
            result += super().from_bits(data, ctx={"length": uint8.from_bits(data).value}).value
            print(result)

            try:
                if Pointer.peek(data):
                    pointer = Pointer.from_bits(data)
                    result += pointer.get_target(data)
                    break
            except ValueError:
                result += "."

        else:
            # Skip separator
            data.read(8)

        return cls(result)


class Pointer(UnsignedInteger):
    LENGTH = 16

    @property
    def is_pointer(self) -> bool:
        return self.value == 4

    @property
    def location(self):
        return self.value & 0x3FFF

    def get_target(self, data: BitArray) -> DomainName:
        return DomainName.from_bits(data.seek(self.location))

    @classmethod
    def from_bits(cls, bits: BitArray, *args, **kwargs) -> Self:
        """Parses a pointer from a bitarray."""

        pointer = cast(cls, super().from_bits(bits))

        if not pointer.is_pointer:
            raise ValueError("Invalid pointer.")

        return pointer


def _generate_int_type(int_length: int, int_type: Type[Integer]) -> Type[Integer]:
    return cast(Type[int_type], type(f"int{int_length}", (int_type,), {"LENGTH": int_length}))


class IPAddressMixin(BaseDNSType):
    ENDIAN: ClassVar[Endian] = Endian.BIG

    def _to_bits(self) -> list[int]:
        """Converts IP address to list of bits."""

        return BitArray.bytes_to_bits(self.packed)  # noqa

    @classmethod
    def from_bits(cls, data: BitArray, *args, **kwargs) -> Self:
        """Parses an IP address from a bitarray."""

        return cls(data.read_bytes(cls.LENGTH // 8))  # type: ignore

    @classmethod
    def _validate(cls, value: Union[str, bytes], info: core_schema.ValidationInfo) -> str:
        try:
            cls(value)  # type: ignore
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {e}")
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, *args):
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.JsonSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "format": "ipv4" if cls.LENGTH == 32 else "ipv6"}


class IPv4Address(ipaddress.IPv4Address, IPAddressMixin):
    LENGTH = 32


class IPv6Address(ipaddress.IPv6Address, IPAddressMixin):
    LENGTH = 128


# Generate integer types
int3 = _generate_int_type(3, Integer)
int4 = _generate_int_type(4, Integer)
int8 = _generate_int_type(8, Integer)
int16 = _generate_int_type(16, Integer)
int32 = _generate_int_type(32, Integer)
int128 = _generate_int_type(128, Integer)


# Generate unsigned integer types
int1 = _generate_int_type(1, UnsignedInteger)
uint3 = _generate_int_type(3, UnsignedInteger)
uint4 = _generate_int_type(4, UnsignedInteger)
uint8 = _generate_int_type(8, UnsignedInteger)
uint16 = _generate_int_type(16, UnsignedInteger)
uint32 = _generate_int_type(32, UnsignedInteger)


# Generate special types
boolean = _generate_int_type(1, Boolean)
