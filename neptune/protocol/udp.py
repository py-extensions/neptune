import asyncio
from asyncio import DatagramProtocol
from math import inf
from typing import ClassVar

from neptune.model.dns.message import DNSMessage
from neptune.protocol.protocol import Protocol, SupportedProtocol
from neptune.types import BitArray
from neptune.controller import DNSController


class DNSUDPProtocol(DatagramProtocol, Protocol):
    NAME: ClassVar[SupportedProtocol] = SupportedProtocol.UDP

    def __init__(self, controller: DNSController):
        self.controller = controller

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        """Handle incoming datagram."""

        message = DNSMessage.from_bits(BitArray(data))

        response = self.controller.process_message(message, ctx=self.generate_context())

        self.transport.sendto(BitArray.bits_to_bytes(response.to_bits()), addr)

    def connection_lost(self, exc):
        print("Connection lost:", exc)


async def start_server():
    print("Starting UDP server")

    controller = DNSController({})

    loop = asyncio.get_running_loop()

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: DNSUDPProtocol(controller), local_addr=("127.0.0.1", 8053)
    )

    try:
        await asyncio.sleep(inf)
    finally:
        transport.close()
