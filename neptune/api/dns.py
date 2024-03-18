from fastapi import APIRouter

from neptune.model.dns.message import DNSMessage

router = APIRouter()


@router.get("/", response_model=DNSMessage)
async def root() -> DNSMessage:
    return DNSMessage.get_mocked()
