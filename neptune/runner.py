import asyncio

from fastapi import FastAPI

import neptune.protocol.udp
from neptune.api import routers

app = FastAPI()

for router in routers:
    app.include_router(router)

asyncio.create_task(neptune.protocol.udp.start_server())
