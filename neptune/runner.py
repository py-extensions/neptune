import asyncio

from fastapi import FastAPI

from neptune.api import routers
import neptune.protocol.udp


app = FastAPI()

for router in routers:
    app.include_router(router)

asyncio.create_task(neptune.protocol.udp.start_server())
