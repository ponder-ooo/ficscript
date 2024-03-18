import asyncio
import sys
import time
from io import TextIOBase, BytesIO
from typing import Any, List

import PIL

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from fictionsuit.core.fictionscript.schematize import schematize
from .fictionscript.scope import Scope
from .fictionscript.ui_parameter import UiParameter
from hypercorn.config import Config
from hypercorn.asyncio import serve

from ..commands.command_group import CommandHandled

from .system import System
from .user_message import UserMessage


class ApiClient:
    def __init__(self, system: System):
        self.system: System = system
        self.cache = { }

    def run(self):
        app = FastAPI()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/fic")
        async def fic(request: RequestBody):
            if request.request_text is None:
                return {"error": "no request text"}
            if request.user_name is None:
                return {"error": "no user name"}

            result = await self.system.enqueue_message(
                ApiMessage(self, request), return_whatever=True
            )

            if result is None or isinstance(result, CommandHandled):
                return {"schema": "nothing"}


            return schematize(result, self.cache)
        
        @app.get("/png/{cache_id}.png")
        async def png(cache_id: str):
            if cache_id not in self.cache:
                return {"error": f"Cache does not contain {cache_id}"}
            result = self.cache[cache_id]
            if not hasattr(result, "pil"):
                return {"error": "Item does not expose an image"}
            
            image = result.pil()

            imageBytes = BytesIO()
            image.save(imageBytes, format="PNG")

            return Response(content=imageBytes.getvalue(), media_type="image/png")

        @app.post("/set_value/{cache_id}")
        async def set_value(cache_id: str, request: ValueBody):
            if cache_id not in self.cache:
                return {"error": f"Cache does not contain {cache_id}"}
            result = self.cache[cache_id]
            if not hasattr(result, "value"):
                return {"error": "Cannot set value"}
            result.value = request.value
            return schematize(result, self.cache)
        
        @app.post("/button/{cache_id}")
        async def button(cache_id: str):
            if cache_id not in self.cache:
                return {"error": f"Cache does not contain {cache_id}"}
            result = self.cache[cache_id]
            if not hasattr(result, "action"):
                return {"error": "Cannot press button"}
            result = await result.action()
            return schematize(result, self.cache)

        asyncio.run(serve(app, Config()))


class ValueBody(BaseModel):
    value: Any | None

class RequestBody(BaseModel):
    user_name: str | None
    request_text: str | None


class ApiMessage(UserMessage):
    """Wraps an api request to the system."""

    def __init__(self, client: ApiClient, request: RequestBody):
        super().__init__(
            request.request_text, request.user_name
        )  # TODO: user accounts, authentication, etc
        self.client = client
        self.request = request
        self.timestamp = time.time()

    async def _send(self, message_content: str) -> bool:
        pass  # TODO: websockets or something

    async def _reply(self, reply_content: str) -> bool:
        pass

    async def _react(self, reaction: str | None) -> bool:
        pass

    async def _undo_react(self, reaction: str | None) -> bool:
        pass

    async def _get_timestamp(self) -> float:
        return self.timestamp
