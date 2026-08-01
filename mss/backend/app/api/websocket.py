import asyncio
import json
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import WebSocket

from app.config import settings


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._user_connections: dict[str, set[WebSocket]] = {}
        self._redis: aioredis.Redis | None = None
        self._pubsub_task: asyncio.Task | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def connect(self, ws: WebSocket, channel: str) -> None:
        await ws.accept()
        if channel not in self._connections:
            self._connections[channel] = set()
        self._connections[channel].add(ws)

        if channel.startswith("notifications:user:"):
            user_id = channel.split(":")[-1]
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(ws)

        if self._pubsub_task is None:
            self._pubsub_task = asyncio.create_task(self._listen_redis())

    async def disconnect(self, ws: WebSocket, channel: str) -> None:
        if channel in self._connections:
            self._connections[channel].discard(ws)
            if not self._connections[channel]:
                del self._connections[channel]

        if channel.startswith("notifications:user:"):
            user_id = channel.split(":")[-1]
            if user_id in self._user_connections:
                self._user_connections[user_id].discard(ws)
                if not self._user_connections[user_id]:
                    del self._user_connections[user_id]

    async def broadcast(self, channel: str, message: dict[str, Any]) -> None:
        redis = await self._get_redis()
        await redis.publish(channel, json.dumps(message))

        if channel in self._connections:
            dead: list[WebSocket] = []
            for ws in self._connections[channel]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections[channel].discard(ws)

    async def send_personal(self, user_id: UUID | str, message: dict[str, Any]) -> None:
        channel = f"notifications:user:{user_id}"
        await self.broadcast(channel, message)

    async def _listen_redis(self) -> None:
        redis = await self._get_redis()
        pubsub = redis.pubsub()
        await pubsub.psubscribe("notifications:user:*")

        async for message in pubsub.listen():
            if message["type"] not in ("pmessage", "message"):
                continue
            channel = message.get("channel", "")
            if isinstance(channel, bytes):
                channel = channel.decode()
            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            if channel in self._connections:
                dead: list[WebSocket] = []
                for ws in self._connections[channel]:
                    try:
                        await ws.send_json(data)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._connections[channel].discard(ws)


manager = ConnectionManager()
