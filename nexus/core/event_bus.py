from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from uuid import uuid4

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

EventHandler = Callable[["Event"], Coroutine[Any, Any, None]]


@dataclass
class Event:
    type: str
    payload: dict[str, Any]
    source: str
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def serialize(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "type": self.type,
                "source": self.source,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def deserialize(cls, raw: str) -> Event:
        data = json.loads(raw)
        return cls(
            id=data["id"],
            type=data["type"],
            source=data["source"],
            payload=data["payload"],
            timestamp=data["timestamp"],
        )


class EventBus:
    """Redis-backed pub/sub event bus for inter-agent communication.

    Supports both distributed (Redis) and in-process (fallback) modes.
    """

    def __init__(self, redis_url: str | None = None):
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._listener_task: asyncio.Task | None = None
        self._running = False

    async def connect(self) -> None:
        if self._redis_url:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe("nexus:events")
            self._running = True
            self._listener_task = asyncio.create_task(self._listen())
            logger.info("EventBus connected to Redis")
        else:
            self._running = True
            logger.info("EventBus running in local mode")

    async def disconnect(self) -> None:
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe("nexus:events")
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        if self._redis:
            await self._redis.publish("nexus:events", event.serialize())
        else:
            await self._dispatch(event)

    async def _listen(self) -> None:
        while self._running and self._pubsub:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    event = Event.deserialize(message["data"])
                    await self._dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("EventBus listener error")
                await asyncio.sleep(1)

    async def _dispatch(self, event: Event) -> None:
        handlers = self._handlers.get(event.type, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception("Handler error for event %s", event.type)
