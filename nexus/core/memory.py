from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class AgentMemory:
    """Shared memory store for inter-agent knowledge.

    Uses Redis for fast access with optional PostgreSQL persistence for
    long-term storage and vector similarity search.
    """

    def __init__(self, redis_url: str | None = None):
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._local: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        if self._redis_url:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            logger.info("AgentMemory connected to Redis")

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()

    async def store(self, agent_id: str, key: str, value: Any, ttl: int | None = None) -> None:
        record = {
            "value": value,
            "agent_id": agent_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        redis_key = f"nexus:memory:{agent_id}:{key}"

        if self._redis:
            await self._redis.set(redis_key, json.dumps(record), ex=ttl)
        else:
            self._local[redis_key] = record

    async def recall(self, agent_id: str, key: str) -> Any | None:
        redis_key = f"nexus:memory:{agent_id}:{key}"

        if self._redis:
            raw = await self._redis.get(redis_key)
            if raw:
                return json.loads(raw)["value"]
            return None

        record = self._local.get(redis_key)
        return record["value"] if record else None

    async def recall_all(self, agent_id: str) -> dict[str, Any]:
        pattern = f"nexus:memory:{agent_id}:*"
        prefix_len = len(f"nexus:memory:{agent_id}:")
        results = {}

        if self._redis:
            async for key in self._redis.scan_iter(match=pattern, count=100):
                raw = await self._redis.get(key)
                if raw:
                    short_key = key[prefix_len:]
                    results[short_key] = json.loads(raw)["value"]
        else:
            for key, record in self._local.items():
                if key.startswith(pattern.replace("*", "")):
                    short_key = key[prefix_len:]
                    results[short_key] = record["value"]

        return results

    async def share(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.store("__shared__", key, value, ttl)

    async def recall_shared(self, key: str) -> Any | None:
        return await self.recall("__shared__", key)

    async def delete(self, agent_id: str, key: str) -> None:
        redis_key = f"nexus:memory:{agent_id}:{key}"
        if self._redis:
            await self._redis.delete(redis_key)
        else:
            self._local.pop(redis_key, None)
