from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class HealthAgent(BaseAgent):
    """Continuously monitors provider health and latency.

    Tracks error rates, runs periodic health checks, and updates
    shared memory so the RouterAgent can avoid unhealthy providers.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: AgentMemory,
        providers: dict[str, BaseProvider],
        check_interval: float = 30.0,
    ):
        super().__init__("health-agent", event_bus, memory)
        self.providers = providers
        self.check_interval = check_interval
        self._error_counts: dict[str, int] = defaultdict(int)
        self._request_counts: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._health_task: asyncio.Task | None = None

    def subscribed_events(self) -> list[str]:
        return ["gateway.response", "gateway.error"]

    async def start(self) -> None:
        await super().start()
        self._health_task = asyncio.create_task(self._periodic_health_check())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def handle_event(self, event: Event) -> None:
        payload = event.payload
        provider = payload.get("provider", "unknown")
        model = payload.get("model", "unknown")
        key = f"{provider}:{model}"

        if event.type == "gateway.error":
            self._error_counts[key] += 1
            self._request_counts[key] += 1
        elif event.type == "gateway.response":
            self._request_counts[key] += 1
            latency = payload.get("latency_ms", 0)
            self._latencies[key].append(latency)
            if len(self._latencies[key]) > 100:
                self._latencies[key] = self._latencies[key][-100:]

        await self._update_health_status()

    async def _periodic_health_check(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health check error")

    async def _run_health_checks(self) -> None:
        health_status: dict[str, bool] = {}

        for name, provider in self.providers.items():
            for model_info in provider.available_models():
                key = f"{name}:{model_info.model_id}"
                try:
                    healthy = await provider.health_check()
                    health_status[key] = healthy
                    if not healthy:
                        self.logger.warning("Provider %s is unhealthy", key)
                except Exception:
                    health_status[key] = False
                    self.logger.exception("Health check failed for %s", key)

        await self.memory.store("router-agent", "provider_health", health_status)
        await self.emit("health.updated", {"status": health_status})

    async def _update_health_status(self) -> None:
        health_status: dict[str, bool] = {}

        for key in self._request_counts:
            total = self._request_counts[key]
            errors = self._error_counts[key]
            if total >= 5:
                error_rate = errors / total
                health_status[key] = error_rate < 0.5
            else:
                health_status[key] = True

        await self.memory.store("router-agent", "provider_health", health_status)

    def health_summary(self) -> dict[str, Any]:
        summary = {}
        for key in set(list(self._request_counts.keys()) + list(self._error_counts.keys())):
            total = self._request_counts.get(key, 0)
            errors = self._error_counts.get(key, 0)
            latencies = self._latencies.get(key, [])
            summary[key] = {
                "total_requests": total,
                "errors": errors,
                "error_rate": round(errors / max(total, 1), 4),
                "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
                "healthy": total < 5 or (errors / max(total, 1)) < 0.5,
            }
        return summary
