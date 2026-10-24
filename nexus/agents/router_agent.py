from __future__ import annotations

import time
from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gateway.providers.base import CompletionRequest
from nexus.gateway.router import Router


class RouterAgent(BaseAgent):
    """Decides which model and provider should handle each request.

    Uses task classification, provider health data from the HealthAgent,
    and cost signals from the CostAgent to make routing decisions.
    """

    def __init__(self, event_bus: EventBus, memory: AgentMemory, router: Router):
        super().__init__("router-agent", event_bus, memory)
        self.router = router

    def subscribed_events(self) -> list[str]:
        return ["gateway.request"]

    async def handle_event(self, event: Event) -> None:
        start = time.monotonic()
        payload = event.payload
        request_id = payload["request_id"]

        messages = payload.get("messages", [])
        strategy = payload.get("strategy", "balanced")

        health_status = await self.recall("provider_health") or {}
        cost_pressure = await self.memory.recall_shared("cost_pressure")

        if cost_pressure and cost_pressure.get("active"):
            strategy = "cost"
            self.logger.info("Cost pressure active, forcing cost strategy")

        request = CompletionRequest(
            messages=messages,
            temperature=payload.get("temperature", 0.7),
            max_tokens=payload.get("max_tokens", 4096),
        )

        decision = self.router.route(request, strategy=strategy, health_status=health_status)

        elapsed_ms = (time.monotonic() - start) * 1000
        self.metrics.total_latency_ms += elapsed_ms

        self.metrics.record_decision({
            "provider": decision.provider,
            "model": decision.model,
            "strategy": strategy,
            "task_type": decision.reasoning.split(",")[0],
        })

        await self.emit("agent.decision", {
            "request_id": request_id,
            "decision": {
                "provider": decision.provider,
                "model": decision.model,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
                "cost_estimate": decision.cost_estimate,
                "latency_estimate": decision.latency_estimate,
            },
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        })

        await self.emit("gateway.route_decided", {
            "request_id": request_id,
            "provider": decision.provider,
            "model": decision.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        })
