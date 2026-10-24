from __future__ import annotations

import time
from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gateway.providers.base import BaseProvider, CompletionRequest


class ExecutorAgent(BaseAgent):
    """Executes routed requests against the selected provider.

    Receives routing decisions from the RouterAgent, calls the provider,
    and publishes the response (or error) for downstream agents.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: AgentMemory,
        providers: dict[str, BaseProvider],
    ):
        super().__init__("executor-agent", event_bus, memory)
        self.providers = providers

    def subscribed_events(self) -> list[str]:
        return ["gateway.route_decided"]

    async def handle_event(self, event: Event) -> None:
        payload = event.payload
        request_id = payload["request_id"]
        provider_name = payload["provider"]
        model = payload["model"]

        provider = self.providers.get(provider_name)
        if not provider:
            await self.emit("gateway.error", {
                "request_id": request_id,
                "provider": provider_name,
                "model": model,
                "error": f"Unknown provider: {provider_name}",
            })
            return

        request = CompletionRequest(
            messages=payload["messages"],
            model=model,
            temperature=payload.get("temperature", 0.7),
            max_tokens=payload.get("max_tokens", 4096),
        )

        start = time.monotonic()
        try:
            response = await provider.complete(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            self.metrics.total_latency_ms += elapsed_ms

            await self.emit("gateway.response", {
                "request_id": request_id,
                "provider": provider_name,
                "model": model,
                "content": response.content,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "task_type": "unknown",
            })

            await self.emit("request.complete", {
                "request_id": request_id,
                "result": {
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                },
            })

        except Exception as exc:
            await self.emit("gateway.error", {
                "request_id": request_id,
                "provider": provider_name,
                "model": model,
                "error": str(exc),
            })

            await self.emit("request.complete", {
                "request_id": request_id,
                "result": {
                    "error": str(exc),
                    "provider": provider_name,
                    "model": model,
                },
            })
