from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central orchestrator that coordinates all agents.

    Manages agent lifecycle, routes requests through the agent pipeline,
    and collects cross-agent decisions into a unified response.
    """

    def __init__(self, event_bus: EventBus, memory: AgentMemory):
        self.event_bus = event_bus
        self.memory = memory
        self.agents: dict[str, BaseAgent] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_decisions: dict[str, list[dict[str, Any]]] = {}

    async def start(self) -> None:
        await self.event_bus.connect()
        await self.memory.connect()
        self.event_bus.subscribe("agent.decision", self._collect_decision)
        self.event_bus.subscribe("request.complete", self._complete_request)

        for agent in self.agents.values():
            await agent.start()

        logger.info(
            "Orchestrator started with %d agents: %s",
            len(self.agents),
            list(self.agents.keys()),
        )

    async def stop(self) -> None:
        for agent in self.agents.values():
            await agent.stop()
        await self.event_bus.disconnect()
        await self.memory.disconnect()
        logger.info("Orchestrator stopped")

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.agent_id] = agent
        logger.info("Registered agent: %s", agent.agent_id)

    async def process_request(
        self,
        request_type: str,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        self._pending_requests[request_id] = future
        self._request_decisions[request_id] = []

        start = time.monotonic()

        await self.event_bus.publish(
            Event(
                type=request_type,
                payload={**payload, "request_id": request_id},
                source="orchestrator",
            )
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            elapsed_ms = (time.monotonic() - start) * 1000
            result["orchestration_ms"] = round(elapsed_ms, 2)
            return result
        except asyncio.TimeoutError:
            decisions = self._request_decisions.get(request_id, [])
            return {
                "request_id": request_id,
                "status": "timeout",
                "partial_decisions": decisions,
                "error": f"Request timed out after {timeout}s",
            }
        finally:
            self._pending_requests.pop(request_id, None)
            self._request_decisions.pop(request_id, None)

    async def _collect_decision(self, event: Event) -> None:
        request_id = event.payload.get("request_id")
        if request_id and request_id in self._request_decisions:
            self._request_decisions[request_id].append(
                {
                    "agent": event.source,
                    "decision": event.payload.get("decision"),
                    "confidence": event.payload.get("confidence", 1.0),
                    "reasoning": event.payload.get("reasoning"),
                }
            )

    async def _complete_request(self, event: Event) -> None:
        request_id = event.payload.get("request_id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests[request_id]
            if not future.done():
                future.set_result(
                    {
                        "request_id": request_id,
                        "status": "completed",
                        "result": event.payload.get("result"),
                        "decisions": self._request_decisions.get(request_id, []),
                    }
                )

    def agent_statuses(self) -> list[dict[str, Any]]:
        return [agent.status() for agent in self.agents.values()]
