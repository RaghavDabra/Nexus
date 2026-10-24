from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory


class AgentState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMetrics:
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0
    last_active: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if self.tasks_completed == 0:
            return 0
        return self.total_latency_ms / self.tasks_completed

    def record_decision(self, decision: dict[str, Any]) -> None:
        self.decisions.append(decision)
        if len(self.decisions) > 100:
            self.decisions = self.decisions[-100:]


class BaseAgent(ABC):
    """Base class for all Nexus agents.

    Each agent has its own memory namespace, subscribes to relevant events,
    and publishes decisions back to the bus for other agents to react to.
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        memory: AgentMemory,
    ):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.memory = memory
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"nexus.agent.{agent_id}")
        self._task: asyncio.Task | None = None

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        ...

    @abstractmethod
    def subscribed_events(self) -> list[str]:
        ...

    async def start(self) -> None:
        for event_type in self.subscribed_events():
            self.event_bus.subscribe(event_type, self._safe_handle)
        self.state = AgentState.IDLE
        self.logger.info("Agent %s started, listening for %s", self.agent_id, self.subscribed_events())

    async def stop(self) -> None:
        self.state = AgentState.SHUTDOWN
        self.logger.info("Agent %s stopped", self.agent_id)

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(type=event_type, payload=payload, source=self.agent_id)
        await self.event_bus.publish(event)

    async def remember(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.memory.store(self.agent_id, key, value, ttl)

    async def recall(self, key: str) -> Any | None:
        return await self.memory.recall(self.agent_id, key)

    async def _safe_handle(self, event: Event) -> None:
        if event.source == self.agent_id:
            return
        self.state = AgentState.PROCESSING
        try:
            await self.handle_event(event)
            self.metrics.tasks_completed += 1
        except Exception:
            self.metrics.tasks_failed += 1
            self.state = AgentState.ERROR
            self.logger.exception("Error handling event %s", event.type)
        finally:
            if self.state != AgentState.ERROR:
                self.state = AgentState.IDLE

    def status(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            },
            "subscriptions": self.subscribed_events(),
        }
