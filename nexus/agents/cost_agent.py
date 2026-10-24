from __future__ import annotations

from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gateway.cost_tracker import CostRecord, CostTracker


class CostAgent(BaseAgent):
    """Monitors spending and optimizes cost across providers.

    Tracks every request's cost, detects budget anomalies, and can
    activate cost pressure mode to force cheaper routing decisions.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: AgentMemory,
        cost_tracker: CostTracker,
        hourly_budget: float = 10.0,
    ):
        super().__init__("cost-agent", event_bus, memory)
        self.cost_tracker = cost_tracker
        self.hourly_budget = hourly_budget
        self._window_cost = 0.0
        self._window_requests = 0

    def subscribed_events(self) -> list[str]:
        return ["gateway.response"]

    async def handle_event(self, event: Event) -> None:
        payload = event.payload
        request_id = payload["request_id"]

        record = CostRecord(
            provider=payload["provider"],
            model=payload["model"],
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
            cost_usd=payload["cost_usd"],
            latency_ms=payload["latency_ms"],
            task_type=payload.get("task_type", "unknown"),
        )
        self.cost_tracker.record(record)

        self._window_cost += record.cost_usd
        self._window_requests += 1

        cost_pressure = self._window_cost > self.hourly_budget * 0.8
        if cost_pressure:
            self.logger.warning(
                "Cost pressure: $%.4f spent (%.0f%% of $%.2f budget)",
                self._window_cost,
                (self._window_cost / self.hourly_budget) * 100,
                self.hourly_budget,
            )

        await self.memory.share("cost_pressure", {
            "active": cost_pressure,
            "current_spend": round(self._window_cost, 6),
            "budget": self.hourly_budget,
            "utilization": round(self._window_cost / self.hourly_budget, 4),
        })

        savings = self._calculate_savings(record)

        await self.emit("agent.decision", {
            "request_id": request_id,
            "decision": {
                "cost_recorded": record.cost_usd,
                "total_spend": round(self._window_cost, 6),
                "budget_utilization": round(self._window_cost / self.hourly_budget, 4),
                "cost_pressure_active": cost_pressure,
                "potential_savings": savings,
            },
            "confidence": 1.0,
            "reasoning": f"Cost=${record.cost_usd:.6f}, budget={self._window_cost/self.hourly_budget:.0%}",
        })

    def _calculate_savings(self, record: CostRecord) -> dict[str, Any]:
        cheapest_cost = float("inf")
        cheapest_model = record.model

        for model_name, cost in self.cost_tracker.cost_by_model().items():
            per_request = cost / max(self.cost_tracker.requests_per_provider().get(
                model_name.split(":")[0] if ":" in model_name else "unknown", 1
            ), 1)
            if per_request < cheapest_cost:
                cheapest_cost = per_request
                cheapest_model = model_name

        return {
            "cheapest_model": cheapest_model,
            "current_cost": record.cost_usd,
        }

    def reset_window(self) -> None:
        self._window_cost = 0.0
        self._window_requests = 0
