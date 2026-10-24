from __future__ import annotations

from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gpu.metrics import ClusterMetrics, MetricsCollector
from nexus.gpu.predictor import WorkloadPredictor
from nexus.gpu.scheduler import GPUScheduler, MigrationPlan, SchedulingStrategy, Workload


class GPUSchedulerAgent(BaseAgent):
    """Reacts to GPU metrics and optimizes workload placement.

    Listens for cluster metrics, computes efficiency scores, generates
    migration plans, and schedules new workloads onto optimal GPUs.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: AgentMemory,
        scheduler: GPUScheduler,
        collector: MetricsCollector,
        predictor: WorkloadPredictor,
    ):
        super().__init__("gpu-scheduler-agent", event_bus, memory)
        self.scheduler = scheduler
        self.collector = collector
        self.predictor = predictor
        self._pending_migrations: list[MigrationPlan] = []

    def subscribed_events(self) -> list[str]:
        return ["gpu.metrics_collected", "gpu.schedule_workload", "gpu.alerts"]

    async def handle_event(self, event: Event) -> None:
        if event.type == "gpu.metrics_collected":
            await self._evaluate_cluster(event)
        elif event.type == "gpu.schedule_workload":
            await self._schedule_workload(event)
        elif event.type == "gpu.alerts":
            await self._handle_alerts(event)

    async def _evaluate_cluster(self, event: Event) -> None:
        cluster = await self.collector.collect()
        efficiency = self.scheduler.compute_cluster_efficiency(cluster)

        await self.memory.share("gpu_efficiency", efficiency)

        if efficiency.get("score", 100) < 50:
            migrations = self.scheduler.generate_migration_plans(cluster)
            self._pending_migrations = migrations

            if migrations:
                await self.emit("gpu.migration_proposed", {
                    "migrations": [
                        {
                            "source": m.source_device_id,
                            "target": m.target_device_id,
                            "workload": m.workload_id,
                            "reason": m.reason,
                            "improvement": m.expected_improvement,
                            "risk": m.risk_level,
                        }
                        for m in migrations
                    ],
                    "cluster_efficiency": efficiency,
                })

        predictions = self.predictor.predict_all(steps_ahead=6)
        preemptive_actions = []
        for pred in predictions:
            if pred.recommended_action and pred.confidence > 0.5:
                preemptive_actions.append({
                    "device_id": pred.device_id,
                    "predicted_util": pred.predicted_utilization,
                    "action": pred.recommended_action,
                    "confidence": pred.confidence,
                    "trend": pred.trend,
                })

        if preemptive_actions:
            await self.emit("gpu.preemptive_actions", {
                "actions": preemptive_actions,
            })

        await self.emit("agent.decision", {
            "request_id": event.payload.get("request_id", "periodic"),
            "decision": {
                "efficiency": efficiency,
                "pending_migrations": len(self._pending_migrations),
                "preemptive_actions": len(preemptive_actions),
            },
            "confidence": 0.8,
            "reasoning": f"Cluster efficiency={efficiency.get('score', 0):.0f}%, "
                         f"{len(self._pending_migrations)} migrations proposed",
        })

    async def _schedule_workload(self, event: Event) -> None:
        payload = event.payload
        workload = Workload(
            workload_id=payload["workload_id"],
            name=payload.get("name", "unnamed"),
            memory_required_mb=payload["memory_required_mb"],
            gpu_utilization_estimate=payload.get("gpu_utilization_estimate", 50),
            priority=payload.get("priority", 0),
            preemptible=payload.get("preemptible", False),
            affinity_gpu_type=payload.get("affinity_gpu_type"),
        )

        cluster = await self.collector.collect()
        strategy_name = payload.get("strategy", self.scheduler.default_strategy.value)
        strategy = SchedulingStrategy(strategy_name)

        decision = self.scheduler.schedule(workload, cluster, strategy)

        if decision:
            await self.emit("gpu.workload_scheduled", {
                "workload_id": workload.workload_id,
                "assigned_device": decision.assigned_device_id,
                "gpu_uuid": decision.assigned_gpu_uuid,
                "strategy": decision.strategy_used,
                "score": decision.score,
                "reasoning": decision.reasoning,
            })
        else:
            await self.emit("gpu.schedule_failed", {
                "workload_id": workload.workload_id,
                "reason": "No suitable GPU available",
            })

    async def _handle_alerts(self, event: Event) -> None:
        alerts = event.payload.get("alerts", [])
        critical = [a for a in alerts if a["severity"] == "critical"]

        if critical:
            self.logger.warning(
                "%d critical GPU alerts — triggering emergency evaluation", len(critical)
            )
            cluster = await self.collector.collect()
            migrations = self.scheduler.generate_migration_plans(cluster)
            if migrations:
                low_risk = [m for m in migrations if m.risk_level == "low"]
                if low_risk:
                    await self.emit("gpu.emergency_migration", {
                        "migrations": [
                            {
                                "source": m.source_device_id,
                                "target": m.target_device_id,
                                "workload": m.workload_id,
                                "reason": m.reason,
                            }
                            for m in low_risk
                        ],
                    })
