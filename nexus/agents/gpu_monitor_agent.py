from __future__ import annotations

import asyncio
from typing import Any

from nexus.core.base_agent import BaseAgent
from nexus.core.event_bus import Event, EventBus
from nexus.core.memory import AgentMemory
from nexus.gpu.metrics import ClusterMetrics, MetricsCollector
from nexus.gpu.predictor import WorkloadPredictor


class GPUMonitorAgent(BaseAgent):
    """Continuously collects GPU metrics and publishes cluster state.

    Feeds data to the predictor for trend analysis and alerts the
    scheduler agent when utilization thresholds are crossed.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: AgentMemory,
        collector: MetricsCollector,
        predictor: WorkloadPredictor,
        poll_interval: float = 10.0,
    ):
        super().__init__("gpu-monitor-agent", event_bus, memory)
        self.collector = collector
        self.predictor = predictor
        self.poll_interval = poll_interval
        self._poll_task: asyncio.Task | None = None
        self._alert_thresholds = {
            "overload_pct": 85,
            "thermal_c": 80,
            "memory_pct": 90,
        }

    def subscribed_events(self) -> list[str]:
        return ["gpu.config_update"]

    async def start(self) -> None:
        await super().start()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def handle_event(self, event: Event) -> None:
        if event.type == "gpu.config_update":
            new_thresholds = event.payload.get("thresholds")
            if new_thresholds:
                self._alert_thresholds.update(new_thresholds)
                self.logger.info("Updated alert thresholds: %s", self._alert_thresholds)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._collect_and_publish()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("GPU metrics collection error")
                await asyncio.sleep(self.poll_interval)

    async def _collect_and_publish(self) -> None:
        cluster = await self.collector.collect()

        for device in cluster.devices:
            self.predictor.record(device.device_id, device.utilization_gpu)

        await self.memory.share("gpu_cluster_metrics", cluster.to_dict())
        await self.memory.share("gpu_predictions", self.predictor.summary())

        await self.emit("gpu.metrics_collected", {
            "cluster": cluster.to_dict(),
            "predictions": self.predictor.summary(),
        })

        alerts = self._check_alerts(cluster)
        if alerts:
            await self.emit("gpu.alerts", {"alerts": alerts})

        self.metrics.tasks_completed += 1

    def _check_alerts(self, cluster: ClusterMetrics) -> list[dict[str, Any]]:
        alerts = []
        for device in cluster.devices:
            if device.utilization_gpu > self._alert_thresholds["overload_pct"]:
                alerts.append({
                    "severity": "warning",
                    "device_id": device.device_id,
                    "type": "gpu_overload",
                    "message": f"GPU {device.device_id} at {device.utilization_gpu:.0f}% utilization",
                    "value": device.utilization_gpu,
                })
            if device.temperature_c > self._alert_thresholds["thermal_c"]:
                alerts.append({
                    "severity": "critical",
                    "device_id": device.device_id,
                    "type": "thermal_warning",
                    "message": f"GPU {device.device_id} at {device.temperature_c}°C",
                    "value": device.temperature_c,
                })
            if device.memory_utilization_pct > self._alert_thresholds["memory_pct"]:
                alerts.append({
                    "severity": "warning",
                    "device_id": device.device_id,
                    "type": "memory_pressure",
                    "message": f"GPU {device.device_id} memory at {device.memory_utilization_pct:.0f}%",
                    "value": device.memory_utilization_pct,
                })
        return alerts
