from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.gpu.metrics import ClusterMetrics, GPUMetrics

logger = logging.getLogger(__name__)


class SchedulingStrategy(str, Enum):
    BIN_PACK = "bin_pack"           # maximize utilization per GPU
    SPREAD = "spread"               # distribute evenly across GPUs
    POWER_AWARE = "power_aware"     # minimize power consumption
    MEMORY_FIRST = "memory_first"   # prioritize memory availability


@dataclass
class Workload:
    workload_id: str
    name: str
    memory_required_mb: int
    gpu_utilization_estimate: float  # expected GPU % usage
    priority: int = 0               # higher = more important
    preemptible: bool = False
    affinity_gpu_type: str | None = None
    max_latency_ms: float | None = None


@dataclass
class SchedulingDecision:
    workload_id: str
    assigned_device_id: int
    assigned_gpu_uuid: str
    strategy_used: str
    score: float
    reasoning: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MigrationPlan:
    source_device_id: int
    target_device_id: int
    workload_id: str
    reason: str
    expected_improvement: float  # utilization improvement %
    risk_level: str  # "low", "medium", "high"


class GPUScheduler:
    """Bin-packs and schedules workloads across a GPU cluster.

    Implements multiple scheduling strategies and generates migration
    plans to rebalance overloaded/underutilized GPUs.
    """

    def __init__(self, default_strategy: SchedulingStrategy = SchedulingStrategy.BIN_PACK):
        self.default_strategy = default_strategy
        self._assignments: dict[str, int] = {}  # workload_id -> device_id

    def schedule(
        self,
        workload: Workload,
        cluster: ClusterMetrics,
        strategy: SchedulingStrategy | None = None,
    ) -> SchedulingDecision | None:
        strat = strategy or self.default_strategy
        candidates = self._rank_candidates(workload, cluster, strat)

        if not candidates:
            logger.warning("No suitable GPU found for workload %s", workload.workload_id)
            return None

        best = candidates[0]
        self._assignments[workload.workload_id] = best["device_id"]

        return SchedulingDecision(
            workload_id=workload.workload_id,
            assigned_device_id=best["device_id"],
            assigned_gpu_uuid=best["uuid"],
            strategy_used=strat.value,
            score=best["score"],
            reasoning=best["reasoning"],
            alternatives=[
                {"device_id": c["device_id"], "score": c["score"]}
                for c in candidates[1:4]
            ],
        )

    def generate_migration_plans(self, cluster: ClusterMetrics) -> list[MigrationPlan]:
        plans = []
        overloaded = [d for d in cluster.devices if d.utilization_gpu > 85]
        underloaded = [d for d in cluster.devices if d.utilization_gpu < 30 and not d.is_idle]
        idle = [d for d in cluster.devices if d.is_idle]

        targets = underloaded + idle

        for src in overloaded:
            if not targets:
                break
            if not src.running_processes:
                continue

            smallest_proc = min(src.running_processes, key=lambda p: p.memory_used_mb)
            best_target = self._find_best_migration_target(smallest_proc.memory_used_mb, targets)

            if best_target:
                improvement = (src.utilization_gpu - best_target.utilization_gpu) / 2
                plans.append(MigrationPlan(
                    source_device_id=src.device_id,
                    target_device_id=best_target.device_id,
                    workload_id=f"pid-{smallest_proc.pid}",
                    reason=f"GPU {src.device_id} at {src.utilization_gpu:.0f}% util, "
                           f"target GPU {best_target.device_id} at {best_target.utilization_gpu:.0f}%",
                    expected_improvement=round(improvement, 1),
                    risk_level="low" if best_target.is_idle else "medium",
                ))

        return plans

    def compute_cluster_efficiency(self, cluster: ClusterMetrics) -> dict[str, Any]:
        if not cluster.devices:
            return {"score": 0, "details": "No GPUs in cluster"}

        active_gpus = [d for d in cluster.devices if not d.is_idle]
        if not active_gpus:
            return {
                "score": 0,
                "total_gpus": cluster.total_gpus,
                "active_gpus": 0,
                "recommendation": "All GPUs idle — consolidate workloads or scale down",
            }

        avg_util = sum(d.utilization_gpu for d in active_gpus) / len(active_gpus)
        util_variance = sum((d.utilization_gpu - avg_util) ** 2 for d in active_gpus) / len(active_gpus)
        balance_score = max(0, 100 - util_variance ** 0.5)

        wasted_gpus = len(cluster.idle_gpus)
        wasted_memory_mb = sum(d.memory_free_mb for d in cluster.idle_gpus)

        efficiency_score = (avg_util * 0.5 + balance_score * 0.3 + (1 - wasted_gpus / cluster.total_gpus) * 100 * 0.2)

        recommendations = []
        if avg_util < 40:
            recommendations.append("Low average utilization — consolidate workloads")
        if wasted_gpus > cluster.total_gpus * 0.3:
            recommendations.append(f"{wasted_gpus} idle GPUs — consider scaling down")
        if balance_score < 50:
            recommendations.append("Unbalanced load — consider spreading workloads")
        if any(d.temperature_c > 80 for d in cluster.devices):
            recommendations.append("Thermal throttling risk — reduce load on hot GPUs")

        return {
            "score": round(efficiency_score, 1),
            "avg_utilization": round(avg_util, 1),
            "balance_score": round(balance_score, 1),
            "total_gpus": cluster.total_gpus,
            "active_gpus": len(active_gpus),
            "idle_gpus": wasted_gpus,
            "wasted_memory_mb": wasted_memory_mb,
            "overloaded_gpus": len(cluster.overloaded_gpus),
            "recommendations": recommendations,
        }

    def _rank_candidates(
        self,
        workload: Workload,
        cluster: ClusterMetrics,
        strategy: SchedulingStrategy,
    ) -> list[dict[str, Any]]:
        candidates = []

        for device in cluster.devices:
            if device.memory_free_mb < workload.memory_required_mb:
                continue
            if workload.affinity_gpu_type and workload.affinity_gpu_type not in device.uuid:
                continue

            score = self._score_device(device, workload, strategy)
            candidates.append({
                "device_id": device.device_id,
                "uuid": device.uuid,
                "score": round(score, 3),
                "reasoning": self._explain_score(device, workload, strategy),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _score_device(
        self, device: GPUMetrics, workload: Workload, strategy: SchedulingStrategy
    ) -> float:
        memory_headroom = (device.memory_free_mb - workload.memory_required_mb) / device.memory_total_mb
        gpu_headroom = (100 - device.utilization_gpu - workload.gpu_utilization_estimate) / 100

        if strategy == SchedulingStrategy.BIN_PACK:
            return (1 - memory_headroom) * 0.4 + (1 - gpu_headroom) * 0.4 + (1 if gpu_headroom > 0.1 else 0) * 0.2

        if strategy == SchedulingStrategy.SPREAD:
            return memory_headroom * 0.4 + gpu_headroom * 0.4 + (1 if device.is_idle else 0) * 0.2

        if strategy == SchedulingStrategy.POWER_AWARE:
            power_efficiency = 1 - (device.power_draw_w / max(device.power_limit_w, 1))
            return power_efficiency * 0.5 + gpu_headroom * 0.3 + memory_headroom * 0.2

        if strategy == SchedulingStrategy.MEMORY_FIRST:
            return memory_headroom * 0.6 + gpu_headroom * 0.3 + (1 if device.temperature_c < 70 else 0) * 0.1

        return gpu_headroom * 0.5 + memory_headroom * 0.5

    def _explain_score(
        self, device: GPUMetrics, workload: Workload, strategy: SchedulingStrategy
    ) -> str:
        return (
            f"GPU {device.device_id}: {device.utilization_gpu:.0f}% util, "
            f"{device.memory_free_mb}MB free, {device.temperature_c}°C, "
            f"strategy={strategy.value}"
        )

    def _find_best_migration_target(
        self, memory_needed_mb: int, targets: list[GPUMetrics]
    ) -> GPUMetrics | None:
        viable = [t for t in targets if t.memory_free_mb >= memory_needed_mb]
        if not viable:
            return None
        return min(viable, key=lambda t: t.utilization_gpu)
