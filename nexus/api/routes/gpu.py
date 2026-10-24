from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])

_collector = None
_scheduler = None
_predictor = None
_event_bus = None


def init_gpu_routes(collector, scheduler, predictor, event_bus):
    global _collector, _scheduler, _predictor, _event_bus
    _collector = collector
    _scheduler = scheduler
    _predictor = predictor
    _event_bus = event_bus


@router.get("/metrics")
async def cluster_metrics():
    if not _collector:
        raise HTTPException(status_code=503, detail="GPU collector not initialized")
    cluster = await _collector.collect()
    return cluster.to_dict()


@router.get("/metrics/devices")
async def device_metrics():
    if not _collector:
        raise HTTPException(status_code=503, detail="GPU collector not initialized")
    cluster = await _collector.collect()
    return {
        "devices": [
            {
                "device_id": d.device_id,
                "uuid": d.uuid,
                "utilization_gpu": d.utilization_gpu,
                "utilization_memory": d.utilization_memory,
                "memory_used_mb": d.memory_used_mb,
                "memory_total_mb": d.memory_total_mb,
                "memory_free_mb": d.memory_free_mb,
                "temperature_c": d.temperature_c,
                "power_draw_w": d.power_draw_w,
                "power_limit_w": d.power_limit_w,
                "is_idle": d.is_idle,
                "is_overloaded": d.is_overloaded,
                "running_processes": len(d.running_processes),
            }
            for d in cluster.devices
        ]
    }


@router.get("/efficiency")
async def cluster_efficiency():
    if not _collector or not _scheduler:
        raise HTTPException(status_code=503, detail="GPU subsystem not initialized")
    cluster = await _collector.collect()
    return _scheduler.compute_cluster_efficiency(cluster)


@router.get("/predictions")
async def gpu_predictions():
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return {"predictions": _predictor.summary()}


@router.get("/migrations")
async def migration_plans():
    if not _collector or not _scheduler:
        raise HTTPException(status_code=503, detail="GPU subsystem not initialized")
    cluster = await _collector.collect()
    plans = _scheduler.generate_migration_plans(cluster)
    return {
        "plans": [
            {
                "source_device_id": p.source_device_id,
                "target_device_id": p.target_device_id,
                "workload_id": p.workload_id,
                "reason": p.reason,
                "expected_improvement": p.expected_improvement,
                "risk_level": p.risk_level,
            }
            for p in plans
        ]
    }


class ScheduleWorkloadRequest(BaseModel):
    workload_id: str
    name: str = "unnamed"
    memory_required_mb: int
    gpu_utilization_estimate: float = 50.0
    priority: int = 0
    preemptible: bool = False
    strategy: str = "bin_pack"


@router.post("/schedule")
async def schedule_workload(body: ScheduleWorkloadRequest):
    if not _collector or not _scheduler:
        raise HTTPException(status_code=503, detail="GPU subsystem not initialized")

    from nexus.gpu.scheduler import SchedulingStrategy, Workload

    cluster = await _collector.collect()
    workload = Workload(
        workload_id=body.workload_id,
        name=body.name,
        memory_required_mb=body.memory_required_mb,
        gpu_utilization_estimate=body.gpu_utilization_estimate,
        priority=body.priority,
        preemptible=body.preemptible,
    )
    decision = _scheduler.schedule(workload, cluster, SchedulingStrategy(body.strategy))

    if not decision:
        raise HTTPException(status_code=409, detail="No suitable GPU available")

    return {
        "workload_id": decision.workload_id,
        "assigned_device_id": decision.assigned_device_id,
        "assigned_gpu_uuid": decision.assigned_gpu_uuid,
        "strategy": decision.strategy_used,
        "score": decision.score,
        "reasoning": decision.reasoning,
        "alternatives": decision.alternatives,
    }
