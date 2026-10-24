import asyncio

import pytest

from nexus.gpu.metrics import MockMetricsCollector
from nexus.gpu.predictor import WorkloadPredictor
from nexus.gpu.scheduler import GPUScheduler, SchedulingStrategy, Workload


@pytest.fixture
def collector():
    return MockMetricsCollector(num_gpus=4)


@pytest.fixture
def scheduler():
    return GPUScheduler()


@pytest.fixture
def predictor():
    return WorkloadPredictor()


async def test_mock_collector_returns_metrics(collector):
    cluster = await collector.collect()
    assert cluster.total_gpus == 4
    assert len(cluster.devices) == 4
    assert all(d.memory_total_mb > 0 for d in cluster.devices)


async def test_mock_collector_has_utilization(collector):
    cluster = await collector.collect()
    for device in cluster.devices:
        assert 0 <= device.utilization_gpu <= 100
        assert device.temperature_c > 0


async def test_scheduler_assigns_workload(scheduler, collector):
    cluster = await collector.collect()
    workload = Workload(
        workload_id="test-1",
        name="test workload",
        memory_required_mb=4000,
        gpu_utilization_estimate=30,
    )
    decision = scheduler.schedule(workload, cluster)
    assert decision is not None
    assert decision.workload_id == "test-1"
    assert 0 <= decision.assigned_device_id < 4


async def test_scheduler_bin_pack_vs_spread(scheduler, collector):
    cluster = await collector.collect()
    workload = Workload(
        workload_id="test-2",
        name="test",
        memory_required_mb=2000,
        gpu_utilization_estimate=20,
    )
    bin_decision = scheduler.schedule(workload, cluster, SchedulingStrategy.BIN_PACK)
    spread_decision = scheduler.schedule(workload, cluster, SchedulingStrategy.SPREAD)
    assert bin_decision is not None
    assert spread_decision is not None


async def test_scheduler_rejects_oversized_workload(scheduler, collector):
    cluster = await collector.collect()
    workload = Workload(
        workload_id="huge",
        name="oversized",
        memory_required_mb=999_999,
        gpu_utilization_estimate=50,
    )
    decision = scheduler.schedule(workload, cluster)
    assert decision is None


async def test_efficiency_scoring(scheduler, collector):
    cluster = await collector.collect()
    efficiency = scheduler.compute_cluster_efficiency(cluster)
    assert "score" in efficiency
    assert "avg_utilization" in efficiency
    assert "recommendations" in efficiency


def test_predictor_records_and_predicts(predictor):
    for i in range(20):
        predictor.record(0, 50 + i * 0.5)

    result = predictor.predict(0, steps_ahead=5)
    assert result.device_id == 0
    assert result.confidence > 0
    assert result.trend in ("rising", "falling", "stable", "volatile", "unknown")


def test_predictor_detects_rising_trend(predictor):
    for i in range(30):
        predictor.record(0, 20 + i * 2)

    result = predictor.predict(0)
    assert result.predicted_utilization > 60
    assert result.trend == "rising"


def test_predictor_anomaly_score(predictor):
    for _ in range(20):
        predictor.record(0, 50)
    predictor.record(0, 95)

    score = predictor.anomaly_score(0)
    assert score > 0.5


async def test_migration_plans(scheduler, collector):
    cluster = await collector.collect()
    plans = scheduler.generate_migration_plans(cluster)
    assert isinstance(plans, list)
    for plan in plans:
        assert plan.source_device_id != plan.target_device_id
        assert plan.risk_level in ("low", "medium", "high")
