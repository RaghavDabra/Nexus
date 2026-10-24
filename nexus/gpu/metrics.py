from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUDevice:
    device_id: int
    name: str
    uuid: str
    memory_total_mb: int
    compute_capability: tuple[int, int]


@dataclass
class GPUMetrics:
    device_id: int
    uuid: str
    timestamp: float
    utilization_gpu: float      # 0-100%
    utilization_memory: float   # 0-100%
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    clock_sm_mhz: int
    clock_mem_mhz: int
    pcie_throughput_tx_kbps: int
    pcie_throughput_rx_kbps: int
    running_processes: list[GPUProcess] = field(default_factory=list)

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def memory_utilization_pct(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @property
    def power_utilization_pct(self) -> float:
        if self.power_limit_w == 0:
            return 0
        return (self.power_draw_w / self.power_limit_w) * 100

    @property
    def is_idle(self) -> bool:
        return self.utilization_gpu < 5 and len(self.running_processes) == 0

    @property
    def is_overloaded(self) -> bool:
        return self.utilization_gpu > 90 or self.memory_utilization_pct > 90


@dataclass
class GPUProcess:
    pid: int
    name: str
    memory_used_mb: int
    gpu_utilization: float


@dataclass
class ClusterMetrics:
    devices: list[GPUMetrics]
    timestamp: float
    cluster_id: str = "default"

    @property
    def total_gpus(self) -> int:
        return len(self.devices)

    @property
    def avg_utilization(self) -> float:
        if not self.devices:
            return 0
        return sum(d.utilization_gpu for d in self.devices) / len(self.devices)

    @property
    def idle_gpus(self) -> list[GPUMetrics]:
        return [d for d in self.devices if d.is_idle]

    @property
    def overloaded_gpus(self) -> list[GPUMetrics]:
        return [d for d in self.devices if d.is_overloaded]

    @property
    def total_memory_mb(self) -> int:
        return sum(d.memory_total_mb for d in self.devices)

    @property
    def used_memory_mb(self) -> int:
        return sum(d.memory_used_mb for d in self.devices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "total_gpus": self.total_gpus,
            "avg_utilization": round(self.avg_utilization, 2),
            "idle_gpus": len(self.idle_gpus),
            "overloaded_gpus": len(self.overloaded_gpus),
            "total_memory_mb": self.total_memory_mb,
            "used_memory_mb": self.used_memory_mb,
            "memory_utilization_pct": round(
                (self.used_memory_mb / max(self.total_memory_mb, 1)) * 100, 2
            ),
            "timestamp": self.timestamp,
        }


class MetricsCollector(ABC):
    @abstractmethod
    async def collect(self) -> ClusterMetrics:
        ...

    @abstractmethod
    async def list_devices(self) -> list[GPUDevice]:
        ...


class MockMetricsCollector(MetricsCollector):
    """Simulates a GPU cluster for development and testing."""

    def __init__(self, num_gpus: int = 8, cluster_id: str = "mock-cluster"):
        self.num_gpus = num_gpus
        self.cluster_id = cluster_id
        self._devices = [
            GPUDevice(
                device_id=i,
                name=random.choice(["NVIDIA A100-SXM4-80GB", "NVIDIA H100-SXM5-80GB", "NVIDIA L40S"]),
                uuid=f"GPU-{i:04d}-mock-{random.randint(1000,9999)}",
                memory_total_mb=81920 if i < num_gpus // 2 else 49152,
                compute_capability=(9, 0) if i < num_gpus // 2 else (8, 9),
            )
            for i in range(num_gpus)
        ]
        self._workload_phase = 0

    async def list_devices(self) -> list[GPUDevice]:
        return self._devices

    async def collect(self) -> ClusterMetrics:
        self._workload_phase += 1
        devices = []

        for dev in self._devices:
            base_util = self._simulate_utilization(dev.device_id)
            num_processes = random.randint(0, 3) if base_util > 10 else 0
            processes = [
                GPUProcess(
                    pid=random.randint(10000, 99999),
                    name=random.choice([
                        "python3 train.py", "tritonserver", "vllm.entrypoints",
                        "nemo_train", "deepspeed", "torchrun",
                    ]),
                    memory_used_mb=random.randint(1000, 20000),
                    gpu_utilization=random.uniform(10, 60),
                )
                for _ in range(num_processes)
            ]
            proc_memory = sum(p.memory_used_mb for p in processes)

            metrics = GPUMetrics(
                device_id=dev.device_id,
                uuid=dev.uuid,
                timestamp=time.time(),
                utilization_gpu=round(base_util + random.uniform(-5, 5), 1),
                utilization_memory=round(base_util * 0.8 + random.uniform(-10, 10), 1),
                memory_used_mb=min(proc_memory + random.randint(500, 2000), dev.memory_total_mb),
                memory_total_mb=dev.memory_total_mb,
                temperature_c=int(40 + base_util * 0.4 + random.randint(-3, 3)),
                power_draw_w=round(100 + base_util * 2.5 + random.uniform(-10, 10), 1),
                power_limit_w=400.0,
                clock_sm_mhz=int(1200 + base_util * 5),
                clock_mem_mhz=int(1593 + random.randint(-50, 50)),
                pcie_throughput_tx_kbps=random.randint(100000, 5000000),
                pcie_throughput_rx_kbps=random.randint(100000, 5000000),
                running_processes=processes,
            )
            devices.append(metrics)

        return ClusterMetrics(
            devices=devices,
            timestamp=time.time(),
            cluster_id=self.cluster_id,
        )

    def _simulate_utilization(self, device_id: int) -> float:
        phase = self._workload_phase % 100
        if device_id < self.num_gpus // 3:
            return min(95, 70 + phase * 0.3 + random.uniform(-5, 5))
        elif device_id < 2 * self.num_gpus // 3:
            return 30 + 40 * abs((phase % 50 - 25) / 25) + random.uniform(-5, 5)
        else:
            return max(0, 10 + random.uniform(-8, 15))
