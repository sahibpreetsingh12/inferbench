from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GpuSample:
    timestamp: float
    memory_used_bytes: int
    memory_total_bytes: int
    gpu_utilization_pct: int
    memory_utilization_pct: int
    power_watts: float
    temperature_c: int


@dataclass
class GpuMonitorResult:
    samples: list[GpuSample] = field(default_factory=list)
    peak_memory_bytes: int = 0
    avg_gpu_utilization_pct: float = 0.0
    avg_power_watts: float = 0.0


# Full GpuMonitor implementation comes in a future session
class GpuMonitor:
    pass
