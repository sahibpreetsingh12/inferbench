from __future__ import annotations
import statistics
from dataclasses import dataclass
from .timing import RequestMetrics
from .gpu_monitor import GpuMonitorResult


def _percentile(values: list[float], p: float) -> float:
    """Linear interpolation percentile (same method as numpy's default)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


@dataclass
class AggregatedMetrics:
    """Statistical summary for one (backend, concurrency) combination."""

    backend: str
    concurrency: int
    num_requests: int

    # TTFT — Time to First Token (seconds)
    ttft_mean: float = 0.0
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0

    # TPOT — Time Per Output Token (seconds)
    tpot_mean: float = 0.0
    tpot_p90: float = 0.0
    tpot_p99: float = 0.0

    # ITL — Inter-Token Latency (milliseconds)
    itl_mean: float = 0.0
    itl_p90: float = 0.0
    itl_p99: float = 0.0

    # Throughput
    total_tokens: int = 0
    wall_clock_s: float = 0.0
    throughput_tok_per_s: float = 0.0
    avg_tokens_per_s_per_request: float = 0.0

    # GPU (populated if a GpuMonitorResult is passed)
    gpu_peak_memory_mb: float = 0.0
    gpu_avg_utilization_pct: float = 0.0
    gpu_avg_power_w: float = 0.0

    # Errors
    error_count: int = 0
    error_rate: float = 0.0


def aggregate_metrics(
    requests: list[RequestMetrics],
    wall_clock_s: float = 0.0,
    gpu_result: GpuMonitorResult | None = None,
) -> AggregatedMetrics:
    """Aggregate a list of per-request metrics into a statistical summary."""
    if not requests:
        raise ValueError("No requests to aggregate")

    successful = [r for r in requests if r.error is None]
    backend = requests[0].backend
    concurrency = requests[0].concurrency

    ttft_vals = [r.ttft_s for r in successful if r.ttft_s > 0]
    tpot_vals = [r.tpot_s for r in successful if r.tpot_s > 0]
    all_itl: list[float] = []
    for r in successful:
        all_itl.extend(r.itl_values_ms)
    tps_vals = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
    total_tokens = sum(r.completion_tokens for r in successful)

    agg = AggregatedMetrics(
        backend=backend,
        concurrency=concurrency,
        num_requests=len(requests),
        error_count=len(requests) - len(successful),
        error_rate=(len(requests) - len(successful)) / len(requests),
        total_tokens=total_tokens,
        wall_clock_s=wall_clock_s,
    )

    if ttft_vals:
        agg.ttft_mean = statistics.mean(ttft_vals)
        agg.ttft_p50 = _percentile(ttft_vals, 50)
        agg.ttft_p90 = _percentile(ttft_vals, 90)
        agg.ttft_p99 = _percentile(ttft_vals, 99)

    if tpot_vals:
        agg.tpot_mean = statistics.mean(tpot_vals)
        agg.tpot_p90 = _percentile(tpot_vals, 90)
        agg.tpot_p99 = _percentile(tpot_vals, 99)

    if all_itl:
        agg.itl_mean = statistics.mean(all_itl)
        agg.itl_p90 = _percentile(all_itl, 90)
        agg.itl_p99 = _percentile(all_itl, 99)

    if tps_vals:
        agg.avg_tokens_per_s_per_request = statistics.mean(tps_vals)

    if wall_clock_s > 0:
        agg.throughput_tok_per_s = total_tokens / wall_clock_s

    if gpu_result:
        agg.gpu_peak_memory_mb = gpu_result.peak_memory_bytes / (1024**2)
        agg.gpu_avg_utilization_pct = gpu_result.avg_gpu_utilization_pct
        agg.gpu_avg_power_w = gpu_result.avg_power_watts

    return agg
