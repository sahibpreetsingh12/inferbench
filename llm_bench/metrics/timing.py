from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """All timing data for a single inference request."""

    request_id: str
    backend: str
    concurrency: int

    # Raw timestamps (time.perf_counter seconds)
    t_request_sent: float = 0.0
    t_first_token: float = 0.0
    t_last_token: float = 0.0
    token_timestamps: list[float] = field(default_factory=list)

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Derived metrics (filled in by compute_request_metrics)
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    itl_values_ms: list[float] = field(default_factory=list)
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0

    error: str | None = None


def compute_request_metrics(m: RequestMetrics) -> RequestMetrics:
    """
    Fill derived fields from raw timestamps.
    Call this once after all tokens have been collected.
    """
    if m.error or not m.token_timestamps:
        return m

    # Time to first token: how long the user waited before seeing anything
    m.ttft_s = m.t_first_token - m.t_request_sent

    # Total end-to-end latency
    m.total_time_s = m.t_last_token - m.t_request_sent

    # Time per output token: average decode speed after the first token
    n = m.completion_tokens
    if n > 1:
        m.tpot_s = (m.t_last_token - m.t_first_token) / (n - 1)

    # Inter-token latency: gap between consecutive tokens (in ms)
    ts = m.token_timestamps
    m.itl_values_ms = [(ts[i + 1] - ts[i]) * 1000.0 for i in range(len(ts) - 1)]

    # Tokens per second from the user's perspective (full request duration)
    if m.total_time_s > 0:
        m.tokens_per_second = n / m.total_time_s

    return m
