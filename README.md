# InferBench

> Cross-backend LLM inference benchmarking with GPU profiling.

## Vision

Most inference benchmarking tools are backend-specific — vLLM benchmarks vLLM, SGLang benchmarks SGLang. **InferBench** takes one config and runs the same workload across **HuggingFace naive**, **vLLM**, and **SGLang**, collecting identical metrics from every backend so you can make an apples-to-apples comparison.

The goal: a single CLI command that answers *"which serving backend is fastest for my model and hardware?"*

```bash
inferbench run configs/example.yaml
```

Outputs:
- TTFT / TPOT / ITL at P50, P90, P99
- Throughput (tok/s) across concurrency levels 1→N
- GPU memory, utilization, and power over time
- Self-contained HTML report with interactive charts

## Target Hardware

Designed and tested on **NVIDIA RTX 5090 (32GB)**. Works with any CUDA GPU that supports the chosen backend.

## What's Been Built (Day 1)

| Module | Status | What it does |
|--------|--------|--------------|
| `llm_bench/config.py` | ✅ | Pydantic config models + YAML loader |
| `llm_bench/metrics/timing.py` | ✅ | `RequestMetrics` dataclass, TTFT/TPOT/ITL computation |
| `llm_bench/metrics/aggregator.py` | ✅ | P50/P90/P99 aggregation across requests |
| `llm_bench/metrics/gpu_monitor.py` | 🔲 stub | Data structures ready, pynvml sampler coming |
| `configs/example.yaml` | ✅ | Example config for Qwen2.5-0.5B with vLLM |

## Roadmap

- [ ] GPU monitor (pynvml background sampler)
- [ ] KV-cache estimator (theoretical + empirical)
- [ ] Backends: HuggingFace naive, vLLM, SGLang
- [ ] Concurrent load generator (asyncio semaphore)
- [ ] Benchmark runner / orchestrator
- [ ] HTML report (Plotly + Jinja2)
- [ ] CLI (`inferbench run`, `inferbench validate`)
- [ ] Test suite

## Quick Start (once complete)

```bash
git clone https://github.com/sahibpreetsingh12/inferbench
cd inferbench
uv sync
inferbench run configs/example.yaml --output results/
```

## Metrics Explained

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| **TTFT** | `t_first_token - t_request_sent` | User-perceived wait before anything appears |
| **TPOT** | `(t_last - t_first) / (n_tokens - 1)` | Decode throughput per user |
| **ITL** | `t[i+1] - t[i]` per token | Scheduling jitter — spikes mean batching stalls |
| **Throughput** | `total_tokens / wall_clock_s` | System-wide capacity |

## License

Apache 2.0
