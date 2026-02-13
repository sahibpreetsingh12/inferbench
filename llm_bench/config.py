from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.90


class BackendConfig(BaseModel):
    name: Literal["hf_naive", "vllm", "sglang"]
    host: str = "127.0.0.1"
    port: int = 8000
    extra_args: list[str] = Field(default_factory=list)


class LoadConfig(BaseModel):
    concurrency_levels: list[int] = Field(default=[1, 2, 4, 8, 16])
    num_requests: int = 50
    prompt_source: Literal["random", "sharegpt"] = "random"
    sharegpt_path: str | None = None
    input_len: int = 256
    output_len: int = 128


class GpuMonitorConfig(BaseModel):
    enabled: bool = True
    sample_interval_ms: int = 100
    device_index: int = 0


class ReportConfig(BaseModel):
    output_dir: str = "results"
    formats: list[Literal["json", "csv", "html"]] = Field(default=["json", "html"])


class BenchmarkConfig(BaseModel):
    model: ModelConfig
    backend: BackendConfig
    load: LoadConfig = Field(default_factory=LoadConfig)
    gpu_monitor: GpuMonitorConfig = Field(default_factory=GpuMonitorConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    warmup_requests: int = 3


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BenchmarkConfig(**raw)
