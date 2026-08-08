"""Inference latency and memory benchmarking."""

from __future__ import annotations

import statistics
import time

import torch

from forgesight.models.patch_memory import PatchMemory


def benchmark_inference(
    model: PatchMemory,
    sample: torch.Tensor,
    *,
    warmup: int = 3,
    iterations: int = 20,
) -> dict[str, float]:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    for _ in range(warmup):
        model.predict(sample)
    timings_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(sample)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    sorted_times = sorted(timings_ms)
    p95_index = min(len(sorted_times) - 1, int(round(0.95 * (len(sorted_times) - 1))))
    median_ms = statistics.median(timings_ms)
    memory_mb = model.memory_bank.numel() * model.memory_bank.element_size() / (1024**2)
    return {
        "latency_median_ms": float(median_ms),
        "latency_p95_ms": float(sorted_times[p95_index]),
        "images_per_second": float(sample.shape[0] * 1000.0 / median_ms),
        "memory_bank_mb": float(memory_mb),
    }
