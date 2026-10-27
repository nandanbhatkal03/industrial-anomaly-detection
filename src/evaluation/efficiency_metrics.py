"""Deployment metrics are first-class (research Gap 5).

IMPORTANT: model size = backbone parameters + fitted artefacts (memory bank / covariances /
projection). Counting only network parameters would hide the real cost of memory-bank and
distribution methods.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch


def peak_rss_mb() -> float | None:
    """Peak resident memory of THIS process (run benchmarks in a fresh process for clean numbers)."""
    try:
        if sys.platform.startswith("win"):
            import psutil
            return psutil.Process(os.getpid()).memory_info().peak_wset / 2**20
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 2**20 if sys.platform == "darwin" else v / 1024
    except Exception:
        return None


def footprint(model, checkpoint_path=None) -> dict:
    params = model.num_parameters()
    art = model.artifact_bytes()
    out = {"params_m": params / 1e6,
           "param_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
           "artifact_mb": art / 2**20}
    out["deployed_size_mb"] = out["param_mb"] + out["artifact_mb"]
    if checkpoint_path and os.path.exists(checkpoint_path):
        out["checkpoint_mb"] = os.path.getsize(checkpoint_path) / 2**20
    return out


def count_flops(model, input_size: int, device) -> float | None:
    """GFLOPs of one forward pass for ops PyTorch can count (conv/matmul). kNN/cdist is not included."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, 3, input_size, input_size, device=device)
        with FlopCounterMode(display=False) as fc:
            model.predict(x)
        return fc.get_total_flops() / 1e9
    except Exception:
        return None


@torch.no_grad()
def benchmark_latency(model, device, input_size: int, warmup: int = 10, iters: int = 50, batch_size: int = 1) -> dict:
    model.eval().to(device)
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model.predict(x)
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    out = {"latency_ms_mean": float(t.mean()), "latency_ms_p50": float(np.percentile(t, 50)),
           "latency_ms_p95": float(np.percentile(t, 95)), "fps": float(batch_size * 1000 / t.mean()),
           "bench_batch_size": batch_size, "device": str(device), "threads": torch.get_num_threads()}
    if device.type == "cuda":
        out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    out["peak_rss_mb"] = peak_rss_mb()
    return out
