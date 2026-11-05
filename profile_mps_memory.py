#!/usr/bin/env python3
"""Profile MPS memory while running a bounded sliding-window inference loop.

The benchmark intentionally uses synthetic tensors so it can be run before
loading the Gaggle checkpoint. On Apple Silicon, run:

    python profile_mps_memory.py

For syntax and control-flow validation on a non-MPS machine:

    python profile_mps_memory.py --allow-cpu
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tracemalloc
from dataclasses import dataclass

import torch


LOGGER = logging.getLogger("profile_mps_memory")
MB = 1024 * 1024
DEFAULT_ITERATIONS = 50
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_HIDDEN_SIZE = 256


@dataclass(frozen=True)
class MemorySnapshot:
    """Memory counters captured at one point in the benchmark."""

    mps_allocated_mb: float
    python_current_mb: float
    python_peak_mb: float


def _mps_allocated_mb() -> float:
    """Return currently live MPS tensor memory, or zero when running on CPU."""

    if not torch.backends.mps.is_available():
        return 0.0
    return torch.mps.current_allocated_memory() / MB


def _snapshot() -> MemorySnapshot:
    python_current, python_peak = tracemalloc.get_traced_memory()
    return MemorySnapshot(
        mps_allocated_mb=_mps_allocated_mb(),
        python_current_mb=python_current / MB,
        python_peak_mb=python_peak / MB,
    )


def _flush_memory(device: torch.device) -> None:
    """Release Python references and the MPS allocator cache."""

    # MPS cache flushing is deliberately explicit after every window. The
    # CPU path exists only to make --allow-cpu validation safe and predictable.
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def _format_delta(value: float, baseline: float) -> str:
    return f"{value:.2f} MB ({value - baseline:+.2f} MB vs baseline)"


@torch.no_grad()
def run_synthetic_benchmark(
    *,
    device: torch.device,
    iterations: int = DEFAULT_ITERATIONS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> None:
    """Run synthetic 512-token inference and report memory per window."""

    # A small layer creates realistic input/output allocations without loading
    # any model weights. It is intentionally kept resident for the whole run.
    dtype = torch.float16 if device.type == "mps" else torch.float32
    projection = torch.nn.Linear(
        hidden_size, hidden_size, bias=False, dtype=dtype
    ).to(device)
    projection.eval()

    _flush_memory(device)
    baseline = _snapshot()
    LOGGER.info(
        "Baseline: MPS allocated=%.2f MB, Python current=%.2f MB, "
        "Python peak=%.2f MB",
        baseline.mps_allocated_mb,
        baseline.python_current_mb,
        baseline.python_peak_mb,
    )

    peak_mps_mb = baseline.mps_allocated_mb
    peak_python_mb = baseline.python_peak_mb

    for iteration in range(1, iterations + 1):
        tracemalloc.reset_peak()
        chunk_tensors = torch.randn(
            (1, chunk_tokens, hidden_size),
            device=device,
            dtype=dtype,
        )
        chunk_output = None
        try:
            # The scalar conversion prevents the output from surviving this
            # iteration while still exercising a real inference pass.
            chunk_output = projection(chunk_tensors)
            _ = chunk_output.mean().item()
            after_inference = _snapshot()
            peak_mps_mb = max(peak_mps_mb, after_inference.mps_allocated_mb)
            peak_python_mb = max(peak_python_mb, after_inference.python_peak_mb)
        finally:
            # Do not let chunk inputs or outputs accumulate across windows.
            del chunk_tensors
            del chunk_output
            _flush_memory(device)

        post_chunk = _snapshot()
        peak_mps_mb = max(peak_mps_mb, post_chunk.mps_allocated_mb)
        peak_python_mb = max(peak_python_mb, post_chunk.python_peak_mb)
        LOGGER.info(
            "Chunk %02d/%02d: post MPS=%s, Python current=%s, "
            "Python peak=%.2f MB",
            iteration,
            iterations,
            _format_delta(post_chunk.mps_allocated_mb, baseline.mps_allocated_mb),
            _format_delta(post_chunk.python_current_mb, baseline.python_current_mb),
            post_chunk.python_peak_mb,
        )

    # The projection itself is an intentional long-lived allocation. Delete it
    # before the final snapshot so the reported post-run value reflects cleanup.
    del projection
    _flush_memory(device)
    final = _snapshot()
    LOGGER.info(
        "Summary: MPS peak=%.2f MB, Python peak=%.2f MB, "
        "final MPS=%.2f MB, final Python current=%.2f MB",
        peak_mps_mb,
        peak_python_mb,
        final.mps_allocated_mb,
        final.python_current_mb,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run the synthetic benchmark on CPU when MPS is unavailable.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    args = parser.parse_args()
    if args.iterations < 1 or args.chunk_tokens < 1 or args.hidden_size < 1:
        parser.error("iterations, chunk-tokens, and hidden-size must be positive")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parse_args()
    mps_available = torch.backends.mps.is_available()
    LOGGER.info("PyTorch version: %s", torch.__version__)
    LOGGER.info("MPS available: %s", mps_available)

    if mps_available:
        device = torch.device("mps")
    elif args.allow_cpu:
        LOGGER.warning("MPS is unavailable; running the synthetic benchmark on CPU.")
        device = torch.device("cpu")
    else:
        LOGGER.error(
            "MPS is unavailable. Run on Apple Silicon with an MPS-enabled "
            "PyTorch build, or pass --allow-cpu for a CPU-only smoke test."
        )
        return 2

    tracemalloc.start()
    try:
        run_synthetic_benchmark(
            device=device,
            iterations=args.iterations,
            chunk_tokens=args.chunk_tokens,
            hidden_size=args.hidden_size,
        )
    finally:
        tracemalloc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
