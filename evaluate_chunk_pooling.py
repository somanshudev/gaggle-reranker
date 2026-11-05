#!/usr/bin/env python3
"""Compare aggregation strategies for overlapping long-document chunks.

The synthetic benchmark uses document/chunk/token score tensors, so it can be
run without downloading the Gaggle checkpoint:

    python evaluate_chunk_pooling.py
    python evaluate_chunk_pooling.py --allow-cpu --documents 8 --chunks 6
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass

import torch


LOGGER = logging.getLogger("evaluate_chunk_pooling")
DEFAULT_DOCUMENTS = 16
DEFAULT_CHUNKS = 8
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64


@dataclass(frozen=True)
class PoolingResult:
    """Aggregated scores for one pooling strategy."""

    name: str
    scores: torch.Tensor


def _flush_memory(device: torch.device) -> None:
    """Release temporary tensors and flush the MPS allocator when applicable."""

    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


@torch.no_grad()
def max_pooling(chunk_scores: torch.Tensor) -> torch.Tensor:
    """Return the maximum chunk score for every document."""

    if chunk_scores.ndim != 2:
        raise ValueError("chunk_scores must have shape [documents, chunks]")
    return torch.max(chunk_scores, dim=1).values


@torch.no_grad()
def mean_pooling(chunk_scores: torch.Tensor) -> torch.Tensor:
    """Return the arithmetic mean chunk score for every document."""

    if chunk_scores.ndim != 2:
        raise ValueError("chunk_scores must have shape [documents, chunks]")
    return torch.mean(chunk_scores, dim=1)


@torch.no_grad()
def _central_token_weights(
    token_count: int,
    boundary_tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a symmetric ramp that down-weights each chunk's boundaries."""

    if token_count < 1:
        raise ValueError("token_count must be positive")
    if not 0 <= boundary_tokens < token_count // 2:
        raise ValueError("boundary_tokens must be >= 0 and less than half the chunk")

    positions = torch.arange(token_count, device=device, dtype=dtype)
    center = (token_count - 1) / 2
    distance = torch.abs(positions - center) / (center or 1)
    # A full central plateau remains at weight 1; boundary ramps taper to 0.25.
    ramp_start = 1.0 - (2.0 * boundary_tokens / token_count)
    weights = torch.where(
        distance <= ramp_start,
        torch.ones_like(distance),
        1.0 - 0.75 * ((distance - ramp_start) / (1.0 - ramp_start)),
    )
    return weights.clamp_min(0.25)


@torch.no_grad()
def weighted_overlap_pooling(
    token_scores: torch.Tensor,
    *,
    boundary_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> torch.Tensor:
    """Pool token scores with central-token emphasis across overlapping chunks.

    ``token_scores`` has shape ``[documents, chunks, tokens]``. Each chunk is
    first reduced with a center-weighted token average, then chunk scores are
    weighted by the same center emphasis. This suppresses boundary artifacts
    without a Python loop or materializing an expanded document tensor.
    """

    if token_scores.ndim != 3:
        raise ValueError("token_scores must have shape [documents, chunks, tokens]")
    _, chunk_count, token_count = token_scores.shape
    token_weights = _central_token_weights(
        token_count,
        boundary_tokens,
        device=token_scores.device,
        dtype=token_scores.dtype,
    )
    chunk_scores = torch.sum(
        token_scores * token_weights.view(1, 1, -1), dim=2
    ) / token_weights.sum()

    # Central chunks receive more weight than the first/last chunks, which
    # contain less context on one side of the document.
    chunk_positions = torch.arange(
        chunk_count, device=token_scores.device, dtype=token_scores.dtype
    )
    chunk_center = (chunk_count - 1) / 2
    chunk_distance = torch.abs(chunk_positions - chunk_center) / (chunk_center or 1)
    chunk_weights = (1.0 - 0.5 * chunk_distance).clamp_min(0.5)
    return torch.sum(chunk_scores * chunk_weights.view(1, -1), dim=1) / chunk_weights.sum()


@torch.no_grad()
def _synthetic_scores(
    *,
    documents: int,
    chunks: int,
    chunk_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create reproducible chunk and token scores on the selected device."""

    generator = torch.Generator(device=device).manual_seed(7)
    token_scores = torch.rand(
        (documents, chunks, chunk_tokens), device=device, generator=generator
    )
    # Chunk scores represent a normal reranker output and are used by max/mean.
    chunk_scores = token_scores.mean(dim=2)
    return chunk_scores, token_scores


@torch.no_grad()
def run_synthetic_evaluation(
    *,
    device: torch.device,
    documents: int,
    chunks: int,
    chunk_tokens: int,
    overlap_tokens: int,
) -> None:
    """Evaluate all methods and print comparable score statistics."""

    chunk_scores, token_scores = _synthetic_scores(
        documents=documents,
        chunks=chunks,
        chunk_tokens=chunk_tokens,
        device=device,
    )
    results: tuple[PoolingResult, ...] = ()
    try:
        results = (
            PoolingResult("max", max_pooling(chunk_scores)),
            PoolingResult("mean", mean_pooling(chunk_scores)),
            PoolingResult(
                "weighted-overlap",
                weighted_overlap_pooling(
                    token_scores, boundary_tokens=overlap_tokens
                ),
            ),
        )
        reference = results[1].scores
        LOGGER.info(
            "Synthetic scores: documents=%d, chunks=%d, tokens/chunk=%d, overlap=%d",
            documents,
            chunks,
            chunk_tokens,
            overlap_tokens,
        )
        LOGGER.info("method              mean       variance    max |shift vs mean|")
        for result in results:
            shifts = torch.abs(result.scores - reference)
            LOGGER.info(
                "%-18s %.6f   %.6f    %.6f",
                result.name,
                torch.mean(result.scores).item(),
                torch.var(result.scores, unbiased=False).item(),
                torch.max(shifts).item(),
            )
        LOGGER.info(
            "Ranking stability: max-vs-mean shift=%.6f, weighted-vs-mean shift=%.6f",
            torch.mean(torch.abs(results[0].scores - reference)).item(),
            torch.mean(torch.abs(results[2].scores - reference)).item(),
        )
    finally:
        # Keep no per-document/chunk tensors alive after the evaluation.
        del results
        del chunk_scores
        del token_scores
        _flush_memory(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()
    if min(args.documents, args.chunks, args.chunk_tokens) < 1:
        parser.error("documents, chunks, and chunk-tokens must be positive")
    if args.overlap_tokens < 0 or args.overlap_tokens >= args.chunk_tokens // 2:
        parser.error("overlap-tokens must be >= 0 and less than half of chunk-tokens")
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
        LOGGER.warning("MPS unavailable; using CPU for the synthetic evaluation.")
        device = torch.device("cpu")
    else:
        LOGGER.error("MPS unavailable; pass --allow-cpu for a CPU smoke test.")
        return 2

    run_synthetic_evaluation(
        device=device,
        documents=args.documents,
        chunks=args.chunks,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
