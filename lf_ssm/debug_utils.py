"""
Debug utilities for LF-SSM training and inference.

Provides timing, tensor inspection, GPU memory reporting, and model
profiling helpers to diagnose slow training and numerical issues.

Usage
=====
>>> from lf_ssm.debug_utils import DebugTimer, log_tensor_stats, log_model_summary
>>> with DebugTimer("Forward pass"):
...     cls, bbox = model(template, search)
>>> log_tensor_stats("cls_score", cls)
"""

import time
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor


# ── Timer ──────────────────────────────────────────────────────────


class DebugTimer:
    """Context manager that measures and prints wall-clock time.

    Parameters
    ----------
    label : str
        Description of the timed section.
    enabled : bool
        If False, the timer is a no-op (zero overhead in production).

    Example
    -------
    >>> with DebugTimer("Forward pass"):
    ...     output = model(x)
    [Timer] Forward pass: 1.234s
    """

    def __init__(self, label: str = "", enabled: bool = True):
        self.label = label
        self.enabled = enabled
        self.elapsed: float = 0.0

    def __enter__(self):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self._start
            print(f"[Timer] {self.label}: {self.elapsed:.4f}s")
        return False


@contextmanager
def silent_timer():
    """Timer that records elapsed time without printing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    class _Result:
        elapsed = 0.0

    result = _Result()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result.elapsed = time.perf_counter() - start


# ── Tensor inspection ──────────────────────────────────────────────


def log_tensor_stats(name: str, t: Tensor) -> None:
    """Print shape, dtype, and basic statistics of a tensor.

    Parameters
    ----------
    name : str
        Human-readable name for the tensor.
    t : Tensor
        The tensor to inspect.

    Example
    -------
    >>> log_tensor_stats("cls_score", cls)
    [Tensor] cls_score | shape=torch.Size([1, 1, 16, 16]) dtype=float32
             min=-2.34 max=1.87 mean=0.12 std=0.45 has_nan=False has_inf=False
    """
    with torch.no_grad():
        t_float = t.float()
        print(
            f"[Tensor] {name} | shape={t.shape} dtype={t.dtype}\n"
            f"         min={t_float.min().item():.4f} "
            f"max={t_float.max().item():.4f} "
            f"mean={t_float.mean().item():.4f} "
            f"std={t_float.std().item():.4f} "
            f"has_nan={torch.isnan(t).any().item()} "
            f"has_inf={torch.isinf(t).any().item()}"
        )


def check_gradients(model: nn.Module, top_k: int = 5) -> None:
    """Print gradient statistics for the top-k largest/smallest grad norms.

    Call after ``loss.backward()`` to verify gradient health.

    Parameters
    ----------
    model : nn.Module
    top_k : int
        Number of parameters to show from each end.
    """
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))

    if not grad_norms:
        print("[Grad] No gradients computed yet.")
        return

    grad_norms.sort(key=lambda x: x[1])
    print(f"[Grad] {len(grad_norms)} parameters with gradients")
    print(f"  Smallest {top_k} grad norms:")
    for name, norm in grad_norms[:top_k]:
        print(f"    {norm:.6e}  {name}")
    print(f"  Largest {top_k} grad norms:")
    for name, norm in grad_norms[-top_k:]:
        print(f"    {norm:.6e}  {name}")

    # Check for vanishing / exploding
    all_norms = [n for _, n in grad_norms]
    n_zero = sum(1 for n in all_norms if n == 0)
    n_large = sum(1 for n in all_norms if n > 100)
    n_nan = sum(1 for _, p in model.named_parameters()
                if p.grad is not None and torch.isnan(p.grad).any())
    if n_zero > 0:
        print(f"  ⚠️  {n_zero} parameters have ZERO gradients")
    if n_large > 0:
        print(f"  ⚠️  {n_large} parameters have grad norm > 100")
    if n_nan > 0:
        print(f"  ❌ {n_nan} parameters have NaN gradients!")


# ── GPU memory ─────────────────────────────────────────────────────


def log_gpu_memory() -> None:
    """Print current and peak GPU memory usage (CUDA only)."""
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available — running on CPU")
        return
    current = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(
        f"[GPU] Memory — Current: {current:.1f} MB | "
        f"Peak: {peak:.1f} MB | Reserved: {reserved:.1f} MB"
    )


# ── Model summary ─────────────────────────────────────────────────


def log_model_summary(model: nn.Module) -> None:
    """Print a parameter breakdown by top-level sub-module.

    Parameters
    ----------
    model : nn.Module
        The model to summarise.
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    total = 0
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        total += count
        print(f"  {name:25s}: {count:>12,d}  ({count / 1e6:.2f}M)")
    print(f"  {'TOTAL':25s}: {total:>12,d}  ({total / 1e6:.2f}M)")
    print("=" * 60)


# ── Forward-pass profiler ─────────────────────────────────────────


def profile_forward_pass(
    model,
    template: Tensor,
    search: Tensor,
) -> dict[str, float]:
    """Time each stage of LFSSM.forward() and print a breakdown.

    Parameters
    ----------
    model : LFSSM
        The LF-SSM model (must have template_embed, search_embed, blocks, norm, head).
    template : Tensor, shape (B, C, H_t, W_t)
    search : Tensor, shape (B, C, H_s, W_s)

    Returns
    -------
    dict mapping stage name → elapsed seconds.
    """
    timings: dict[str, float] = {}

    print("=" * 60)
    print("FORWARD PASS PROFILING")
    print(f"  Template:  {tuple(template.shape)}")
    print(f"  Search:    {tuple(search.shape)}")
    print(f"  Device:    {next(model.parameters()).device}")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # 1. Patch embedding
        with silent_timer() as t:
            z_t = model.template_embed(template)
        timings["template_embed"] = t.elapsed
        print(f"  template_embed : {t.elapsed:.4f}s → {tuple(z_t.shape)}")

        with silent_timer() as t:
            z_s = model.search_embed(search)
        timings["search_embed"] = t.elapsed
        print(f"  search_embed   : {t.elapsed:.4f}s → {tuple(z_s.shape)}")

        # 2. Concatenate
        z = torch.cat([z_t, z_s], dim=1)
        print(f"  concatenated   : {tuple(z.shape)}")

        # 3. GSM Blocks
        total_blocks = 0.0
        for i, blk in enumerate(model.blocks):
            with silent_timer() as t:
                z = blk(z)
            timings[f"block_{i}"] = t.elapsed
            total_blocks += t.elapsed
            print(f"  block_{i:02d}       : {t.elapsed:.4f}s")

        timings["all_blocks"] = total_blocks
        print(f"  ── all blocks  : {total_blocks:.4f}s (avg {total_blocks / len(model.blocks):.4f}s/block)")

        # 4. Final norm
        with silent_timer() as t:
            z = model.norm(z)
        timings["norm"] = t.elapsed

        # 5. Extract search tokens + head
        z_search = z[:, model.cfg.num_patches_template:, :]
        feat_size = model.cfg.search_feat_size
        z_spatial = z_search.transpose(1, 2).reshape(
            -1, model.cfg.embed_dim, feat_size, feat_size
        )

        with silent_timer() as t:
            cls, bbox = model.head(z_spatial)
        timings["head"] = t.elapsed
        print(f"  head           : {t.elapsed:.4f}s")

        total = sum(timings[k] for k in timings if k != "all_blocks")
        timings["total"] = total
        print(f"  ── TOTAL       : {total:.4f}s")

        # Bottleneck analysis
        block_pct = total_blocks / total * 100 if total > 0 else 0
        print(f"\n  ⚡ GSM Blocks account for {block_pct:.1f}% of forward time")
        tokens = model.cfg.num_patches_total
        loops = len(model.blocks) * 2 * tokens
        print(f"  ⚡ Sequential loop iterations per sample: {loops:,d}")
        print(f"     ({len(model.blocks)} blocks × 2 directions × {tokens} tokens)")
        print("=" * 60)

    return timings


# ── Epoch progress formatter ──────────────────────────────────────


class EpochLogger:
    """Tracks and prints per-batch statistics during an epoch.

    Parameters
    ----------
    num_batches : int
        Total number of batches in the epoch.
    epoch : int
        Current epoch number (for display).
    print_every : int
        Print a progress line every N batches. Default 5.
    """

    def __init__(self, num_batches: int, epoch: int, print_every: int = 5):
        self.num_batches = num_batches
        self.epoch = epoch
        self.print_every = print_every
        self.batch_times: list[float] = []
        self.losses: dict[str, list[float]] = {}
        self._epoch_start = time.perf_counter()

    def log_batch(
        self,
        batch_idx: int,
        batch_time: float,
        loss_dict: dict[str, float],
    ) -> None:
        """Record a batch and optionally print progress.

        Parameters
        ----------
        batch_idx : int
            0-based batch index.
        batch_time : float
            Wall-clock time for this batch in seconds.
        loss_dict : dict
            Loss components, e.g. {'total': 1.23, 'cls': 0.5, ...}.
        """
        self.batch_times.append(batch_time)
        for k, v in loss_dict.items():
            self.losses.setdefault(k, []).append(v)

        should_print = (
            batch_idx == 0  # Always print first batch
            or (batch_idx + 1) % self.print_every == 0
            or (batch_idx + 1) == self.num_batches  # Always print last
        )

        if should_print:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            eta = avg_time * (self.num_batches - batch_idx - 1)
            loss_str = " | ".join(
                f"{k}={sum(v) / len(v):.4f}" for k, v in self.losses.items()
            )
            pct = (batch_idx + 1) / self.num_batches * 100
            print(
                f"  Epoch {self.epoch} [{batch_idx + 1}/{self.num_batches} "
                f"({pct:.0f}%)] "
                f"batch_time={batch_time:.2f}s avg={avg_time:.2f}s "
                f"ETA={eta:.0f}s | {loss_str}"
            )
            sys.stdout.flush()

            # Extra diagnostics on first batch
            if batch_idx == 0:
                log_gpu_memory()

    def summarise(self) -> dict[str, float]:
        """Print epoch summary and return average losses."""
        elapsed = time.perf_counter() - self._epoch_start
        avg_losses = {
            k: sum(v) / len(v) for k, v in self.losses.items()
        }
        avg_time = sum(self.batch_times) / len(self.batch_times)
        throughput = self.num_batches / elapsed if elapsed > 0 else 0

        print(f"  ── Epoch {self.epoch} done in {elapsed:.1f}s "
              f"({avg_time:.2f}s/batch, {throughput:.1f} batches/s) ──")
        return avg_losses
