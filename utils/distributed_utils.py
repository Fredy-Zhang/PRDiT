#!/usr/bin/env python3
"""Helpers for lower-overhead distributed training utilities."""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def _is_distributed_ready() -> bool:
    """Return whether `torch.distributed` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def _get_default_device() -> torch.device:
    """Return a reasonable default device for the current process."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class DistributedMetricsCollector:
    """Track per-rank metrics locally to avoid frequent collective communication."""

    def __init__(self, rank: int, world_size: int, collect_freq: int = 100, max_history: int = 1000):
        self.rank = rank
        self.world_size = world_size
        self.collect_freq = collect_freq
        self.max_history = max_history
        self.local_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.global_step = 0

    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Store a local scalar metric value."""
        if step is None:
            step = self.global_step
        self.local_metrics[name].append((step, float(value)))

    def get_local_average(self, name: str, window: int = 100) -> float:
        """Return the recent arithmetic mean for one metric."""
        if name not in self.local_metrics or not self.local_metrics[name]:
            return 0.0
        recent_values = [value for _, value in list(self.local_metrics[name])[-window:]]
        return float(sum(recent_values) / len(recent_values))

    def get_smoothed_metric(self, name: str, alpha: float = 0.9) -> float:
        """Return an exponentially smoothed value for one metric."""
        if name not in self.local_metrics or not self.local_metrics[name]:
            return 0.0

        values = [value for _, value in self.local_metrics[name]]
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * smoothed + (1.0 - alpha) * value
        return float(smoothed)

    def should_collect_global(self) -> bool:
        """Return whether the current step hits the configured collection interval."""
        self.global_step += 1
        return self.global_step % self.collect_freq == 0


class AsyncCheckpointManager:
    """Checkpoint helper with bounded retention for epoch checkpoints."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_checkpoint: Optional[Path] = None

    def save_checkpoint_async(
        self,
        state_dict: dict[str, Any],
        filename: str,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint immediately and prune older epoch checkpoints."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(state_dict, checkpoint_path)

        if is_best:
            self.best_checkpoint = checkpoint_path
            torch.save(state_dict, self.checkpoint_dir / "best_model.pt")

        self._cleanup_old_checkpoints()
        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the most recent epoch checkpoints."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        for old_file in checkpoint_files[:-self.max_checkpoints]:
            try:
                old_file.unlink()
                logger.debug("Removed old checkpoint: %s", old_file)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", old_file, exc)


class EfficientDistributedTrainer:
    """Thin wrapper for low-synchronization distributed training helpers."""

    def __init__(self, rank: int, world_size: int, device: torch.device, collect_freq: int = 100):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_distributed = world_size > 1
        self.metrics = DistributedMetricsCollector(rank, world_size, collect_freq=collect_freq)
        self.sync_freq = max(1, world_size * 10)
        self.last_sync_step = 0

    def log_metrics(self, metrics_dict: dict[str, float], step: int) -> None:
        """Track a batch of local metrics without synchronizing across ranks."""
        for name, value in metrics_dict.items():
            self.metrics.add_metric(name, value, step)

    def get_current_metrics(self) -> dict[str, float]:
        """Return smoothed local metrics for logging."""
        return {
            name: self.metrics.get_smoothed_metric(name)
            for name in self.metrics.local_metrics.keys()
        }

    def should_sync(self, step: int) -> bool:
        """Return whether a barrier should be triggered at this step."""
        if not self.is_distributed:
            return False
        if step - self.last_sync_step >= self.sync_freq:
            self.last_sync_step = step
            return True
        return False

    def minimal_sync(self) -> None:
        """Run a barrier only when distributed training is active."""
        if self.is_distributed and _is_distributed_ready():
            dist.barrier()


def setup_efficient_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training when requested and return rank/world/device."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, _get_default_device()

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")

    if not _is_distributed_ready():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=torch.distributed.default_pg_timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        device_index = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    if hasattr(torch.distributed, "set_debug_level"):
        torch.distributed.set_debug_level(torch.distributed.DebugLevel.OFF)

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Destroy the process group when distributed training is active."""
    if _is_distributed_ready():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            logger.warning("Error during distributed cleanup: %s", exc)


class LossTracker:
    """Track recent and smoothed loss values on a single rank."""

    def __init__(self, window_size: int = 100, smooth_factor: float = 0.95):
        self.window_size = window_size
        self.smooth_factor = smooth_factor
        self.losses = deque(maxlen=window_size)
        self.smoothed_loss: Optional[float] = None

    def add_loss(self, loss: float) -> None:
        """Record a new loss value."""
        loss = float(loss)
        self.losses.append(loss)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self.smooth_factor * self.smoothed_loss + (1.0 - self.smooth_factor) * loss

    def get_average(self) -> float:
        """Return the average loss over the sliding window."""
        return float(sum(self.losses) / len(self.losses)) if self.losses else 0.0

    def get_smoothed(self) -> float:
        """Return the exponentially smoothed loss."""
        return float(self.smoothed_loss) if self.smoothed_loss is not None else 0.0

    def get_recent(self, n: int = 10) -> list[float]:
        """Return the most recent `n` loss values."""
        return list(self.losses)[-n:]


def optimize_ddp_model(
    model: torch.nn.Module,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Wrap a model with DDP when distributed training is active."""
    if not _is_distributed_ready():
        return model

    if torch.cuda.is_available():
        device_index = dist.get_rank() % torch.cuda.device_count()
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_index],
            output_device=device_index,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=not find_unused_parameters,
            bucket_cap_mb=100,
        )

    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=not find_unused_parameters,
        bucket_cap_mb=100,
    )


def efficient_seed_setup(base_seed: int) -> int:
    """Set deterministic per-rank seeds and return the rank-adjusted seed."""
    if _is_distributed_ready():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        seed = base_seed * world_size + rank
    else:
        seed = base_seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


if __name__ == "__main__":
    rank, world_size, device = setup_efficient_distributed()
    print(f"Efficient distributed setup: rank {rank}/{world_size} on {device}")

    trainer = EfficientDistributedTrainer(rank, world_size, device)
    trainer.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)

    cleanup_distributed()
