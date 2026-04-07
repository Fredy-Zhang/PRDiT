"""Training pipeline for diffusion-based denoising models on LIDC-IDRI data.

This module is intentionally slim: general-purpose utilities live in ``util.py``.

Responsibilities
----------------
- ``train_step``     — single forward-backward-optimiser step with AMP support.
- ``evaluate_model`` — validation loop that saves reconstructed sample images.
- ``Trainer``        — stateful coordinator wiring all components together.
- ``main``           — distributed-training entry point called by ``torchrun``.
"""

import argparse
import logging
import os
import random
from contextlib import nullcontext
from copy import deepcopy
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb

from diffusion import loading_diffusion
from models import load_model
from util import (
    Args,
    Config,
    cleanup,
    create_experiment_dirs,
    create_logger,
    initialize_model_with_pretrained,
    initialize_optimizer,
    load_config,
    log_params,
    manage_checkpoints,
    print_optimizer_params,
    requires_grad,
    return_train_val_loaders,
    save_evaluation_samples,
    setup_torch_config,
    setup_wandb,
    update_ema,
    wandb_enabled,
)


# ── Training step ────────────────────────────────────────────────────────────


def train_step(
    model: torch.nn.Module,
    diffusion: Any,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ema: torch.nn.Module,
    device: Any,
    gradient_clip: float,
    scaler: Optional[torch.amp.GradScaler] = None,
    trainable_params=None,
    all_params=None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Optional[Tuple[float, float, float]]:
    """Perform one forward–backward–optimiser step.

    Handles mixed-precision training via ``torch.amp.autocast`` and an optional
    ``GradScaler`` (required for ``float16``; not needed for ``bfloat16``).
    Returns ``None`` and skips the step when the computed loss is non-finite or
    suspiciously large (> 1e5), protecting against NaN propagation.

    Args:
        model:            The (optionally DDP-wrapped) model.
        diffusion:        Diffusion process exposing ``training_losses``.
        x:                Input batch tensor already on *device*.
        optimizer:        AdamW (or compatible) optimiser.
        ema:              Exponential-moving-average shadow model.
        device:           Target CUDA device (``int``, ``str``, or ``torch.device``).
        gradient_clip:    Maximum gradient norm applied before the optimiser step.
        scaler:           ``GradScaler`` for FP16 AMP (``None`` for BF16 / no AMP).
        trainable_params: Parameters clipped for normal (depth > 0) models.
        all_params:       Parameters clipped for depth-0 models (no attention).
        amp_dtype:        ``torch.bfloat16``, ``torch.float16``, or ``None``.

    Returns:
        ``(total_loss, noise_loss, img_loss)`` as Python floats, or ``None`` if
        the step was skipped.
    """
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)

    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    model_module = model.module if hasattr(model, "module") else model
    is_depth_zero = hasattr(model_module, "depth") and model_module.depth == 0
    # depth-0 models have no attention; clip all params to avoid silent no-ops.
    clip_params = all_params if is_depth_zero else trainable_params

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)

    with autocast_ctx:
        loss_dict = diffusion.training_losses(model, x, t)
        if isinstance(loss_dict, dict) and "noise_loss" in loss_dict and "img_loss" in loss_dict:
            noise_loss = loss_dict["noise_loss"].mean()
            img_loss = loss_dict["img_loss"].mean()
            total_loss = noise_loss + img_loss
        else:
            total_loss = loss_dict["loss"].mean() if isinstance(loss_dict, dict) else loss_dict.mean()
            noise_loss = img_loss = total_loss

    total_loss_value = total_loss.detach().float().item()
    if not torch.isfinite(total_loss) or total_loss_value > 1e5:
        return None

    if scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, gradient_clip)
        optimizer.step()

    # Unwrap compiled model before EMA update so state_dict stays portable.
    model_for_ema = model_module._orig_mod if hasattr(model_module, "_orig_mod") else model_module
    update_ema(ema, model_for_ema)
    return total_loss_value, noise_loss.detach().float().item(), img_loss.detach().float().item()


# ── Validation / evaluation ──────────────────────────────────────────────────


def evaluate_model(
    model: torch.nn.Module,
    val_loader: Any,
    diffusion: Any,
    device: Any,
    rank: int,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Run a full validation pass and save example reconstructions to disk.

    On rank 0 the first batch produces:
    - noisy inputs
    - coarse / fine path outputs (when ``depth > 0``)
    - final image reconstructions and ground-truth slices

    Args:
        model:          EMA model to evaluate.
        val_loader:     DataLoader yielding ``{"image": Tensor}`` batches.
        diffusion:      Diffusion process exposing ``q_sample``.
        device:         CUDA device for inference.
        rank:           Process rank (visualisation only on rank 0).
        experiment_dir: Root directory where sample images are saved.
        image_size:     Spatial dimension used by ``save_evaluation_samples``.
        epoch:          Current epoch index (embedded in saved file names).
        logger:         Logger instance.

    Returns:
        Dict with keys ``"eval_loss"``, ``"eval_loss_avg"``, ``"epoch"``.
    """
    model.eval()
    eval_loss = 0.0
    num_batches = 0

    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)

    inner_model = model.module if hasattr(model, "module") else model
    is_depth_zero = hasattr(inner_model, "depth") and inner_model.depth == 0

    # Pre-create visualisation sub-dirs before iterating to avoid redundant calls.
    if rank == 0 and not is_depth_zero:
        for sub in ("coarse_path", "fine_path", "coarse_path_noise", "fine_path_noise"):
            os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["image"].to(device, non_blocking=True)
            t = torch.full((x.shape[0],), 100, device=device)
            samples = diffusion.q_sample(x, t)

            # First batch only: save coarse/fine intermediate paths.
            if i == 0 and not is_depth_zero:
                coarse_out, fine_out = model(samples, t, return_intermediate=True)
                coarse_noise, coarse_img = coarse_out.chunk(2, dim=1)
                fine_noise, fine_img = fine_out.chunk(2, dim=1)

                if rank == 0:
                    save_evaluation_samples(samples, f"{experiment_dir}/noisy", image_size, epoch, 2, logger)
                    save_evaluation_samples(coarse_img, f"{experiment_dir}/coarse_path", image_size, epoch, 2, logger)
                    save_evaluation_samples(fine_img, f"{experiment_dir}/fine_path", image_size, epoch, 2, logger)
                    save_evaluation_samples(coarse_noise, f"{experiment_dir}/coarse_path_noise", image_size, epoch, 2, logger)
                    save_evaluation_samples(fine_noise, f"{experiment_dir}/fine_path_noise", image_size, epoch, 2, logger)
                    logger.info("Saved visualization of both coarse and fine path outputs")

            model_output = model(samples, t)
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            # When the model predicts both noise and image, take the image half.
            if model_output.shape[1] > x.shape[1]:
                _, img_recon = model_output.chunk(2, dim=1)
            else:
                img_recon = model_output

            eval_loss += (img_recon - x).pow(2).mean().item()
            num_batches += 1

            if rank == 0 and i == 0:
                save_evaluation_samples(img_recon, experiment_dir, image_size, epoch, 2, logger)
                save_evaluation_samples(x, f"{experiment_dir}/gt", image_size, epoch, 2, logger)

    avg_loss = eval_loss / num_batches if num_batches > 0 else 0.0
    if rank == 0:
        logger.info(f"Validation Loss (Average): {avg_loss:.4f}")

    model.train()
    return {"eval_loss": eval_loss, "eval_loss_avg": avg_loss, "epoch": epoch}


# ── Trainer ──────────────────────────────────────────────────────────────────


class Trainer:
    """Stateful training coordinator for diffusion models.

    Encapsulates model, EMA, optimiser, data loaders, and the main training /
    evaluation loop.  Designed to be instantiated once per process in a
    ``torchrun`` distributed job.

    Args:
        config: Experiment configuration object.
        rank:   Rank of the current process within the distributed group.
        device: CUDA device index or ``torch.device`` for this process.
        seed:   Per-rank random seed.
        debug:  When ``True``, emit extra diagnostic output during setup.
    """

    def __init__(
        self,
        config: Config,
        rank: int,
        device: Any,
        seed: int,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.rank = rank
        self.device = device
        self.debug = debug
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1
        self._last_eval_loss: Optional[float] = None

        # Cache Args once; reused across _setup_model, _setup_data, and train().
        self.args = Args(config)

        if rank == 0:
            self.experiment_dir, self.checkpoint_dir = create_experiment_dirs(self.args)
        else:
            self.experiment_dir = self.checkpoint_dir = None

        self.logger = create_logger(None if rank != 0 else self.experiment_dir)

        self._setup_wandb()
        self._setup_model()
        self._setup_data(seed=seed)

    # ── Setup helpers ────────────────────────────────────────────────────────

    def _setup_wandb(self) -> None:
        """Initialise wandb on rank 0 when enabled by the config."""
        if self.rank == 0:
            setup_wandb(self.config, self.rank)

    def _setup_model(self) -> None:
        """Instantiate model, EMA, DDP wrapper, diffusion, and optimiser.

        Also handles:
        - Optional ``torch.compile`` with mode adapted to model depth.
        - Pretrained weight initialisation or checkpoint resumption.
        - AMP dtype and ``GradScaler`` selection based on GPU capability.
        """
        self.model = load_model(self.config)
        if hasattr(self.model, "log_config"):
            self.model.log_config(self.rank)

        is_depth_zero = hasattr(self.model, "depth") and self.model.depth == 0
        # Freeze the unused fine head on depth-0 models.
        if (
            is_depth_zero
            and hasattr(self.model, "t_embedder")
            and hasattr(self.model.t_embedder, "fine_head")
        ):
            self.model.t_embedder.fine_head.requires_grad_(False)

        self.model = self.model.to(self.device)

        if getattr(self.config.training, "compile_model", False):
            compile_mode = getattr(self.config.training, "compile_mode", "default")
            # depth-0 models have no dynamic attention control flow.
            actual_mode = "reduce-overhead" if is_depth_zero else compile_mode
            if self.rank == 0:
                self.logger.info(f"Compiling model with torch.compile(mode='{actual_mode}')")
            try:
                self.model = torch.compile(self.model, backend="inductor", mode=actual_mode)
            except Exception as exc:
                if self.rank == 0:
                    self.logger.warning(f"torch.compile failed: {exc}. Continuing without compilation.")

        # Build EMA from the uncompiled model so state_dict is portable.
        uncompiled = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        self.ema = deepcopy(uncompiled).to(self.device)
        requires_grad(self.ema, False)

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

        self.diffusion = loading_diffusion(self.config, rank=self.rank)

        if self.rank == 0 and self.debug:
            self.logger.info("\n=== Before state of all block parameters ===")
            log_params(self.model, self.logger)

        pretrained_flag = False
        if self.config.model.from_scratch:
            new_param_names = None
        else:
            if self.rank == 0:
                self.logger.info("Initializing the model with pretrained weights")
                self.logger.info(f"Path: {self.config.model.pretrained_path}")
                self.logger.info("===========================================")
            self.model, pretrained_flag, new_param_names = initialize_model_with_pretrained(
                model=self.model,
                config=self.config,
                device=f"cuda:{self.rank}",
                rank=self.rank,
                gain=0.1,
            )

        if getattr(self.config.model, "checkpoint", None):
            checkpoint = torch.load(
                self.config.model.checkpoint, map_location=f"cuda:{self.rank}"
            )
            self.model.module.load_state_dict(checkpoint["model"])

        self.optimizer = initialize_optimizer(
            self.model,
            self.config,
            pretrained_flag=pretrained_flag,
            new_param_names=new_param_names,
            rank=self.rank,
            debug=self.debug,
        )

        model_module = self.model.module if hasattr(self.model, "module") else self.model
        self.all_params = list(model_module.parameters())
        self.trainable_params = [p for p in model_module.parameters() if p.requires_grad]

        # Select AMP dtype based on hardware support.
        amp_enabled = bool(getattr(self.config.training, "amp", True))
        if amp_enabled and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.amp_dtype: Optional[torch.dtype] = torch.bfloat16
                self.scaler: Optional[torch.amp.GradScaler] = None
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.amp_dtype = None
            self.scaler = None

        if self.rank == 0 and self.debug:
            print_optimizer_params(
                self.optimizer,
                self.model,
                self.config.training.learning_rate,
                self.config.training.fine_tune_lr,
                self.logger,
            )
            self.logger.info("\n=== Final state of all block parameters ===")
            log_params(self.model, self.logger)

    def _setup_data(self, seed: int) -> None:
        """Create train and validation data loaders.

        Args:
            seed: Random seed for the training loader.
        """
        self.train_loader, self.val_loader = return_train_val_loaders(
            self.args, self.rank, self.config, seed, self.debug
        )
        if self.rank == 0:
            self.logger.info(
                f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )
            self.logger.info(
                f"Dataset contains {len(self.train_loader.dataset):,} images "
                f"({self.config.data.path})"
            )

    def _setup_training_state(self) -> None:
        """Finalise state before the epoch loop begins.

        - Synchronises EMA with the current model weights (decay=0).
        - Sets model to train mode and EMA to eval mode.
        - Optionally enables high-precision matmul for flash-attention models.
        """
        model_for_ema = (
            self.model.module._orig_mod
            if hasattr(self.model.module, "_orig_mod")
            else self.model.module
        )
        update_ema(self.ema, model_for_ema, decay=0)
        self.model.train()
        self.ema.eval()

        is_depth_zero = hasattr(self.model.module, "depth") and self.model.module.depth == 0
        if self.config.model.flash_attn and not is_depth_zero:
            if self.rank == 0:
                self.logger.info("Enabling flash attention optimizations for transformer model")
            torch.set_float32_matmul_precision("high")
        elif is_depth_zero and self.rank == 0:
            self.logger.info("Using full precision for depth=0 model (no attention)")

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch: int,
        train_steps: int,
        loss: Optional[float] = None,
        save_optimizer: bool = False,
    ) -> None:
        """Persist a model checkpoint to ``self.checkpoint_dir``.

        Only executes on rank 0.  Includes model, EMA, config, epoch, step
        count, and optionally the optimiser state.

        Args:
            epoch:          Epoch index (used in the file name).
            train_steps:    Total optimiser steps taken so far.
            loss:           Most recent validation loss (informational).
            save_optimizer: When ``True``, embed the optimiser state dict for
                            exact training resumption.
        """
        if self.rank != 0:
            return

        checkpoint = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "train_steps": train_steps,
            "loss": loss,
        }
        if save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        checkpoint_path = f"{self.checkpoint_dir}/{epoch:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop over ``config.training.epochs`` epochs.

        Each epoch:
        1. Iterates the training loader, calling :func:`train_step` per batch.
        2. Logs aggregate loss statistics every ``config.logging.log_every`` steps.
        3. Evaluates the EMA model every ``config.logging.eval_every`` epochs.
        4. Saves a checkpoint every ``config.logging.ckpt_every`` epochs.

        Raises:
            Any exception from :func:`train_step` or :meth:`evaluate`; the
            ``finally`` block always calls :func:`cleanup` and finishes wandb.
        """
        self._setup_training_state()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.is_distributed:
            dist.barrier()

        if self.rank == 0:
            print(f"Starting training on {self.world_size} GPU(s)...")

        train_steps = 0
        log_steps = 0
        running_loss = running_noise_loss = running_img_loss = 0.0
        start_time = time()
        gradient_clip = self.args.gradient_clip

        self.logger.info(f"Training for {self.config.training.epochs} epochs...")
        try:
            for epoch in range(self.config.training.epochs):
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

                for batch in self.train_loader:
                    step_result = train_step(
                        self.model,
                        self.diffusion,
                        batch["image"].to(self.device, non_blocking=True),
                        self.optimizer,
                        self.ema,
                        self.device,
                        gradient_clip,
                        scaler=self.scaler,
                        trainable_params=self.trainable_params,
                        all_params=self.all_params,
                        amp_dtype=self.amp_dtype,
                    )

                    if step_result is None:
                        if self.rank == 0:
                            self.logger.warning("Skipping step due to invalid loss.")
                        continue

                    total_loss, noise_loss, img_loss = step_result
                    running_loss += total_loss
                    running_noise_loss += noise_loss
                    running_img_loss += img_loss
                    log_steps += 1
                    train_steps += 1

                    if train_steps % self.config.logging.log_every == 0 and log_steps > 0:
                        elapsed = time() - start_time
                        steps_per_sec = log_steps / max(elapsed, 1e-8)
                        stats = torch.tensor(
                            [running_loss, running_noise_loss, running_img_loss],
                            device=self.device,
                            dtype=torch.float32,
                        ) / log_steps

                        if dist.is_initialized() and dist.get_world_size() > 1:
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            stats /= dist.get_world_size()

                        avg_loss, avg_noise_loss, avg_img_loss = stats.tolist()

                        if self.rank == 0:
                            if wandb_enabled(self.config):
                                wandb.log({
                                    "train/total_loss": avg_loss,
                                    "train/noise_loss": avg_noise_loss,
                                    "train/img_loss": avg_img_loss,
                                    "train/epoch": epoch,
                                })
                            self.logger.info(
                                f"Epoch {epoch:4d} | Step {train_steps:7d} | "
                                f"Loss: {avg_loss:.6f} | Speed: {steps_per_sec:.2f} steps/s"
                            )

                        running_loss = running_noise_loss = running_img_loss = 0.0
                        log_steps = 0
                        start_time = time()

                if (epoch + 1) % self.config.logging.eval_every == 0:
                    metrics = self.evaluate(epoch)
                    if self.rank == 0 and metrics is not None:
                        if wandb_enabled(self.config):
                            wandb.log({
                                "eval/loss": metrics["eval_loss"],
                                "eval/avg_loss": metrics["eval_loss_avg"],
                                "epoch": metrics["epoch"],
                            })
                        self._last_eval_loss = metrics["eval_loss_avg"]
                        self.logger.info(
                            f"Evaluation at epoch {epoch}: "
                            f"eval_loss={self._last_eval_loss:.4f}"
                        )

                if (epoch + 1) % self.config.logging.ckpt_every == 0:
                    manage_checkpoints(self.checkpoint_dir, self.rank)
                    self.save_checkpoint(
                        epoch + 1,
                        train_steps,
                        self._last_eval_loss,
                        save_optimizer=((epoch + 1) == self.config.training.epochs),
                    )
                    if self.is_distributed:
                        dist.barrier()

            # Final checkpoint regardless of ckpt_every schedule.
            if self.rank == 0:
                self.save_checkpoint(
                    self.config.training.epochs,
                    train_steps,
                    self._last_eval_loss,
                    save_optimizer=True,
                )
            self.logger.info("Training completed successfully!")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            if self.rank == 0 and wandb_enabled(self.config):
                wandb.finish()
            cleanup()

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, epoch: int) -> Optional[Dict[str, float]]:
        """Evaluate the EMA model on the validation set.

        Only runs on rank 0; other ranks return ``None`` immediately to avoid
        duplicate work and I/O.

        Args:
            epoch: Current epoch index passed through to :func:`evaluate_model`.

        Returns:
            Metrics dict from :func:`evaluate_model`, or ``None`` on non-zero ranks.
        """
        if self.rank != 0:
            return None
        return evaluate_model(
            self.ema,
            self.val_loader,
            self.diffusion,
            self.device,
            self.rank,
            self.experiment_dir,
            self.config.data.image_size,
            epoch,
            self.logger,
        )


# ── Entry point ──────────────────────────────────────────────────────────────


def main(config: Config, debug: bool = False, from_scratch: bool = False) -> None:
    """Distributed training entry point.

    Initialises the NCCL process group, sets per-rank seeds, constructs the
    :class:`Trainer`, and starts training.

    Args:
        config:       Experiment configuration object.
        debug:        Forwarded to :class:`Trainer` for verbose diagnostics.
        from_scratch: When ``True``, overrides ``config.model.from_scratch``
                      to skip pretrained weight loading.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    setup_torch_config()

    dist.init_process_group("nccl")
    assert config.training.batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.training.seed * dist.get_world_size() + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    config.model.from_scratch = from_scratch

    Trainer(config, rank, device, seed, debug=debug).train()


def get_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config file name inside configs/local or configs/global.",
    )
    parser.add_argument(
        "--from_scratch", action="store_true",
        help="Override config and train from scratch.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose trainer diagnostics.",
    )
    return parser


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config_subdir = "local" if args.from_scratch else "global"
    config_path = os.path.join("configs", config_subdir, args.config)
    main(load_config(config_path), args.debug, args.from_scratch)
