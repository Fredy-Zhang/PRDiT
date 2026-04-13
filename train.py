"""Training pipeline for PRDiT diffusion models.

Responsibilities
----------------
- ``train_step``     — single forward-backward-optimiser step.
- ``evaluate_model`` — dispatch to reconstruction (depth=0) or generative (depth>0) evaluation.
- ``Trainer``        — stateful coordinator wiring all components together.
- ``main``           — distributed-training entry point called by ``torchrun``.

Stage 1 (local denoiser, depth=0):
    OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config lidc.yaml --from_scratch

Stage 2 (global residual DiT, depth>0):
    OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config lidc.yaml
"""

import argparse
import logging
import os
import random
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
    log_slices_to_wandb,
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


# =============================================================================
# Training step
# =============================================================================

# Log input data range once per training run.
_data_range_logged = True


def train_step(
    model: torch.nn.Module,
    diffusion: Any,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ema: torch.nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> Optional[Tuple[float, float, float]]:
    """Perform one forward–backward–optimiser step.

    Skips the step (returns ``None``) when the loss is non-finite or > 1e5.

    Returns
    -------
    tuple of float or None
        ``(total_loss, noise_loss, img_loss)`` as Python floats, or ``None``
        when the loss is non-finite or exceeds ``1e5``.
    """
    global _data_range_logged
    if _data_range_logged:
        logging.info(
            "Input data range — min: %.4f  max: %.4f  mean: %.4f  std: %.4f",
            x.min().item(), x.max().item(), x.mean().item(), x.std().item(),
        )
        _data_range_logged = False

    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

    optimizer.zero_grad(set_to_none=True)
    loss_dict = diffusion.training_losses(model, x, t)

    if isinstance(loss_dict, dict) and "noise_loss" in loss_dict and "img_loss" in loss_dict:
        noise_loss = loss_dict["noise_loss"].mean()
        img_loss   = loss_dict["img_loss"].mean()
        total_loss = noise_loss + img_loss
    else:
        total_loss = loss_dict["loss"].mean() if isinstance(loss_dict, dict) else loss_dict.mean()
        noise_loss = img_loss = total_loss

    total_loss_val = total_loss.detach().float().item()
    if not torch.isfinite(total_loss) or total_loss_val > 1e5:
        return None

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()

    model_module = model.module if hasattr(model, "module") else model
    update_ema(ema, model_module)
    return total_loss_val, noise_loss.detach().float().item(), img_loss.detach().float().item()


# =============================================================================
# Evaluation
# =============================================================================


def _evaluate_reconstruction(
    model: torch.nn.Module,
    val_loader: Any,
    diffusion: Any,
    device: torch.device,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Depth-0 evaluation: single forward-pass denoising with MSE loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model (or EMA shadow) to evaluate.
    val_loader : DataLoader
        Validation data loader.
    diffusion : IaNDiffusion
        Diffusion object providing ``q_sample``.
    device : torch.device
        Target CUDA device.
    experiment_dir : str
        Root directory for saving output samples.
    image_size : int
        Spatial dimension of the 3-D volume.
    epoch : int
        Current epoch index (used for file names and wandb logging).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    dict
        ``{"eval_loss_avg": float, "epoch": int}``
    """
    eval_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["image"].to(device, non_blocking=True)
            t = torch.full((x.shape[0],), 100, device=device)
            samples = diffusion.q_sample(x, t)

            model_output = model(samples, t)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            if model_output.shape[1] > x.shape[1]:
                _, img_recon = model_output.chunk(2, dim=1)
            else:
                img_recon = model_output

            eval_loss += (img_recon - x).pow(2).mean().item()
            num_batches += 1

            if i == 0:
                os.makedirs(f"{experiment_dir}/gt", exist_ok=True)
                save_evaluation_samples(img_recon, experiment_dir, image_size, epoch, 2, logger)
                save_evaluation_samples(x, f"{experiment_dir}/gt", image_size, epoch, 2, logger)
                log_slices_to_wandb("eval/reconstruction", img_recon, epoch)
                log_slices_to_wandb("eval/ground_truth",   x,         epoch)

    avg_loss = eval_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss (Average): {avg_loss:.6f}")
    return {"eval_loss_avg": avg_loss, "epoch": epoch}


def _evaluate_generative(
    model: torch.nn.Module,
    diffusion: Any,
    device: torch.device,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    num_samples: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Depth>0 evaluation: full reverse diffusion from noise.

    Saves ``xs`` (trajectory final) and ``x0`` (predicted clean image) samples
    to disk and uploads centre slices to wandb.

    Parameters
    ----------
    model : torch.nn.Module
        The model (or EMA shadow) to evaluate.
    diffusion : IaNDiffusion
        Diffusion object providing ``p_sample_loop``.
    device : torch.device
        Target CUDA device.
    experiment_dir : str
        Root directory for saving output samples.
    image_size : int
        Spatial dimension of the 3-D volume.
    epoch : int
        Current epoch index.
    num_samples : int
        Number of volumes to generate.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    dict
        ``{"eval_loss_avg": 0.0, "epoch": int}`` (no scalar metric for generative models).
    """
    xs_path = os.path.join(experiment_dir, "xs")
    x0_path = os.path.join(experiment_dir, "x0")
    os.makedirs(xs_path, exist_ok=True)
    os.makedirs(x0_path, exist_ok=True)

    z = torch.randn(num_samples, 1, image_size, image_size, image_size, device=device) * 0.5

    with torch.no_grad():
        xs_samples, x0_samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, new_sampling=True, model_kwargs={},
        )

    xs_final = xs_samples[-1]
    save_evaluation_samples(xs_final,   xs_path, image_size, epoch, 2, logger)
    save_evaluation_samples(x0_samples, x0_path, image_size, epoch, 2, logger)
    log_slices_to_wandb("eval/generated_xs", xs_final,   epoch)
    log_slices_to_wandb("eval/predicted_x0", x0_samples, epoch)

    logger.info(
        "Generated %d samples | xs [%.3f, %.3f] | x0 [%.3f, %.3f]",
        num_samples,
        xs_final.min().item(), xs_final.max().item(),
        x0_samples.min().item(), x0_samples.max().item(),
    )
    return {"eval_loss_avg": 0.0, "epoch": epoch}


def evaluate_model(
    model: torch.nn.Module,
    val_loader: Any,
    diffusion: Any,
    device: torch.device,
    rank: int,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    logger: logging.Logger,
    num_gen_samples: int = 4,
) -> Dict[str, float]:
    """Dispatch to the correct evaluation path based on model depth.

    Routes to :func:`_evaluate_reconstruction` (depth=0, MSE metric) or
    :func:`_evaluate_generative` (depth>0, no scalar metric).  No-ops on
    non-zero ranks and returns a zero-filled dict.

    Parameters
    ----------
    model : torch.nn.Module
        DDP-wrapped model (or plain module).
    val_loader : DataLoader
        Validation data loader (used only for depth-0).
    diffusion : IaNDiffusion
        Diffusion object.
    device : torch.device
        Target CUDA device.
    rank : int
        Process rank; evaluation runs only on rank 0.
    experiment_dir : str
        Root directory for saving output samples.
    image_size : int
        Spatial dimension of the 3-D volume.
    epoch : int
        Current epoch index.
    logger : logging.Logger
        Logger instance.
    num_gen_samples : int, optional
        Number of volumes to generate for depth>0 evaluation (default ``4``).

    Returns
    -------
    dict
        ``{"eval_loss_avg": float, "epoch": int}``
    """
    if rank != 0:
        return {"eval_loss_avg": 0.0, "epoch": epoch}

    inner = model.module if hasattr(model, "module") else model
    is_depth_zero = hasattr(inner, "depth") and inner.depth == 0

    model.eval()
    try:
        if is_depth_zero:
            return _evaluate_reconstruction(
                model, val_loader, diffusion, device,
                experiment_dir, image_size, epoch, logger,
            )
        else:
            return _evaluate_generative(
                model, diffusion, device,
                experiment_dir, image_size, epoch, num_gen_samples, logger,
            )
    finally:
        model.train()


# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    """Stateful coordinator for distributed PRDiT training.

    Parameters
    ----------
    config : Config
        Experiment configuration.
    rank : int
        Process rank in the distributed group.
    device : int
        CUDA device index for this process.
    seed : int
        Per-rank random seed.
    debug : bool, optional
        Emit extra diagnostics when ``True`` (default ``False``).
    """

    def __init__(
        self,
        config: Config,
        rank: int,
        device: int,
        seed: int,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.rank = rank
        self.device = device
        self.debug = debug
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1

        # Best eval-loss tracker — used only in depth-0 (local denoiser) runs.
        self._best_eval_loss: float = float("inf")

        # Args facade caches config-derived scalars (e.g. gradient_clip).
        self.args = Args(config)

        if rank == 0:
            self.experiment_dir, self.checkpoint_dir = create_experiment_dirs(self.args)
        else:
            self.experiment_dir = self.checkpoint_dir = None

        self.logger = create_logger(None if rank != 0 else self.experiment_dir)

        self._setup_wandb()
        self._setup_model()
        self._setup_data(seed)

    # -- Setup ----------------------------------------------------------------

    def _setup_wandb(self) -> None:
        if self.rank == 0:
            setup_wandb(self.config, self.rank)

    def _setup_model(self) -> None:
        """Build model, EMA, DDP wrapper, diffusion, and optimiser."""
        self.model = load_model(self.config)
        if hasattr(self.model, "log_config"):
            self.model.log_config(self.rank)

        # On depth-0 runs the fine_head is unused — freeze it to keep param
        # counts honest and avoid spurious gradient flow.
        is_depth_zero = hasattr(self.model, "depth") and self.model.depth == 0
        if (
            is_depth_zero
            and hasattr(self.model, "t_embedder")
            and hasattr(self.model.t_embedder, "fine_head")
        ):
            self.model.t_embedder.fine_head.requires_grad_(False)

        self.model = self.model.to(self.device)
        self.ema = deepcopy(self.model).to(self.device)
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
            self.logger.info("=== Parameter state before weight loading ===")
            log_params(self.model, self.logger)

        # Load pretrained weights for stage-2; skip for stage-1 (from_scratch).
        pretrained_flag = False
        new_param_names = None
        if not self.config.model.from_scratch:
            if self.rank == 0:
                self.logger.info(
                    f"Loading pretrained weights from: {self.config.model.pretrained_path}"
                )
            self.model, pretrained_flag, new_param_names = initialize_model_with_pretrained(
                model=self.model,
                config=self.config,
                device=f"cuda:{self.rank}",
                rank=self.rank,
                gain=0.1,
            )

        # Optional explicit checkpoint override (e.g. for fine-tuning).
        if getattr(self.config.model, "checkpoint", None):
            ckpt = torch.load(self.config.model.checkpoint, map_location=f"cuda:{self.rank}")
            self.model.module.load_state_dict(ckpt["model"])

        self.optimizer = initialize_optimizer(
            self.model,
            self.config,
            pretrained_flag=pretrained_flag,
            new_param_names=new_param_names,
            rank=self.rank,
            debug=self.debug,
        )

        if self.rank == 0 and self.debug:
            print_optimizer_params(
                self.optimizer, self.model,
                self.config.training.learning_rate,
                self.config.training.fine_tune_lr,
                self.logger,
            )
            self.logger.info("=== Parameter state after weight loading ===")
            log_params(self.model, self.logger)

    def _setup_data(self, seed: int) -> None:
        """Create train and validation data loaders."""
        self.train_loader, self.val_loader = return_train_val_loaders(
            self.args, self.rank, self.config, seed, self.debug
        )
        if self.rank == 0:
            self.logger.info(
                f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )
            self.logger.info(
                f"Dataset: {len(self.train_loader.dataset):,} images ({self.config.data.path})"
            )

    def _setup_training_state(self) -> None:
        """Sync EMA to model weights, set train/eval modes, configure matmul."""
        update_ema(self.ema, self.model.module, decay=0)
        self.model.train()
        self.ema.eval()

        is_depth_zero = hasattr(self.model.module, "depth") and self.model.module.depth == 0
        if self.config.model.flash_attn and not is_depth_zero:
            torch.set_float32_matmul_precision("high")
            if self.rank == 0:
                self.logger.info("Flash attention enabled: matmul precision set to 'high'.")

    # -- Checkpoint -----------------------------------------------------------

    def save_checkpoint(self, epoch, train_steps: int, save_optimizer: bool = False) -> None:
        """Save a checkpoint to ``checkpoint_dir``.

        Parameters
        ----------
        epoch : int or str
            Epoch number, or ``"best"`` to write ``best.pt`` during depth-0
            training.
        train_steps : int
            Total optimiser steps completed so far.
        save_optimizer : bool, optional
            Include optimiser state dict when ``True`` (default ``False``).
        """
        if self.rank != 0:
            return

        checkpoint = {
            "model":       self.model.module.state_dict(),
            "ema":         self.ema.state_dict(),
            "config":      self.config.to_dict(),
            "epoch":       epoch,
            "train_steps": train_steps,
        }
        if save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        filename = "best.pt" if epoch == "best" else f"{epoch:06d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        suffix = " (with optimizer)" if save_optimizer else ""
        self.logger.info(f"Saved checkpoint: {path}{suffix}")

    # -- Training loop --------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        self._setup_training_state()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.is_distributed:
            dist.barrier()

        train_steps = 0
        log_steps   = 0
        running_loss = running_noise_loss = running_img_loss = 0.0
        start_time   = time()
        gradient_clip = self.args.gradient_clip

        if self.rank == 0:
            self.logger.info(
                f"Training on {self.world_size} GPU(s) for {self.config.training.epochs} epochs."
            )

        try:
            for epoch in range(self.config.training.epochs):
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

                for batch in self.train_loader:
                    result = train_step(
                        self.model,
                        self.diffusion,
                        batch["image"].to(self.device, non_blocking=True),
                        self.optimizer,
                        self.ema,
                        self.device,
                        gradient_clip,
                    )

                    if result is None:
                        if self.rank == 0:
                            self.logger.warning("Skipping step: invalid loss.")
                        continue

                    total_loss, noise_loss, img_loss = result
                    running_loss       += total_loss
                    running_noise_loss += noise_loss
                    running_img_loss   += img_loss
                    log_steps   += 1
                    train_steps += 1

                    if train_steps % self.config.logging.log_every == 0 and log_steps > 0:
                        elapsed       = time() - start_time
                        steps_per_sec = log_steps / max(elapsed, 1e-8)

                        stats = torch.tensor(
                            [running_loss, running_noise_loss, running_img_loss],
                            device=self.device, dtype=torch.float32,
                        ) / log_steps
                        if dist.is_initialized() and self.world_size > 1:
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            stats /= self.world_size

                        avg_loss, avg_noise, avg_img = stats.tolist()

                        if self.rank == 0:
                            if wandb_enabled(self.config):
                                wandb.log({
                                    "train/total_loss": avg_loss,
                                    "train/noise_loss": avg_noise,
                                    "train/img_loss":   avg_img,
                                    "train/epoch":      epoch,
                                })
                            self.logger.info(
                                f"Epoch {epoch:4d} | Step {train_steps:7d} | "
                                f"Loss: {avg_loss:.6f} "
                                f"(noise: {avg_noise:.6f}, img: {avg_img:.6f}) | "
                                f"Speed: {steps_per_sec:.2f} steps/s"
                            )

                        running_loss = running_noise_loss = running_img_loss = 0.0
                        log_steps  = 0
                        start_time = time()

                # -- End-of-epoch hooks ---------------------------------------

                if (epoch + 1) % self.config.logging.eval_every == 0:
                    eval_loss = self._evaluate(epoch)
                    # depth-0 only: persist best checkpoint by val MSE.
                    if eval_loss is not None and eval_loss < self._best_eval_loss:
                        self._best_eval_loss = eval_loss
                        if self.rank == 0:
                            self.save_checkpoint("best", train_steps)
                            self.logger.info(
                                f"New best eval loss: {eval_loss:.6f} — saved best.pt"
                            )

                if (epoch + 1) % self.config.logging.ckpt_every == 0:
                    manage_checkpoints(self.checkpoint_dir, self.rank)
                    # Include optimiser state every 5000 epochs for exact resumption.
                    save_opt = (
                        (epoch + 1) % 5000 == 0
                        or (epoch + 1) == self.config.training.epochs
                    )
                    self.save_checkpoint(epoch + 1, train_steps, save_opt)
                    if self.is_distributed:
                        dist.barrier()

            # Final checkpoint regardless of ckpt_every schedule.
            if self.rank == 0:
                self.save_checkpoint(self.config.training.epochs, train_steps, save_optimizer=True)
            self.logger.info("Training completed successfully!")

        except Exception as exc:
            self.logger.error(f"Training failed: {exc}")
            raise
        finally:
            if self.rank == 0 and wandb_enabled(self.config):
                wandb.finish()
            cleanup()

    # -- Evaluation -----------------------------------------------------------

    def _evaluate(self, epoch: int) -> Optional[float]:
        """Run evaluation and return the average validation loss (depth-0 only).

        Returns
        -------
        float or None
            Average validation MSE for depth-0 models; ``None`` for depth > 0.
        """
        if self.rank != 0:
            return None

        metrics = evaluate_model(
            self.ema,
            self.val_loader,
            self.diffusion,
            self.device,
            self.rank,
            self.experiment_dir,
            self.config.data.image_size,
            epoch,
            self.logger,
            num_gen_samples=getattr(self.config.training, "num_gen_samples", 4),
        )

        inner = self.ema.module if hasattr(self.ema, "module") else self.ema
        if hasattr(inner, "depth") and inner.depth == 0:
            return metrics.get("eval_loss_avg")
        return None


# =============================================================================
# Entry point
# =============================================================================


def main(config: Config, debug: bool = False, from_scratch: bool = False) -> None:
    """Distributed training entry point (called by ``torchrun``).

    Parameters
    ----------
    config : Config
        Populated experiment configuration.
    debug : bool, optional
        Enable verbose diagnostics (default ``False``).
    from_scratch : bool, optional
        Train stage 1 from scratch without pretrained weights (default ``False``).
    """
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    setup_torch_config()
    dist.init_process_group("nccl")

    assert config.training.batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank   = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed   = config.training.seed * dist.get_world_size() + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    config.model.from_scratch = from_scratch
    Trainer(config, rank, device, seed, debug=debug).train()


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config filename inside configs/local/ or configs/global/.")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train stage 1 from scratch (no pretrained weights).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose diagnostics.")
    return parser


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config_subdir = "local" if args.from_scratch else "global"
    config_path = os.path.join("configs", config_subdir, args.config)
    main(load_config(config_path), args.debug, args.from_scratch)
