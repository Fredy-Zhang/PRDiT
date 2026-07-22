"""Stage-3 training pipeline: super-resolution refinement on top of a trained PRDiT.

Reuses `train.train_step` (diffusion-object-agnostic) and
`train._evaluate_reconstruction` (generic reconstruction eval) unchanged —
`PRDiTSR.forward` matches `PRDiT.forward`'s output contract, so both drop in
with no modification. Only model construction/loading and the optimiser's
differential learning rates (SR head vs. base model) are stage-3-specific,
so they live in `SRTrainer` here rather than in `train.Trainer`, which is
built around the depth==0/depth>0 (stage-1/stage-2) distinction.

Stage 3a (frozen base — train the SR head from scratch):
    OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train_sr.py --config lidc.yaml

Stage 3b (joint fine-tune — unfreeze the base at a very low LR):
    Set `sr_model.resume_checkpoint` in configs/sr/lidc_finetune.yaml to stage
    3a's best checkpoint, then:
    OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train_sr.py --config lidc_finetune.yaml
"""

import argparse
import glob
import os
import random
from copy import deepcopy
from time import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb

from diffusion import loading_diffusion
from models import load_sr_model
from train import _evaluate_reconstruction, train_step
from util import (
    Args,
    Config,
    cleanup,
    create_experiment_dirs,
    create_logger,
    load_config,
    manage_checkpoints,
    requires_grad,
    return_train_val_loaders,
    setup_torch_config,
    setup_wandb,
    update_ema,
    wandb_enabled,
)


class SRTrainer:
    """Stateful coordinator for distributed stage-3 (super-resolution) training."""

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
        self._best_eval_loss = float("inf")

        self.args = Args(config)

        if rank == 0:
            self.experiment_dir, self.checkpoint_dir = create_experiment_dirs(self.args)
        else:
            self.experiment_dir = self.checkpoint_dir = None

        self.logger = create_logger(None if rank != 0 else self.experiment_dir)

        if self.rank == 0:
            setup_wandb(self.config, self.rank)

        self._setup_model()
        self._setup_data(seed)

    # -- Setup ----------------------------------------------------------------

    def _setup_model(self) -> None:
        """Build the PRDiTSR model, EMA, DDP wrapper, diffusion, and optimiser."""
        self.model = load_sr_model(self.config)

        resume_path = getattr(self.config.sr_model, "resume_checkpoint", None)
        if resume_path:
            checkpoint = torch.load(resume_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=True)
            if self.rank == 0:
                self.logger.info(f"Resumed stage-3 model from {resume_path}")

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

        # Differential LR: SR head at the full LR, base model (only trainable
        # when `base_model.finetune: true`) at `fine_tune_lr`.
        sr_params = [
            p for n, p in self.model.module.named_parameters()
            if not n.startswith("base.") and p.requires_grad
        ]
        base_params = [
            p for n, p in self.model.module.named_parameters()
            if n.startswith("base.") and p.requires_grad
        ]

        param_groups = [{
            "params": sr_params,
            "lr": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
        }]
        if base_params:
            param_groups.append({
                "params": base_params,
                "lr": self.config.training.fine_tune_lr,
                "weight_decay": self.config.training.weight_decay,
            })
        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.logger.info(
                f"Stage-3 model: {trainable:,} / {total:,} trainable parameters "
                f"(base_model.finetune={bool(getattr(self.config.base_model, 'finetune', False))})"
            )

    def _setup_data(self, seed: int) -> None:
        """Create train and validation data loaders at the SR target resolution."""
        self.train_loader, self.val_loader = return_train_val_loaders(
            self.args, self.rank, self.config, seed, self.debug
        )
        if self.rank == 0:
            self.logger.info(
                f"Dataset: {len(self.train_loader.dataset):,} volumes ({self.config.data.path})"
            )

    # -- Checkpoint -------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch,
        train_steps: int,
        save_optimizer: bool = False,
        best_loss: Optional[float] = None,
    ) -> None:
        """Save a checkpoint to `checkpoint_dir` (mirrors `train.Trainer.save_checkpoint`)."""
        if self.rank != 0:
            return

        checkpoint = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "train_steps": train_steps,
        }
        if save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        if epoch == "best":
            for old in glob.glob(os.path.join(self.checkpoint_dir, "best_*.pt")):
                os.remove(old)
            loss_tag = f"_{best_loss:.6f}" if best_loss is not None else ""
            filename = f"best{loss_tag}.pt"
        else:
            filename = f"{epoch:06d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    # -- Evaluation ---------------------------------------------------------------

    def _evaluate(self, epoch: int) -> Optional[float]:
        """Reconstruction-style eval against the real high-res ground truth."""
        if self.rank != 0:
            return None
        self.ema.eval()
        try:
            metrics = _evaluate_reconstruction(
                self.ema,
                self.val_loader,
                self.diffusion,
                self.device,
                self.experiment_dir,
                self.config.data.image_size,
                epoch,
                self.logger,
            )
        finally:
            self.ema.train()
        return metrics.get("eval_loss_avg")

    # -- Training loop --------------------------------------------------------

    def train(self) -> None:
        """Run the full stage-3 training loop."""
        update_ema(self.ema, self.model.module, decay=0)
        self.model.train()
        self.ema.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.is_distributed:
            dist.barrier()

        total_steps_target = self.config.training.total_steps
        if self.rank == 0:
            self.logger.info(
                f"Training stage-3 on {self.world_size} GPU(s) for {total_steps_target:,} total steps."
            )

        train_steps = 0
        log_steps = 0
        running_loss = running_noise_loss = running_img_loss = 0.0
        start_time = time()
        gradient_clip = self.args.gradient_clip
        epoch = 0

        try:
            while train_steps < total_steps_target:
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

                for batch in self.train_loader:
                    if train_steps >= total_steps_target:
                        break

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
                                    "train/img_loss": avg_img,
                                    "train/epoch": epoch,
                                    "train/step": train_steps,
                                })
                            self.logger.info(
                                f"Epoch {epoch:4d} | Step {train_steps:7d}/{total_steps_target:,} | "
                                f"Loss: {avg_loss:.6f} "
                                f"(noise: {avg_noise:.6f}, img: {avg_img:.6f}) | "
                                f"Speed: {steps_per_sec:.2f} steps/s"
                            )

                        running_loss = running_noise_loss = running_img_loss = 0.0
                        log_steps = 0
                        start_time = time()

                    if train_steps % self.config.logging.eval_every == 0:
                        eval_loss = self._evaluate(epoch)
                        if eval_loss is not None and eval_loss < self._best_eval_loss:
                            self._best_eval_loss = eval_loss
                            if self.rank == 0:
                                self.save_checkpoint("best", train_steps, best_loss=eval_loss)
                                self.logger.info(f"New best eval loss: {eval_loss:.6f}")

                    if train_steps % self.config.logging.ckpt_every == 0:
                        manage_checkpoints(self.checkpoint_dir, self.rank)
                        save_opt = (
                            train_steps % 5000 == 0
                            or train_steps >= total_steps_target
                        )
                        self.save_checkpoint(train_steps, train_steps, save_opt)
                        if self.is_distributed:
                            dist.barrier()

                epoch += 1

            if self.rank == 0:
                self.save_checkpoint(train_steps, train_steps, save_optimizer=True)
            self.logger.info("Stage-3 training completed successfully!")

        except Exception as exc:
            self.logger.error(f"Training failed: {exc}")
            raise
        finally:
            if self.rank == 0 and wandb_enabled(self.config):
                wandb.finish()
            cleanup()


# =============================================================================
# Entry point
# =============================================================================


def main(config: Config, debug: bool = False) -> None:
    """Distributed stage-3 training entry point (called by `torchrun`)."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."

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

    SRTrainer(config, rank, device, seed, debug=debug).train()


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config filename inside configs/sr/.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose diagnostics.")
    return parser


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    config_path = os.path.join("configs", "sr", args.config)
    main(load_config(config_path), args.debug)
