"""Inference script for sampling from a trained stage-3 PRDiTSR model.

Mirrors `sample.py` almost exactly: `PRDiTSR.forward` matches `PRDiT.forward`'s
output contract, so the same `IaNDiffusion.p_sample_loop` reverse-diffusion
sampler works unmodified, just at the SR target resolution
(`config.data.image_size`, e.g. 256) instead of the base resolution.

Usage
-----
Single-GPU::

    python sample_sr.py --config lidc.yaml \\
        --ckpt results_sr/000-PRDiT-SR-.../checkpoints/best_0.001800.pt \\
        --total-samples 50 --num-samples 4 --num-sampling-steps 1000
"""

import argparse
import math
import os
import time

import torch

from diffusion.ian_diffusion import IaNDiffusion
from utils.download import find_model
from models import load_sr_model
from util import load_config, save_evaluation_samples


def sample(args: argparse.Namespace) -> None:
    """Run the full stage-3 sampling loop for a single configuration."""
    config = load_config(os.path.join("configs", "sr", args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.training.seed)
    torch.set_grad_enabled(False)

    xs_path = os.path.join(args.output_dir, "xs")
    x0_path = os.path.join(args.output_dir, "x0")
    os.makedirs(xs_path, exist_ok=True)
    os.makedirs(x0_path, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    # `load_sr_model` also loads `config.base_model.checkpoint` into the base
    # PRDiT; the checkpoint below then overwrites the full PRDiTSR state
    # (base + SR head) with the trained stage-3 weights.
    model = load_sr_model(config).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded stage-3 model from {args.ckpt}")

    diffusion = IaNDiffusion(
        timestep_respacing=str(args.num_sampling_steps),
        loss_type="l2",
    )

    # ── Sampling loop ─────────────────────────────────────────────────────────
    num_batches = math.ceil(args.total_samples / args.num_samples)
    batch_times: list[float] = []
    total_sampling_time = 0.0

    for batch_idx in range(num_batches):
        current_batch_size = min(
            args.num_samples,
            args.total_samples - batch_idx * args.num_samples,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        z = torch.randn(
            current_batch_size,
            config.base_model.in_channels,
            config.data.image_size,
            config.data.image_size,
            config.data.image_size,
            device=device,
        ) * 0.5

        xs_samples, x0_samples = diffusion.p_sample_loop(
            model.forward,
            z.shape,
            z,
            new_sampling=args.new,
            model_kwargs={},
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        batch_times.append(elapsed)
        total_sampling_time += elapsed

        xs_final = xs_samples[-1]
        start_idx = batch_idx * args.num_samples
        save_evaluation_samples(xs_final, xs_path, config.data.image_size, epoch=start_idx, logger=None)
        save_evaluation_samples(x0_samples, x0_path, config.data.image_size, epoch=start_idx, logger=None)

        print(
            f"[{batch_idx + 1}/{num_batches}] "
            f"samples {start_idx + 1}-{start_idx + current_batch_size} | "
            f"XS [{xs_final.min():.3f}, {xs_final.max():.3f}] std={xs_final.std():.3f} | "
            f"X0 [{x0_samples.min():.3f}, {x0_samples.max():.3f}] std={x0_samples.std():.3f} | "
            f"{elapsed:.2f}s"
        )

    avg_batch = total_sampling_time / len(batch_times)
    avg_sample = total_sampling_time / args.total_samples
    print(
        f"\nDone. {args.total_samples} samples saved to {args.output_dir}\n"
        f"  Total time   : {total_sampling_time:.2f}s\n"
        f"  Per batch    : {avg_batch:.2f}s\n"
        f"  Per sample   : {avg_sample:.2f}s\n"
        f"  Batch times  : {[f'{t:.2f}' for t in batch_times]}"
    )


def get_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Sample high-resolution 3-D CT volumes from a trained PRDiTSR model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Config filename inside configs/sr/ (e.g. lidc.yaml).")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the stage-3 model checkpoint (.pt file).")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of volumes to generate per batch (default: 4).")
    parser.add_argument("--total-samples", type=int, default=1000,
                        help="Total number of volumes to generate (default: 1000).")
    parser.add_argument("--num-sampling-steps", type=int, default=1000,
                        help="Number of DDPM reverse diffusion steps (default: 1000).")
    parser.add_argument("--output-dir", type=str, default="samples_sr",
                        help="Directory in which xs/ and x0/ sub-folders are created (default: samples_sr).")
    parser.add_argument("--new", action="store_true",
                        help="Use the new p_sample_loop sampling schema.")
    return parser


if __name__ == "__main__":
    sample(get_argument_parser().parse_args())
