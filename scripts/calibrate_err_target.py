"""Calibrate ``--err-target`` for the ``pc_heun`` flow-matching solver.

The ``pc_heun`` noise gate is ``eta_base * min(err_norm / err_target, 1) * (1-t)``
where ``err_norm`` is the dt^2-normalised Euler-vs-Heun local error. To make the
gate actually open, ``err_target`` must sit near the model's typical
``err_norm``. This probe loads the checkpoint exactly like ``sample.py`` and
reports the ``err_norm`` distribution over a few real ODE trajectories.

Pick ``ERR_TARGET`` from the printed median:
  * ``ERR_TARGET = median``      -> eta ~= eta_base at typical curvature
  * ``ERR_TARGET = median / 2``  -> gate opens more eagerly (more noise)

Run on a GPU compute node (not the login node)::

    python scripts/calibrate_err_target.py \\
        --config lidc.yaml \\
        --ckpt {MODEL_CHECKPOINT} \\
        --num-samples 4 --steps 10 20 40 50 80 100
"""

import argparse
import os
import statistics as stats
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.flow_matching import FlowMatching
from models import load_model
from util import load_config
from utils.download import find_model


@torch.no_grad()
def err_norm_trajectory(model, fm, z, num_steps, model_kwargs):
    """Return the per-step dt^2-normalised local-error values along one Heun run."""
    x = z
    dt = 1.0 / num_steps
    dims = tuple(range(1, x.ndim))

    def rms(t):
        return t.pow(2).mean(dim=dims, keepdim=True).sqrt()

    vals = []
    for step in range(num_steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        v = model(x, fm._model_t(t), **model_kwargs)
        x_euler = x + dt * v
        t1 = torch.full((x.shape[0],), min((step + 1) * dt, 1.0), device=x.device, dtype=x.dtype)
        v1 = model(x_euler, fm._model_t(t1), **model_kwargs)
        x_pc = x + 0.5 * dt * (v + v1)
        err = rms(x_pc - x_euler) / (rms(x_pc) + 1e-6)
        vals.append((err / (dt * dt)).mean().item())
        x = x_pc
    return vals


def main():
    p = argparse.ArgumentParser(description="Calibrate err_target for pc_heun.")
    p.add_argument("--config", type=str, required=True, help="Config filename under configs/global/.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to the .pt checkpoint.")
    p.add_argument("--num-samples", type=int, default=4, help="Volumes per probe trajectory.")
    p.add_argument("--steps", type=int, nargs="+", default=[10, 20, 40, 50, 80, 100],
                   help="Step counts to probe (err_norm should be ~flat across these).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(os.path.join("configs", "global", args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model = load_model(config).to(device)
    model.load_state_dict(find_model(args.ckpt))
    model.eval()
    print(f"Loaded model from {args.ckpt} on {device}")

    fm = FlowMatching()
    model_kwargs = {"y": None} if config.model.num_classes else {}

    z = torch.randn(
        args.num_samples,
        config.model.in_channels,
        config.data.image_size,
        config.data.image_size,
        config.data.image_size,
        device=device,
    )

    print(f"\n{'steps':>6} {'median':>10} {'min':>10} {'max':>10}   (err_norm = err / dt^2)")
    all_medians = []
    for n in args.steps:
        vals = err_norm_trajectory(model, fm, z.clone(), n, model_kwargs)
        med = stats.median(vals)
        all_medians.append(med)
        print(f"{n:>6} {med:>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")

    overall = stats.median(all_medians)
    print("\n" + "=" * 60)
    print(f"Suggested ERR_TARGET (median across step counts): {overall:.4f}")
    print(f"  eager gate (more noise): ERR_TARGET={overall / 2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
