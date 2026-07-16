"""3D FID sweep over the flow-matching solver x sampling-step ablation.

Computes a 3D FID score for every ``(solver, num_steps)`` cell produced by
``scripts/solver_sweep_fm.sh`` and writes a single tidy CSV so the solvers can
be compared at matched NFE.

The expensive pieces are done **once** and shared across all cells:

* the 3D ResNet-50 feature extractor is loaded a single time, and
* the real-data reference statistics (``mu_real`` / ``sigma_real``) are
  computed once and cached to ``--path_to_activations`` (or reloaded from there
  with ``--reuse_real_stats``).

Only the per-cell fake activations are recomputed, so the whole grid costs one
real preload plus one fake pass per cell (versus reloading everything 15x when
shelling out to ``fid.py`` per directory).

Directory layout expected (matches ``scripts/solver_sweep_fm.sh``)::

    <sweep_root>/<solver>_steps_<N>/<fake_subdir>/   # generated volumes

Usage::

    python evaluations/eval_fid_sweep.py \\
        --sweep_root samples/solver_sweep \\
        --dataset lidc-idri --img_size 128 \\
        --data_root_real /path/to/LIDC-IDRI \\
        --split_dir datasets/splits/lidc \\
        --pretrain_path evaluations/pretrained/resnet_50_23dataset.pth \\
        --path_to_activations samples/solver_sweep/fid_acts \\
        --num_samples 10

Output::

    <sweep_root>/fid_summary.csv   # solver,steps,nfe,num_fake,fid
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Mirror fid.py's import setup so `model` (evaluations/model.py) and the
# top-level `datasets` package both resolve regardless of the launch cwd.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for _p in (current_dir, parent_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.lidc import LIDCVolumes
from datasets.rad_chest import RADChestCTDataset
from fid import (
    calculate_frechet_distance,
    get_activations,
    get_feature_extractor,
    process_feature_vecs,
    set_randomness,
)

# Function evaluations per integration step for each solver. Used only for the
# reported ``nfe`` column so cells can be compared at matched compute.
NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4, "pc": 1, "pc_heun": 2}


def build_dataset(sets, root, mode):
    """Construct a real/fake dataset, mirroring ``fid.py``'s ``_build_dataset``."""
    if sets.dataset == "rad_chestCT":
        kw = {"split_dir": sets.split_dir} if mode == "real" else {}
        return RADChestCTDataset(root, img_size=sets.img_size, mode=mode, **kw)
    if sets.dataset == "lidc-idri":
        kw = {"split_dir": sets.split_dir} if mode == "real" else {}
        return LIDCVolumes(root, img_size=sets.img_size, mode=mode, **kw)
    raise ValueError(f"Unsupported dataset: {sets.dataset}. Use 'rad_chestCT' or 'lidc-idri'.")


def activations_for(model, dataset, sets, device, requested):
    """Extract features for up to ``requested`` samples from ``dataset``.

    ``get_activations`` pre-allocates ``sets.num_samples`` rows, so we clamp it
    to the dataset size for this call and restore it afterwards.
    """
    n = min(requested, len(dataset))
    loader = DataLoader(
        dataset, batch_size=sets.batch_size, shuffle=False,
        num_workers=sets.num_workers, pin_memory=False,
    )
    saved = sets.num_samples
    sets.num_samples = n
    try:
        activs = get_activations(model, loader, sets, device)
    finally:
        sets.num_samples = saved
    return activs


def get_real_stats(model, sets, device):
    """Return ``(mu_real, sigma_real)``, using cached stats when requested."""
    os.makedirs(sets.path_to_activations, exist_ok=True)
    mu_path = os.path.join(sets.path_to_activations, "mu_real.npy")
    sigma_path = os.path.join(sets.path_to_activations, "sigma_real.npy")

    if sets.reuse_real_stats and os.path.exists(mu_path) and os.path.exists(sigma_path):
        print(f"Reusing cached real stats from {sets.path_to_activations}.")
        return np.load(mu_path), np.load(sigma_path)

    if not sets.data_root_real:
        raise ValueError(
            "--data_root_real is required to compute real statistics "
            "(no cached mu_real.npy/sigma_real.npy found for --reuse_real_stats)."
        )
    real_data = build_dataset(sets, sets.data_root_real, "real")
    print(f"Real reference: {len(real_data)} volumes available.")
    activs = activations_for(model, real_data, sets, device, sets.num_samples)
    mu_real, sigma_real = process_feature_vecs(activs)
    np.save(mu_path, mu_real)
    np.save(sigma_path, sigma_real)
    print(f"Real activations: {activs.shape} (cached to {sets.path_to_activations})")
    return mu_real, sigma_real


def parse_opts():
    parser = argparse.ArgumentParser(description="3D FID sweep over the solver x step ablation.")
    # Sweep grid / layout.
    parser.add_argument("--sweep_root", default="samples/solver_sweep", type=str,
                        help="Root produced by scripts/solver_sweep_fm.sh.")
    parser.add_argument("--solvers", nargs="+", default=["euler", "heun", "rk4"],
                        help="Solver names to evaluate (default: euler heun rk4).")
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 30, 40, 50],
                        help="Sampling-step counts to evaluate (default: 10 20 30 40 50).")
    parser.add_argument("--fake_subdir", default="xs", type=str,
                        help="Sub-directory of each run holding the generated volumes (default: xs).")
    parser.add_argument("--output_csv", default=None, type=str,
                        help="CSV output path (default: <sweep_root>/fid_summary.csv).")
    # Real reference / feature extractor (shared across the grid).
    parser.add_argument("--dataset", default="lidc-idri", type=str, help="rad_chestCT | lidc-idri")
    parser.add_argument("--img_size", default=128, type=int)
    parser.add_argument("--data_root_real", default=None, type=str,
                        help="Path to real data (not needed when --reuse_real_stats hits the cache).")
    parser.add_argument("--split_dir", default=None, type=str,
                        help="Directory containing train.txt/val.txt for the real split.")
    parser.add_argument("--pretrain_path",
                        default=os.path.join(current_dir, "pretrained", "resnet_50_23dataset.pth"),
                        type=str, help="Path to pretrained 3D ResNet-50 weights.")
    parser.add_argument("--path_to_activations", default=None, type=str,
                        help="Directory for cached real stats (default: <sweep_root>/fid_acts).")
    parser.add_argument("--reuse_real_stats", action="store_true",
                        help="Reuse cached mu_real.npy/sigma_real.npy if present.")
    parser.add_argument("--num_samples", default=1000, type=int,
                        help="Max volumes per side for FID (clamped to what each dir holds).")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--manual_seed", default=192, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", default=0, type=int)
    # ResNet architecture args (defaults match the pretrained weights).
    parser.add_argument("--model", default="resnet", type=str)
    parser.add_argument("--model_depth", default=50, type=int)
    parser.add_argument("--resnet_shortcut", default="B", type=str)
    parser.add_argument("--input_D", default=128, type=int)
    parser.add_argument("--input_H", default=128, type=int)
    parser.add_argument("--input_W", default=128, type=int)
    parser.add_argument("--n_seg_classes", default=2, type=int)
    parser.add_argument("--new_layer_names", default=["conv_seg"], type=list)
    parser.add_argument("--phase", default="test", type=str)
    return parser.parse_args()


def main():
    sets = parse_opts()
    sets.target_type = "normal"
    sets.dims = 2048
    if sets.path_to_activations is None:
        sets.path_to_activations = os.path.join(sets.sweep_root, "fid_acts")
    if sets.output_csv is None:
        sets.output_csv = os.path.join(sets.sweep_root, "fid_summary.csv")

    device = torch.device("cpu" if sets.no_cuda else f"cuda:{sets.gpu_id}")
    set_randomness(sets.manual_seed)

    print("Loading feature extractor...")
    model = get_feature_extractor(sets).to(device)

    print("Computing / loading real reference statistics...")
    mu_real, sigma_real = get_real_stats(model, sets, device)

    if sets.num_samples < sets.dims:
        print(
            f"\n[WARNING] num_samples={sets.num_samples} < feature dim {sets.dims}. "
            "The covariance is rank-deficient, so absolute FID values are unstable "
            "and mainly useful as a relative ranking within this sweep.\n"
        )

    rows = []
    for solver in sets.solvers:
        nfe_per_step = NFE_PER_STEP.get(solver)
        if nfe_per_step is None:
            print(f"[skip] unknown solver '{solver}' (expected one of {list(NFE_PER_STEP)}).")
            continue
        for steps in sets.steps:
            run_dir = os.path.join(sets.sweep_root, f"{solver}_steps_{steps}")
            fake_dir = os.path.join(run_dir, sets.fake_subdir)
            nfe = nfe_per_step * steps

            if not os.path.isdir(fake_dir):
                print(f"[skip] missing fake dir: {fake_dir}")
                rows.append({"solver": solver, "steps": steps, "nfe": nfe,
                             "num_fake": 0, "fid": ""})
                continue

            print(f"\n=== solver={solver} steps={steps} NFE={nfe} ===")
            fake_data = build_dataset(sets, fake_dir, "fake")
            activs = activations_for(model, fake_data, sets, device, sets.num_samples)
            mu_fake, sigma_fake = process_feature_vecs(activs)
            fid = float(calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))
            print(f"3D FID: {fid:.4f}  (fake volumes used: {activs.shape[0]})")
            rows.append({"solver": solver, "steps": steps, "nfe": nfe,
                         "num_fake": activs.shape[0], "fid": f"{fid:.4f}"})

    os.makedirs(os.path.dirname(os.path.abspath(sets.output_csv)), exist_ok=True)
    with open(sets.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["solver", "steps", "nfe", "num_fake", "fid"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"FID sweep complete. Summary written to: {sets.output_csv}")
    print("=" * 60)
    print(f"{'solver':<8}{'steps':>7}{'nfe':>7}{'num_fake':>10}{'fid':>12}")
    for r in rows:
        print(f"{r['solver']:<8}{r['steps']:>7}{r['nfe']:>7}{r['num_fake']:>10}{str(r['fid']):>12}")


if __name__ == "__main__":
    main()
