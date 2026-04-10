#!/usr/bin/env python3
"""3D Maximum Mean Discrepancy (MMD) computation for CT volumes.

Extracts deep features from real and generated volumes using a pretrained
3D ResNet-50 backbone, then computes the MMD with an RBF kernel whose
bandwidth is chosen by the median heuristic.

Usage::

    python evaluations/mmd.py \\
        --dataset rad_chestCT \\
        --img_size 128 \\
        --data_root_real /path/to/real \\
        --data_root_fake /path/to/fake \\
        --pretrain_path resnet_50.pth \\
        --path_to_activations /tmp/acts \\
        --num_samples 1000
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datasets.rad_chest import RADChestCTDataset
from model import generate_model
from datasets.lidc import LIDCVolumes
from fid import check_data_range_compatibility, set_randomness


def parse_opts():
    """Build and parse the CLI argument parser for 3D MMD evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='rad_chestCT | lidc-idri')
    parser.add_argument('--data_root_real', required=True, type=str)
    parser.add_argument('--data_root_fake', required=True, type=str)
    parser.add_argument('--split_dir', default=None, type=str, help='Directory containing train.txt and val.txt (rad_chestCT real mode)')
    parser.add_argument('--pretrain_path', required=True, type=str, help='Path to pretrained 3D ResNet-50 weights')
    parser.add_argument('--img_size', required=True, type=int)
    parser.add_argument('--path_to_activations', required=True, type=str, help='Directory to save activation stats')
    parser.add_argument('--num_samples', default=1000, type=int, help='Number of volumes for MMD computation')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--manual_seed', default=1, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--gpu_id', default=0, type=int)
    # ResNet architecture args (keep defaults matching pretrained weights)
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--model_depth', default=50, type=int)
    parser.add_argument('--resnet_shortcut', default='B', type=str)
    parser.add_argument('--input_D', default=128, type=int)
    parser.add_argument('--input_H', default=128, type=int)
    parser.add_argument('--input_W', default=128, type=int)
    parser.add_argument('--n_seg_classes', default=2, type=int)
    parser.add_argument('--new_layer_names', default=['conv_seg'], type=list)
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--dims', default=2048, type=int)
    return parser.parse_args()


def get_feature_extractor(sets):
    """Load a 3D ResNet-50 feature extractor from a pretrained checkpoint.

    Parameters
    ----------
    sets : argparse.Namespace
        CLI options; must expose ``pretrain_path``.

    Returns
    -------
    torch.nn.Module
        Feature extractor in eval mode.
    """
    model, _ = generate_model(sets)
    ckpt = torch.load(sets.pretrain_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    print("Initialized feature extractor with pretrained weights (2048-D).")
    return model


def get_activations(model, loader, sets, device):
    """Extract feature activations from ``loader`` up to ``sets.num_samples``.

    Parameters
    ----------
    model : torch.nn.Module
        Feature extractor (eval mode).
    loader : torch.utils.data.DataLoader
        Yields batches of volumes; accepts ``dict``, ``tuple``, or raw tensors.
    sets : argparse.Namespace
        Must expose ``num_samples`` (int) and ``dims`` (int).
    device : torch.device
        Device on which inference runs.

    Returns
    -------
    numpy.ndarray, shape (N, dims)
        Collected feature vectors (float64).
    """
    activs = np.zeros((sets.num_samples, sets.dims), dtype=np.float64)
    idx = 0
    first_batch = True

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            img = batch[0]
        elif isinstance(batch, dict):
            img = batch['image']
        else:
            img = batch

        img = img.to(device)
        with torch.no_grad():
            feat = model(img)

        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)

        if first_batch:
            assert feat.shape[1] == sets.dims, \
                f"Feature dim mismatch: got {feat.shape[1]} vs sets.dims={sets.dims}"
            first_batch = False

        b = feat.shape[0]
        end = min(idx + b, sets.num_samples)
        activs[idx:end] = feat[:end - idx].cpu().numpy().astype(np.float64)
        idx = end
        if idx >= sets.num_samples:
            break

    if idx < sets.num_samples:
        print(f"Warning: collected {idx} samples instead of {sets.num_samples}")
        activs = activs[:idx]

    return activs


def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute the squared MMD between two feature sets using an RBF kernel.

    Parameters
    ----------
    x : torch.Tensor, shape (N, d)
        Features from the real distribution.
    y : torch.Tensor, shape (M, d)
        Features from the generated distribution.
    gamma : float
        RBF bandwidth parameter ``γ`` in ``exp(-γ‖u − v‖²)``.

    Returns
    -------
    torch.Tensor, scalar
        Estimated MMD² (may be slightly negative due to finite sampling).
    """
    xx = x @ x.t()
    yy = y @ y.t()
    xy = x @ y.t()
    x_norm = (x ** 2).sum(1, keepdim=True)
    y_norm = (y ** 2).sum(1, keepdim=True)
    K_xx = torch.exp(-gamma * (x_norm + x_norm.t() - 2 * xx))
    K_yy = torch.exp(-gamma * (y_norm + y_norm.t() - 2 * yy))
    K_xy = torch.exp(-gamma * (x_norm + y_norm.t() - 2 * xy))
    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd2


def main():
    sets = parse_opts()
    sets.target_type = "normal"

    device = torch.device('cpu' if sets.no_cuda else f'cuda:{sets.gpu_id}')
    set_randomness(sets.manual_seed)

    print("Loading feature extractor...")
    model = get_feature_extractor(sets)
    model = model.to(device)

    print("Initializing dataloaders...")
    if sets.dataset == 'rad_chestCT':
        real_ds = RADChestCTDataset(sets.data_root_real, img_size=sets.img_size, mode="real", split_dir=sets.split_dir)
        fake_ds = RADChestCTDataset(sets.data_root_fake, img_size=sets.img_size, mode="fake")
        print(f"Real: {len(real_ds)} samples | Fake: {len(fake_ds)} samples")
    elif sets.dataset == 'lidc-idri':
        real_ds = LIDCVolumes(sets.data_root_real, img_size=sets.img_size, mode='real', split_dir=sets.split_dir)
        fake_ds = LIDCVolumes(sets.data_root_fake, img_size=sets.img_size, mode='fake')
        print(f"Real: {len(real_ds)} samples | Fake: {len(fake_ds)} samples")
    else:
        raise ValueError(f"Unsupported dataset: {sets.dataset}. Use 'rad_chestCT' or 'lidc-idri'.")

    check_data_range_compatibility(real_ds, fake_ds, sets.dataset, num_samples=5, tolerance=0.1)

    real_loader = DataLoader(real_ds, batch_size=sets.batch_size, shuffle=False,
                             num_workers=sets.num_workers, pin_memory=False)
    fake_loader = DataLoader(fake_ds, batch_size=sets.batch_size, shuffle=False,
                             num_workers=sets.num_workers, pin_memory=False)

    print(f"Computing 3D MMD with {sets.num_samples} samples (seed={sets.manual_seed})...")

    print("Extracting real activations...")
    act_real = get_activations(model, real_loader, sets, device)
    np.save(os.path.join(sets.path_to_activations, 'mmd_act_real.npy'), act_real)
    print(f"Real activations: {act_real.shape}")

    print("Extracting fake activations...")
    act_fake = get_activations(model, fake_loader, sets, device)
    np.save(os.path.join(sets.path_to_activations, 'mmd_act_fake.npy'), act_fake)
    print(f"Fake activations: {act_fake.shape}")

    x = torch.from_numpy(act_real).to(device)
    y = torch.from_numpy(act_fake).to(device)

    # Median heuristic for RBF bandwidth
    with torch.no_grad():
        dists = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(-1).view(-1)
        gamma = 1.0 / (2.0 * dists.median().item())

    mmd2 = compute_mmd(x, y, gamma=gamma)
    mmd = torch.sqrt(torch.clamp(mmd2, min=0.0))

    print("=" * 50)
    print(f"3D MMD Score: {mmd.item():.6e}")
    print(f"3D MMD²:      {mmd2.item():.6e}")
    print(f"Gamma (median heuristic): {gamma:.6e}")
    print("=" * 50)


if __name__ == "__main__":
    main()
