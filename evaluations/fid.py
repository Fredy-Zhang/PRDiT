"""3D Fréchet Inception Distance (FID) computation for CT volumes.

Extracts deep features from real and generated volumes using a pretrained
3D ResNet-50 backbone, then computes the Fréchet distance between the two
feature distributions.

Usage::

    python evaluations/fid.py \\
        --dataset rad_chestCT \\
        --img_size 128 \\
        --data_root_real /path/to/real \\
        --data_root_fake /path/to/fake \\
        --pretrain_path resnet_50.pth \\
        --path_to_activations /tmp/acts \\
        --num_samples 1000
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scipy import linalg

from datasets.rad_chest import RADChestCTDataset
from model import generate_model
from datasets.lidc import LIDCVolumes


def get_feature_extractor(sets):
    """Load a 3D ResNet-50 feature extractor from a pretrained checkpoint.

    Parameters
    ----------
    sets : argparse.Namespace
        CLI options; must expose ``pretrain_path`` and ``dims``.

    Returns
    -------
    torch.nn.Module
        Feature extractor in eval mode.
    """
    model, _ = generate_model(sets)
    checkpoint = torch.load(sets.pretrain_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    sets.dims = 2048
    model.eval()
    print("Initialized feature extractor with pretrained weights (2048-D).")
    return model


def get_activations(model, data_loader, sets, device):
    """Extract feature activations from ``data_loader`` up to ``sets.num_samples``.

    Parameters
    ----------
    model : torch.nn.Module
        Feature extractor (eval mode).
    data_loader : torch.utils.data.DataLoader
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

    for batch in data_loader:
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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Fréchet distance between two multivariate Gaussians.

    Implements:
    ``d² = ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁·Σ₂))``

    Parameters
    ----------
    mu1 : array-like, shape (d,)
        Mean of the first distribution (real features).
    sigma1 : array-like, shape (d, d)
        Covariance of the first distribution.
    mu2 : array-like, shape (d,)
        Mean of the second distribution (generated features).
    sigma2 : array-like, shape (d, d)
        Covariance of the second distribution.
    eps : float, optional
        Small regularisation added to the diagonal when the product of
        covariances is singular (default ``1e-6``).

    Returns
    -------
    float
        Fréchet distance (lower is better; 0 means identical distributions).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariance matrices have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'FID: singular product, adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def process_feature_vecs(activations):
    """Compute the sample mean and covariance from a feature matrix.

    Parameters
    ----------
    activations : numpy.ndarray, shape (N, d)
        Feature vectors collected from the dataset.

    Returns
    -------
    mu : numpy.ndarray, shape (d,)
        Sample mean.
    sigma : numpy.ndarray, shape (d, d)
        Sample covariance.
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def check_data_range_compatibility(real_data, fake_data, dataset_name, num_samples=5, tolerance=0.1):
    print(f"\n{'='*60}")
    print(f"Checking data range compatibility for {dataset_name}")
    print(f"{'='*60}")

    def _indices(dataset):
        return np.linspace(0, len(dataset) - 1, min(num_samples, len(dataset)), dtype=int)

    def _get_sample(dataset, idx, dataset_name, mode):
        if dataset_name == 'rad_chestCT':
            return dataset[idx][0]
        else:
            item = dataset[idx]
            if mode == 'real':
                return item['image'] if isinstance(item, dict) else item
            else:
                return item[0]  # (image, name) tuple

    def _minmax(dataset, indices, dataset_name, mode):
        mins, maxs = [], []
        for idx in indices:
            s = _get_sample(dataset, idx, dataset_name, mode)
            if isinstance(s, torch.Tensor):
                mins.append(s.min().item())
                maxs.append(s.max().item())
            else:
                mins.append(float(np.min(s)))
                maxs.append(float(np.max(s)))
        return min(mins), max(maxs)

    real_min, real_max = _minmax(real_data, _indices(real_data), dataset_name, 'real')
    fake_min, fake_max = _minmax(fake_data, _indices(fake_data), dataset_name, 'fake')

    print(f"Real data range:  min={real_min:.5f}, max={real_max:.5f}")
    print(f"Fake data range:  min={fake_min:.5f}, max={fake_max:.5f}")

    min_diff = abs(real_min - fake_min)
    max_diff = abs(real_max - fake_max)
    print(f"Difference:       min_diff={min_diff:.5f}, max_diff={max_diff:.5f}  (tolerance={tolerance})")

    if min_diff > tolerance or max_diff > tolerance:
        raise AssertionError(
            f"Data range mismatch: real=[{real_min:.5f},{real_max:.5f}], "
            f"fake=[{fake_min:.5f},{fake_max:.5f}]. "
            f"Normalize fake data to match real data range."
        )
    print(f"Data ranges are compatible.")
    print(f"{'='*60}\n")


def set_randomness(seed):
    """Fix all global random seeds for reproducible evaluation.

    Parameters
    ----------
    seed : int
        Seed applied to PyTorch, NumPy, Python's ``random``, and CUDA.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_opts():
    """Build and parse the CLI argument parser for 3D FID evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='rad_chestCT | lidc-idri')
    parser.add_argument('--img_size', required=True, type=int, help='Image size')
    parser.add_argument('--data_root_real', required=True, type=str, help='Path to real data')
    parser.add_argument('--data_root_fake', required=True, type=str, help='Path to fake/generated data')
    parser.add_argument('--split_dir', default=None, type=str, help='Directory containing train.txt and val.txt (rad_chestCT real mode)')
    parser.add_argument('--pretrain_path', required=True, type=str, help='Path to pretrained 3D ResNet-50 weights')
    parser.add_argument('--path_to_activations', required=True, type=str, help='Directory to save activation stats')
    parser.add_argument('--num_samples', default=1000, type=int, help='Number of volumes for FID computation')
    parser.add_argument('--reuse_real_stats', action='store_true',
                        help='Reuse cached mu_real.npy/sigma_real.npy in --path_to_activations and skip the real preload (for sampler ablations sharing one real reference)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--manual_seed', default=192, type=int)
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
    return parser.parse_args()


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "normal"
    sets.dims = 2048

    device = torch.device('cpu' if sets.no_cuda else f'cuda:{sets.gpu_id}')
    set_randomness(sets.manual_seed)

    print("Loading feature extractor...")
    model = get_feature_extractor(sets)
    model = model.to(device)

    def _build_dataset(root, mode):
        if sets.dataset == 'rad_chestCT':
            kw = {"split_dir": sets.split_dir} if mode == "real" else {}
            return RADChestCTDataset(root, img_size=sets.img_size, mode=mode, **kw)
        elif sets.dataset == 'lidc-idri':
            kw = {"split_dir": sets.split_dir} if mode == "real" else {}
            return LIDCVolumes(root, img_size=sets.img_size, mode=mode, **kw)
        raise ValueError(f"Unsupported dataset: {sets.dataset}. Use 'rad_chestCT' or 'lidc-idri'.")

    mu_real_path = os.path.join(sets.path_to_activations, 'mu_real.npy')
    sigma_real_path = os.path.join(sets.path_to_activations, 'sigma_real.npy')
    reuse_real = sets.reuse_real_stats and os.path.exists(mu_real_path) and os.path.exists(sigma_real_path)

    print("Initializing dataloaders...")
    fake_data = _build_dataset(sets.data_root_fake, "fake")

    if reuse_real:
        # The real reference is identical across sampler ablations; skip the
        # expensive real preload + feature extraction and reuse cached stats.
        print(f"Reusing cached real stats from {sets.path_to_activations} (skipping real preload).")
        mu_real, sigma_real = np.load(mu_real_path), np.load(sigma_real_path)
        print(f"Fake: {len(fake_data)} samples")
    else:
        real_data = _build_dataset(sets.data_root_real, "real")
        print(f"Real: {len(real_data)} samples | Fake: {len(fake_data)} samples")
        check_data_range_compatibility(real_data, fake_data, sets.dataset, num_samples=5, tolerance=0.1)
        real_loader = DataLoader(real_data, batch_size=sets.batch_size, shuffle=False,
                                 num_workers=sets.num_workers, pin_memory=False)
        print(f"Computing 3D FID with {sets.num_samples} samples (seed={sets.manual_seed})...")
        print("Extracting real activations...")
        activations_real = get_activations(model, real_loader, sets, device)
        mu_real, sigma_real = process_feature_vecs(activations_real)
        np.save(mu_real_path, mu_real)
        np.save(sigma_real_path, sigma_real)
        print(f"Real activations: {activations_real.shape}")

    fake_loader = DataLoader(fake_data, batch_size=sets.batch_size, shuffle=False,
                             num_workers=sets.num_workers, pin_memory=False)

    print("Extracting fake activations...")
    activations_fake = get_activations(model, fake_loader, sets, device)
    mu_fake, sigma_fake = process_feature_vecs(activations_fake)
    np.save(os.path.join(sets.path_to_activations, 'mu_fake.npy'), mu_fake)
    np.save(os.path.join(sets.path_to_activations, 'sigma_fake.npy'), sigma_fake)
    print(f"Fake activations: {activations_fake.shape}")

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print("=" * 50)
    print(f"3D FID Score: {fid:.4f}")
    print("=" * 50)
