"""Multi-Scale SSIM (MS-SSIM) diversity evaluation for generated CT volumes.

Computes the mean MS-SSIM across all unique pairs of generated volumes using
MONAI's ``MultiScaleSSIMMetric``.  Lower mean MS-SSIM indicates higher sample
diversity.

Usage::

    python evaluations/ms_ssim.py \\
        --sample_dir /path/to/generated \\
        --dataset rad_chestCT \\
        --img_size 128
"""

import argparse
import sys

import numpy as np
import torch

sys.path.append(".")
sys.path.append("..")

from generative.metrics import MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets.rad_chest_fast import RadChestCTDataset
from datasets.lidc import LIDCVolumes
from datasets.rad_chest import RADChestCTDataset

def parse_args():
    """Build and parse the CLI argument parser for MS-SSIM evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--sample_dir", type=str, required=True, help="Location of the samples to evaluate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of loader workers")
    parser.add_argument("--dataset", choices=['rad_chestCT','lidc-idri'], required=True, help="Dataset (rad_chestCT)")
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process. If None, process all samples.")

    args = parser.parse_args()
    return args


def main(args):
    """Run the MS-SSIM diversity evaluation loop.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from :func:`parse_args`.
    """
    set_determinism(seed=args.seed)
    #print_config()

    if args.dataset == 'rad_chestCT':
        dataset_1 = RADChestCTDataset(args.sample_dir, img_size=args.img_size, mode="fake", preprocess="rs")
        dataset_2 = RADChestCTDataset(args.sample_dir, img_size=args.img_size, mode="fake", preprocess="rs")

    elif args.dataset == 'lidc-idri':
        dataset_1 = LIDCVolumes(args.sample_dir, img_size=args.img_size, mode='fake')
        dataset_2 = LIDCVolumes(args.sample_dir, img_size=args.img_size, mode='fake')
    else:
        raise NotImplementedError("Dataloader for this dataset is not implemented. Use 'rad_chestCT' or 'lidc-idri'.")

    if args.max_samples is not None:
        n = min(len(dataset_1), args.max_samples)
        indices = list(range(n))
        dataset_1 = Subset(dataset_1, indices)
        dataset_2 = Subset(dataset_2, indices)
    print(f"Number of samples: {len(dataset_1)}")
        
    dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False, num_workers=args.num_workers)
    dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=3)

    print("Computing MS-SSIM (this takes a while)...")
    ms_ssim_list = []
    pbar = tqdm(enumerate(dataloader_1), total=len(dataloader_1))
    for step, batch in pbar:
        img = batch[0]
        for batch2 in dataloader_2:
            img2 = batch2 [0]
            if batch[1] == batch2[1]:
                continue
            ms_ssim_list.append(ms_ssim(img.to(device), img2.to(device)).item())
        pbar.update()

    ms_ssim_list = np.array(ms_ssim_list)
    print("Calculated MS-SSIMs. Computing mean ...")
    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
