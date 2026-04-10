"""Dataset factory for voxel-based training datasets.

This module is the main entry point used by training code to construct dataset
instances from configuration values.

It currently supports:
- ``lidc`` for preprocessed LIDC-IDRI NIfTI volumes
- ``rad_chestct`` for preprocessed RAD-ChestCT ``.npz`` volumes

The factory expects:
- ``dataroot`` pointing to the preprocessed dataset directory
- ``train_list`` and ``val_list`` pointing to split text files
- ``roi_size`` describing the target training resolution

Based on ``task``, it dispatches to the appropriate dataset loader and applies
optional normalization and augmentation settings consistently.
"""

import os

from datasets.lidc import LIDCVolumes
from datasets.rad_chest import RADChestCTDataset


def get_voxel_dataset(
    dataroot,
    task="rad_chestct",
    roi_size=(128, 128, 128),
    data_type="train",
    train_list=None,
    val_list=None,
    augment=False,
    normalize=False,
    rank=0,
):
    """Construct a voxel dataset for the requested task and split.

    Parameters
    ----------
    dataroot : str
        Directory containing the preprocessed dataset files.
    task : str, optional
        Dataset type — ``"lidc"`` or ``"rad_chestct"`` (default ``"rad_chestct"``).
    roi_size : tuple of int, optional
        Volume edge length ``(H, W, D)``; only the first element is used
        (default ``(128, 128, 128)``).
    data_type : str, optional
        Split — ``"train"`` or ``"val"`` (default ``"train"``).
    train_list : str or None
        Path to the training split ``.txt`` file.
    val_list : str or None
        Path to the validation split ``.txt`` file.
    augment : bool, optional
        Enable random flip augmentation (default ``False``).
    normalize : bool, optional
        Apply ``2*x - 1`` voxel normalization (default ``False``).
    rank : int, optional
        Process rank for distributed training; controls progress output
        (default ``0``).

    Returns
    -------
    torch.utils.data.Dataset
        Constructed dataset instance for the requested task and split.
    """
    if task not in {"lidc", "rad_chestct"}:
        raise ValueError("Invalid task. Supported tasks: 'lidc' and 'rad_chestct'.")
    if data_type not in {"train", "val"}:
        raise ValueError(f"Invalid data_type: {data_type}. Supported types: 'train', 'val'.")
    if train_list is None or val_list is None:
        raise ValueError("train_list and val_list must both be provided.")

    if rank == 0:
        print(f"Loading {task} dataset...")

    img_size = roi_size[0]
    assert img_size in [64, 128, 256], "Dataset only supports image sizes: 64, 128, 256"

    split_file = train_list if data_type == "train" else val_list
    split_file = os.path.expanduser(split_file)

    if task == "lidc":
        return LIDCVolumes(
            directory=dataroot,
            normalize=(lambda x: 2 * x - 1) if normalize else None,
            mode=data_type,
            img_size=img_size,
            split_file=split_file,
            augment=augment,
            rank=rank,
        )

    return RADChestCTDataset(
        directory=dataroot,
        mode=data_type,
        img_size=img_size,
        split_file=split_file,
        normalize=(lambda x: 2 * x - 1) if normalize else None,
        augment=augment,
        rank=rank,
    )
