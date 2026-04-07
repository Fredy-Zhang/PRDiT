import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from datasets.runtime import get_rank_logger, read_split_file

"""RAD-ChestCT dataset loader.

This module loads preprocessed RAD-ChestCT volumes referenced by split text
files. Split entries may be absolute paths, relative paths, or filename-only
entries such as `trn07793.npz`.

Expected preprocessed file format:
- Compressed `.npz` files containing a normalized 3D volume.
- The loader accepts either a `volume` key or a `ct` key for compatibility.

The loader:
- resolves split entries against the dataset directory and split-file directory
- loads one volume at a time from disk
- converts it to channel-first tensor format
- downsamples to 128 or 64 when requested
- applies optional normalization and random flip augmentation

This is designed to work with outputs produced by
`scripts/preprocess_rad_chestct.py` and split files produced by
`scripts/split_train_val.py`.
"""


class RADChestCTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        mode="train",
        img_size=256,
        split_file=None,
        normalize=None,
        augment=False,
        rank=0,
    ):
        """
        RAD-ChestCT dataset loader for preprocessed volumes.

        Args:
            directory: Path to preprocessed `.npz` files with a `volume` key
            mode: Split type ('train' | 'val')
            img_size: Target image size (256, 128, or 64)
            split_file: Path to `train.txt` or `val.txt`
            rank: Process rank for distributed training
        """
        super().__init__()
        assert img_size in (256, 128, 64), "img_size must be 256, 128, or 64"
        assert mode in ("train", "val"), "mode must be 'train' or 'val'"
        
        self.rank = rank
        self.logger = get_rank_logger("RADChestCT", rank)
        
        self.directory = str(Path(directory).expanduser())
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.rank = rank
        self.normalize = normalize
        if split_file is None:
            raise ValueError("split_file must be provided.")
        self.split_path = str(Path(split_file).expanduser())

        if rank == 0:
            self.logger.info(f"Initializing RAD-ChestCT dataset in {mode} mode")
            self.logger.info(f"Preprocessed data directory: {directory}")
            self.logger.info(f"Split file: {self.split_path}")
            self.logger.info(f"Image size: {img_size}x{img_size}x{img_size}")
            self.logger.info(f"Using custom normalization: {normalize is not None}")
            self.logger.info(f"Using augmentation: {augment}")

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(
                f"Missing split file: {self.split_path}."
            )

        entries = read_split_file(self.split_path)
        if not entries:
            raise ValueError(f"Split file is empty: {self.split_path}")
        self.file_paths = self._resolve_split_entries(entries)

        if rank == 0:
            self.logger.info(f"Loaded {len(self.file_paths)} file paths from split")

        if rank == 0 and len(self.file_paths) > 0:
            first_sample = self._load_volume(self.file_paths[0]).cpu().numpy()
            self.logger.info("Data Statistics (first sample):")
            self.logger.info("--- Preprocessed RAD-ChestCT [0,1] range ---")
            self.logger.info(f"Min:  {np.min(first_sample):.6f}")
            self.logger.info(f"Max:  {np.max(first_sample):.6f}")
            self.logger.info(f"Mean: {np.mean(first_sample):.6f}")
            self.logger.info(f"Std:  {np.std(first_sample):.6f}")

        if rank == 0:
            self.logger.info(f"Final {mode} size: {len(self.file_paths)}")

    def _resolve_split_entries(self, entries):
        """Resolve absolute, relative, and filename-only paths from the split file."""
        resolved = []
        split_dir = os.path.dirname(self.split_path)
        for entry in entries:
            if os.path.isabs(entry):
                resolved.append(entry)
                continue

            directory_candidate = os.path.join(self.directory, entry)
            split_dir_candidate = os.path.join(split_dir, entry)

            if os.path.exists(directory_candidate):
                resolved.append(directory_candidate)
            elif os.path.exists(split_dir_candidate):
                resolved.append(split_dir_candidate)
            else:
                resolved.append(directory_candidate)
        missing = [path for path in resolved if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Split file {self.split_path} references missing files. First missing file: {missing[0]}"
            )
        return resolved

    def downsample(self, image: torch.Tensor) -> torch.Tensor:
        """Downsample a volume tensor to the requested training resolution."""
        if self.img_size == 128:
            return nn.AvgPool3d(2, 2)(image)
        if self.img_size == 64:
            return nn.AvgPool3d(2, 2)(nn.AvgPool3d(2, 2)(image))
        return image

    def _load_volume(self, file_path: str) -> torch.Tensor:
        """Load one preprocessed RAD-ChestCT volume from disk."""
        data = np.load(file_path)
        if "volume" in data:
            array = data["volume"]
        elif "ct" in data:
            array = data["ct"]
        else:
            raise KeyError(
                f"Expected preprocessed file {file_path} to contain a 'volume' or 'ct' key."
            )
        image = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        # image = self.downsample(image)
        if self.normalize:
            image = self.normalize(image)
        return image

    def __len__(self):
        """Return the number of samples in the current split."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Return one volume sample, with optional random flip augmentation."""
        image = self._load_volume(self.file_paths[idx])

        if self.augment:
            prob = np.random.rand()
            if prob > 0.5:
                image = torch.flip(image, dims=[-1])

        return {"image": image}
