"""RAD-ChestCT dataset loader.

This module loads preprocessed RAD-ChestCT volumes referenced by split text
files. Split entries may be absolute paths, relative paths, or filename-only
entries such as ``trn07793.npz``.

Expected preprocessed file format:
- Compressed ``.npz`` files containing a normalized 3D volume.
- The loader accepts either a ``volume`` key or a ``ct`` key for compatibility.

The loader:

- resolves split entries against the dataset directory and split-file directory
- loads one volume at a time from disk
- converts it to channel-first tensor format
- downsamples to 128 or 64 when requested
- applies optional normalization and random flip augmentation

This is designed to work with outputs produced by
``scripts/preprocess_rad_chestct.py`` and split files produced by
``scripts/split_train_val.py``.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from datasets.runtime import get_rank_logger, read_split_file


class RADChestCTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        mode="train",
        img_size=256,
        split_file=None,
        split_dir=None,
        normalize=None,
        augment=False,
        rank=0,
    ):
        """Initialize the RAD-ChestCT dataset.

        Parameters
        ----------
        directory : str or Path
            Path to the directory containing preprocessed ``.npz`` volumes.
        mode : str, optional
            One of ``'train'``, ``'val'``, ``'real'``, or ``'fake'``
            (default ``'train'``).
        img_size : int, optional
            Target voxel edge length — 256, 128, or 64 (default ``256``).
        split_file : str or list of str, optional
            Explicit split file path for ``'train'`` / ``'val'``, or list of
            paths for ``'real'``. Takes precedence over ``split_dir``.
        split_dir : str or Path, optional
            Directory containing ``train.txt`` and ``val.txt``; used for
            ``mode='real'`` when ``split_file`` is not given.
        normalize : callable or None, optional
            Voxel normalization applied after loading (default ``None``).
        augment : bool, optional
            Apply random left-right flip augmentation (default ``False``).
        rank : int, optional
            Process rank; only rank 0 emits progress output (default ``0``).
        """
        super().__init__()
        assert img_size in (256, 128, 64), "img_size must be 256, 128, or 64"
        assert mode in ("train", "val", "real", "fake"), \
            "mode must be 'train', 'val', 'real', or 'fake'"

        self.rank = rank
        self.logger = get_rank_logger("RADChestCT", rank)

        self.directory = str(Path(directory).expanduser())
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize

        if rank == 0:
            self.logger.info(f"Initializing RAD-ChestCT dataset in '{mode}' mode")
            self.logger.info(f"Directory: {self.directory}")
            self.logger.info(f"Image size: {img_size}x{img_size}x{img_size}")

        if mode == "fake":
            # Collect all .npz files directly under the directory
            self.file_paths = sorted(
                str(p) for p in Path(self.directory).glob("*.npz")
            )
            if not self.file_paths:
                raise FileNotFoundError(
                    f"No .npz files found in fake directory: {self.directory}"
                )
        elif mode == "real":
            # Resolve split files: explicit list > split_dir/train.txt + split_dir/val.txt
            if not split_file:
                if split_dir is None:
                    raise ValueError(
                        "For mode='real', provide either split_dir (directory containing "
                        "train.txt and val.txt) or an explicit split_file list."
                    )
                split_dir = str(Path(split_dir).expanduser())
                split_file = [
                    os.path.join(split_dir, "train.txt"),
                    os.path.join(split_dir, "val.txt"),
                ]
            self.file_paths = self._load_real_splits(split_file)
        else:
            # 'train' or 'val': use the provided split file (single path string)
            if split_file is None:
                raise ValueError("split_file must be provided for mode='train' or 'val'.")
            self.split_path = str(Path(split_file).expanduser())
            if not os.path.exists(self.split_path):
                raise FileNotFoundError(f"Missing split file: {self.split_path}")
            entries = read_split_file(self.split_path)
            if not entries:
                raise ValueError(f"Split file is empty: {self.split_path}")
            self.file_paths = self._resolve_split_entries(entries)

        if rank == 0:
            self.logger.info(f"Loaded {len(self.file_paths)} samples")

    def _load_real_splits(self, split_files):
        """Merge multiple split files to build the full ground-truth file list."""
        entries = []
        for path in split_files:
            path = str(Path(path).expanduser())
            if not os.path.exists(path):
                raise FileNotFoundError(f"Split file not found: {path}")
            loaded = read_split_file(path)
            entries.extend(loaded)
            self.logger.info(f"Loaded {len(loaded)} entries from {path}")
        if not entries:
            raise ValueError(f"All provided split files are empty: {split_files}")
        return self._resolve_split_entries(entries)

    def _resolve_split_entries(self, entries):
        """Resolve absolute, relative, and filename-only paths from the split file."""
        resolved = []
        split_dir = os.path.dirname(getattr(self, "split_path", self.directory))
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

        if self.augment and np.random.rand() > 0.5:
            image = torch.flip(image, dims=[-1])

        return {"image": image}
